import itertools
import os
import pickle
import random
from argparse import Namespace

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import rich
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class RNA_DS(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        bpp_path: str,
        loop: bool = False,
        infer: bool = False,
        flip: int = 0,
        noise_bpp: float = 0,
    ):
        if "sequence" in df:
            df = df.rename(columns={"sequence": "seq", "sequence_id": "seq_id"})
        self.df = df
        self.infer = infer
        self.p_flip = flip / 2
        self.len_max = self.df.seq.apply(len).max()
        self.kmap_seq = {x: i for i, x in enumerate("_AUGC")}
        self.kmap_loop = {x: i for i, x in enumerate("_SMIBHEX")} if loop else None
        self.bpp_path = bpp_path
        self.noise_bpp = noise_bpp

    def __len__(self):
        return len(self.df)

    def _pad(self, x: torch.Tensor):
        if x.ndim == 2 and x.size(0) == x.size(1):
            return F.pad(x, [0, self.len_max - len(x)] * 2)
        z = [0] * (1 + (x.ndim - 1) * 2)
        v = float("nan") if x.dtype == torch.float else 0
        return F.pad(x, z + [self.len_max - len(x)], value=v)

    def pad(self, d: dict):
        return {k: self._pad(v) for k, v in d.items()}

    def _flip(self, x: torch.Tensor):
        if x.ndim == 2 and x.size(0) == x.size(1):
            return x.flip([0, 1])
        return x.flip(0)

    def flip(self, d: dict):
        return {k: self._flip(v) for k, v in d.items()}

    def get_seq(self, r):
        return {"seq": torch.IntTensor([self.kmap_seq[_] for _ in r.seq])}

    def get_mask(self, r):
        mask = torch.zeros(self.len_max)
        mask[: len(r.seq)] = 1
        return {"mask": mask}

    def get_react(self, r):
        react = torch.Tensor(np.array([r.react_DMS, r.react_2A3]))
        return {"react": react.transpose(0, 1).clip(0, 1)}

    def get_error(self, r):
        error = torch.Tensor(np.array([r.error_DMS, r.error_2A3]))
        return {"error": error.transpose(0, 1).clip(0, 1)}

    def get_loop(self, r):
        if not self.kmap_loop:
            return {}
        loop = [self.kmap_loop[_] for _ in r.eterna_loop]  # r.contra_loop
        return {"loop": torch.LongTensor(loop)}

    def get_bpp(self, r):
        with open(f"{self.bpp_path}/{r.seq_id}.pkl", "rb") as f:
            bpp = np.array(pickle.load(f)["bpp"].todense())
            bpp = torch.HalfTensor(bpp).clip(0, 1)
        if self.noise_bpp:
            bpp += (self.noise_bpp**0.5) * torch.randn_like(bpp)
        return {"bpp": bpp}

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        out = self.get_seq(r) | self.get_loop(r) | self.get_bpp(r)
        if not self.infer:
            out = out | self.get_react(r)
        if self.p_flip and random.random() <= self.p_flip:
            out = self.flip(out)
        return self.pad(out) | self.get_mask(r)


class RNA_DM(pl.LightningDataModule):
    def __init__(
        self,
        hp: Namespace,
        n_workers: int = 0,
        df_infer: pd.DataFrame | None = None,
        df_train: pd.DataFrame | None = None,
        df_valid: pd.DataFrame | None = None,
    ):
        super().__init__()
        self.df_train, self.df_valid, self.df_infer = df_train, df_valid, df_infer
        self.bpp_path_infer = "../data/preprocess_eterna/test_sparse_bpps"
        self.bpp_path_train = "../data/preprocess_eterna/test_sparse_bpps"
        self.bpp_path_valid = "../data/preprocess_eterna/train_sparse_bpps"
        if df_valid is not None:
            args = df_valid, hp.fold, hp.n_folds, hp.seed
            _, self.df_valid = self.fold_splitter(*args)
        elif df_train is not None:
            args = df_train, hp.fold, hp.n_folds, hp.seed
            self.df_train, self.df_valid = self.fold_splitter(*args)
            self.bpp_path_train = "../data/preprocess_eterna/train_sparse_bpps"
        self.sn_bias_sched, self._sn_bias = hp.sn_bias_sched, 0
        self.flip = hp.aug_flip, hp.val_flip, hp.tta_flip
        self.noise = hp.noise_bpp
        self.kwargs = {"batch_size": hp.batch_size, "num_workers": n_workers}

    @staticmethod
    def fold_splitter(df: pd.DataFrame, fold_n: int, n_folds: int, seed: int):
        fname = f"cache/{n_folds}_{seed}.parquet"
        try:
            folds = pd.read_parquet(fname).values.tolist()
        except:
            with rich.get_console().status("Splitting Folds..."):
                folds = StratifiedGroupKFold(n_folds, True, seed)
                folds = list(folds.split(df, df.seq.apply(len), df.seq_id))
                os.makedirs("cache", exist_ok=True)
                pd.DataFrame(folds).to_parquet(fname)
        fold = folds[fold_n]
        train_df, val_df = df.iloc[fold[0]], df.iloc[fold[1]]
        val_df = val_df[(val_df.SN_DMS >= 1) & (val_df.SN_2A3 >= 1)]
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    def _get_bias(self):
        if self.trainer.current_epoch in self.sn_bias_sched:
            self._sn_bias = self.sn_bias_sched[self.trainer.current_epoch]
        return self._sn_bias

    def train_dataloader(self):
        assert self.df_train is not None
        df_train = self.df_train
        train_sn = df_train[["SN_DMS", "SN_2A3"]].clip(0, 1).sum(1).values
        wt = (train_sn + self._get_bias()).clip(0, 2)
        sampler = WeightedRandomShuffler(wt, int(2e5), False)
        kwargs = self.kwargs | {"sampler": sampler}
        args = df_train, self.bpp_path_train, False, False, self.flip[0], self.noise
        return DataLoader(RNA_DS(*args), **kwargs)

    def val_dataloader(self):
        assert self.df_valid is not None
        args = self.df_valid, self.bpp_path_valid, False, False
        dl = DataLoader(RNA_DS(*args), **self.kwargs)
        if self.flip[1]:
            dl = [dl, DataLoader(RNA_DS(*args, 2), **self.kwargs)]
        return dl

    def predict_dataloader(self):
        assert self.df_infer is not None
        args = self.df_infer, self.bpp_path_infer, False, True
        dl = DataLoader(RNA_DS(*args), **self.kwargs)
        if self.flip[2]:
            dl = [dl, DataLoader(RNA_DS(*args, 2), **self.kwargs)]
        return dl


class WeightedRandomShuffler(WeightedRandomSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=self.generator,
        )
        rand_tensor = rand_tensor[torch.randperm(rand_tensor.size(0))]
        yield from iter(rand_tensor.view(rand_tensor.size()).tolist())
