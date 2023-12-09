import logging
import os
import random
import warnings
from argparse import Namespace
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from einops._torch_specific import allow_ops_in_compiled_graph
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedGroupKFold
from apex.normalization import FusedLayerNorm, FusedRMSNorm


def sort_weight_decay_params(model: nn.Module) -> tuple[list, list]:
    # https://github.com/karpathy/minGPT
    whitelist = (nn.Linear, nn.GRU, nn.Conv1d, nn.Conv2d)
    blacklist = (nn.Embedding, nn.BatchNorm1d, FusedLayerNorm, FusedRMSNorm)
    decay, no_decay = set(), set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if "bias" in pn:
                no_decay.add(fpn)
            elif "weight" in pn and isinstance(m, whitelist):
                decay.add(fpn)
            elif "weight" in pn and isinstance(m, blacklist):
                no_decay.add(fpn)
    params = {pn: p for pn, p in model.named_parameters()}
    inter, union = (decay & no_decay), (decay | no_decay)
    missed = params.keys() - union
    assert len(inter) == 0, f"Duplicated: {str(inter)}"
    assert len(missed) == 0, f"Missed: {str(missed)}"
    decay, no_decay = sorted(list(decay)), sorted(list(no_decay))
    decay = [params[pn] for pn in sorted(list(decay))]
    no_decay = [params[pn] for pn in no_decay]
    return decay, no_decay


def pseudo_gen(df_infer: pd.DataFrame, preds: list[pd.DataFrame]) -> pd.DataFrame:
    react_DMS = [df.reactivity_DMS_MaP.to_numpy() for df in preds]
    react_2A3 = [df.reactivity_2A3_MaP.to_numpy() for df in preds]
    std_DMS, std_2A3 = np.std(react_DMS, axis=0), np.std(react_2A3, axis=0)
    react_DMS, react_2A3 = sum(react_DMS) / len(preds), sum(react_2A3) / len(preds)
    react_DMS[std_DMS > np.quantile(std_DMS, 0.75)] = np.nan
    react_2A3[std_2A3 > np.quantile(std_2A3, 0.75)] = np.nan
    chunk = lambda row, react: react[row["id_min"] : row["id_max"] + 1]
    df_infer["react_DMS"] = df_infer.apply(partial(chunk, react=react_DMS), axis=1)
    df_infer["react_2A3"] = df_infer.apply(partial(chunk, react=react_2A3), axis=1)
    df_infer = df_infer.rename(columns={"sequence": "seq", "sequence_id": "seq_id"})
    df_infer["SN_DMS"], df_infer["SN_2A3"] = 1, 1
    return df_infer[df_infer.seq.apply(len) <= 207].reset_index(drop=True)


def collate_preds(preds: list) -> dict:
    get_k, out = lambda k, ld: [p[k] for p in ld], {}
    preds = [preds] if isinstance(preds[0], dict) else preds
    for pred in preds:
        for k in pred[0]:
            v = torch.concat(get_k(k, pred))
            out[k] = out[k] + v if k in out else v
    out = {k: v / len(preds) for k, v in out.items()}
    return out


def submission(preds: dict, fn: str) -> None:
    def mutate_map(df: pd.DataFrame, fname: str):
        id1, id2 = 269545321, 269724007
        shape, font_size = (391, 457), 6
        pred_DMS = df[id1 : id2 + 1]["reactivity_DMS_MaP"].to_numpy()
        pred_2A3 = df[id1 : id2 + 1]["reactivity_2A3_MaP"].to_numpy()
        fig = plt.figure()
        plt.subplot(121)
        plt.title(f"reactivity_DMS_MaP", fontsize=font_size)
        plt.imshow(pred_DMS.reshape(*shape), vmin=0, vmax=1, cmap="gray_r")
        plt.subplot(122)
        plt.title(f"reactivity_2A3_MaP", fontsize=font_size)
        plt.imshow(pred_2A3.reshape(*shape), vmin=0, vmax=1, cmap="gray_r")
        plt.tight_layout()
        plt.savefig(fname, dpi=500)
        plt.clf()
        plt.close()

    react = preds["react"].float()
    react = pd.DataFrame(react, columns=["reactivity_DMS_MaP", "reactivity_2A3_MaP"])
    react.insert(0, "id", react.index)
    if fn:
        react.to_parquet(f"{fn}.parquet", index=False)
        mutate_map(react, f"{fn}.png")


def grid_search(hp: dict, hp_skips: list) -> list:
    def search(hp: dict) -> list:
        kl = [k for k, v in hp.items() if type(v) == list]
        if not kl:
            args = Namespace()
            for k, v in hp.items():
                setattr(args, k, v)
            return [args]
        out = []
        for item in hp[kl[0]]:
            hp_ = hp.copy()
            hp_[kl[0]] = item
            out += search(hp_)
        return out

    def skip(hp: Namespace, hp_skips: list) -> bool:
        if not hp_skips:
            return False
        for hp_skip in hp_skips:
            for k, v in hp_skip.items():
                v = [v] if not isinstance(v, list) else v
                if not getattr(hp, k) in v:
                    match = False
                    break
                match = True
            if match:
                return True
        return False

    return [_ for _ in search(hp) if not skip(_, hp_skips)]


def init_seed(seed: int, debug: bool):
    if not debug:
        warnings.filterwarnings("ignore")
        for n in logging.root.manager.loggerDict:
            logging.getLogger(n).setLevel(logging.WARN)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("medium")
    allow_ops_in_compiled_graph()


class ExCB(Callback):
    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            raise exception


class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step=None) -> None:
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)

    @property
    def log_dir(self) -> str:
        version = (
            self.version
            if isinstance(self.version, str)
            else f"version_{self.version:02}"
        )
        log_dir = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir
