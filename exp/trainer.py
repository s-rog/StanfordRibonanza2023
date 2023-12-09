#!/usr/bin/env python3
import gc
import os
import sys
import time

import lightning.pytorch as pl
import lightning.pytorch.callbacks as CB
import pandas as pd
import rich
import torch
from rich import print

sys.path.append("../main")
from bottle import RNA_Lightning
from data import RNA_DM
from utils import *

if __name__ == "__main__":
    pseudo = False
    debug = 0
    n_workers = 8
    pred = True
    ckpt = True
    seed = 420
    early_stop = 15
    n_trials = 0
    n_folds = 5
    run_folds = range(n_folds)
    log_dir = "e32"
    pred_dir = "subs"
    metric = "loss/V"
    hp_conf = {
        "n_epochs": 100,
        "lr": 5e-4,
        "lr_warmup": 0.01,
        "lr_scale": 3,
        "wt_decay": 1e-1,
        "grad_clip": 5,
        "grad_accum_sched": {},
        "batch_size": 400,
        "sn_bias_sched": {(i * 10): v / 10 for i, v in enumerate(range(-12, -7))},
        "aug_flip": False,
        "tta_flip": False,
        "val_flip": False,
        "layer_gru": (1,),
        "layer_bpp": (1,) * 6 + (0,) * 6,
        "noise_bpp": 0.025,
        "pos_bias_params": (32, 128),
        "norm_rms": True,
        "qkv_bias": False,
        "ffn_bias": False,
        "ffn_multi": 4,
        "n_layers": 12,
        "n_heads": 6,
        "d_heads": 48,
        "p_dropout": 0.1,
        "att_fn": ["sdpa", "xmea"][1],
        "pretrained": False if pseudo else "e31/version_17/epoch=60-step=30500.ckpt",
        "n_folds": n_folds,
        "seed": seed,
        "note": "PL" if pseudo else "",
    }
    hp_skips = []
    df_train = "../data/train_data_processed_ALL_2.parquet"
    df_infer = "../data/test_sequences_processed_ALL_.parquet"
    df_pseudo = "../data/pseudo_label_e31v05-09.parquet"
    try:
        with rich.get_console().status("Reticulating Splines"):
            init_seed(seed, debug)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True) if pred else None
            df_infer = pd.read_parquet(df_infer) if pred else None
            df_train = pd.read_parquet(df_train)
            df_pseudo = pd.read_parquet(df_pseudo) if pseudo else None
            trials = grid_search(hp_conf, hp_skips)
            if (not n_trials) or len(trials) < n_trials:
                n_trials = len(trials)
        print(f"Log: {log_dir} | EStop: {early_stop} | Ckpt: {ckpt} | Pred: {pred}")
        for i, hp in enumerate(trials[:n_trials]):
            for j, f in enumerate(run_folds):
                print(f"Trial {i + 1}/{n_trials}", end=" ")
                print(f"Fold {j + 1}/{len(run_folds)} ({f}/{hp.n_folds})")
                hp.fold = f
                tbl = TBLogger(os.getcwd(), log_dir, default_hp_metric=False)
                cb = [CB.RichProgressBar(), ExCB()]
                ckpt_args = tbl.log_dir, None, metric, False, True, 5
                cb += [CB.ModelCheckpoint(*ckpt_args)] if ckpt else []
                cb += [CB.EarlyStopping(metric, 0, early_stop)] if early_stop else []
                if hp.grad_accum_sched:
                    cb += [CB.GradientAccumulationScheduler(hp.grad_accum_sched)]
                dm_args = [hp, n_workers, df_infer]
                dm_args += [df_pseudo, df_train] if pseudo else [df_train]
                dm = RNA_DM(*dm_args)
                if hp.pretrained:
                    model = RNA_Lightning.load_from_checkpoint(hp.pretrained, hp=hp)
                else:
                    model = RNA_Lightning(hp)
                trainer = pl.Trainer(
                    precision="16-mixed",
                    accelerator="gpu",
                    benchmark=True,
                    max_epochs=hp.n_epochs,
                    gradient_clip_val=hp.grad_clip,
                    fast_dev_run=debug,
                    num_sanity_val_steps=0,
                    enable_model_summary=False,
                    logger=tbl,
                    callbacks=cb,
                    reload_dataloaders_every_n_epochs=1,
                )
                gc.collect()
                try:
                    torch._dynamo.reset()
                    trainer.fit(model, datamodule=dm)
                except KeyboardInterrupt:
                    print("Fit Interrupted")
                    if i + 1 < n_trials:
                        with rich.get_console().status("Quit?") as s:
                            for k in range(3):
                                s.update(f"Quit? {3-k}")
                                time.sleep(1)
                    continue
                if pred:
                    try:
                        cp = None if debug else "best"
                        torch._dynamo.reset()
                        preds = trainer.predict(model, datamodule=dm, ckpt_path=cp)
                    except KeyboardInterrupt:
                        print("Prediction Interrupted")
                        continue
                    with rich.get_console().status("Processing Submission"):
                        preds = collate_preds(preds)
                        fn = f"{pred_dir}/{log_dir}v{tbl.version:02}"
                        submission(preds, fn if not debug else None)
                    del preds
    except KeyboardInterrupt:
        print("Goodbye")
        sys.exit()
