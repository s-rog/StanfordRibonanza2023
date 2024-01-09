import sys
from argparse import Namespace

import lightning.pytorch as pl
import pandas as pd
import torch

torch.set_float32_matmul_precision("medium")
sys.path.append("main")
from bottle import RNA_Lightning
from data import RNA_DM
from utils import collate_preds, submission


if __name__ == "__main__":
  ckpt_map = {
      "fold_0_epoch18-step9500.ckpt": "fold0.parquet",
      "fold_1_epoch14-step7500.ckpt": "fold1.parquet",
      "fold_2_epoch26-step13500.ckpt": "fold2.parquet",
      "fold_3_epoch14-step7500.ckpt": "fold3.parquet",
      "fold_4_epoch19-step10000.ckpt": "fold4.parquet",
  }
  df_infer = pd.read_parquet("data/test_sequences_processed_ALL.parquet")
  n_workers = 8

  for k, v in ckpt_map.items():
      hp = Namespace(**torch.load(k)["hyper_parameters"])
      dm = RNA_DM(hp, n_workers, df_infer)
      model = RNA_Lightning.load_from_checkpoint(k, hp=hp, strict=True)
      model.eval()
      preds = pl.Trainer(
          precision="16-mixed",
          accelerator="gpu",
          benchmark=True,
          enable_model_summary=False,
          logger=False,
      ).predict(model, datamodule=dm)
      submission(collate_preds(preds), v)
  
