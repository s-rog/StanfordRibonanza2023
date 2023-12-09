from argparse import Namespace

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from modules import RNA_Model
from utils import sort_weight_decay_params
from apex.optimizers import FusedAdam as Adam


class RNA_Lightning(pl.LightningModule):
    def __init__(self, hp: Namespace):
        super(RNA_Lightning, self).__init__()
        self.save_hyperparameters(hp)
        self.hp = self.hparams
        if len(hp.layer_gru) < hp.n_layers:
            m = -(-hp.n_layers // len(hp.layer_gru))
            hp.layer_gru = (list(hp.layer_gru) * m)[: hp.n_layers]
        if len(hp.layer_bpp) < hp.n_layers:
            m = -(-hp.n_layers // len(hp.layer_bpp))
            hp.layer_bpp = (list(hp.layer_bpp) * m)[: hp.n_layers]
        self.model = RNA_Model(**vars(hp))
        self.forward = self.model.forward

    def configure_optimizers(self):
        if self.hp.wt_decay:
            decay, no_decay = sort_weight_decay_params(self.model)
            opt_groups = [
                {"params": decay, "weight_decay": self.hp.wt_decay},
                {"params": no_decay, "weight_decay": 0},
            ]
            opt = Adam(opt_groups)
        else:
            opt = Adam(self.model.parameters(), weight_decay=0)
        return opt

    def on_train_start(self):
        self.n_steps = self.trainer.estimated_stepping_batches
        self.n_warmup_steps = self.n_steps * self.hp.lr_warmup
        self.hp_metric, self.val_cache = 0.2, []
        self.logger.log_hyperparams(self.hp, {"hp/metric": self.hp_metric})

    def on_train_epoch_start(self):
        epoch, step = self.trainer.current_epoch, self.trainer.global_step
        self.logger.log_metrics({"hp/epoch": epoch}, step)

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        l = F.l1_loss(x, y, reduction="none")
        return l.nanmean()

    def fit_forward(self, x: dict, batch: dict):
        log_prefix = "loss/T" if self.training else "loss/V"
        x["react"] = x["react"] if self.training else x["react"].clip(0, 1)
        loss = self.loss(x["react"], batch["react"])
        log_d = {log_prefix: loss}
        self.log_dict(log_d, on_step=False, on_epoch=True, add_dataloader_idx=False)
        return loss

    def update_LR(self):
        if (step := self.trainer.global_step) < (n_warmup := self.n_warmup_steps):
            lr_m = float(step) / float(max(1, n_warmup))
        else:
            shift = (scale := n_warmup * self.hp.lr_scale) - n_warmup
            lr_m = 1.0 / (((step + shift) / scale) ** 0.5)
        self.log("hp/lr", (lr := self.hp.lr * lr_m))
        for p in self.optimizers().optimizer.param_groups:
            p["lr"] = lr

    def training_step(self, batch, batch_idx):
        self.update_LR()
        return self.fit_forward(self(batch), batch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x = self(batch)
        if dataloader_idx:
            mask, lmax = batch["mask"].sum(1), batch["seq"].size(1)
            for k, v in self.val_cache.pop(0).items():
                v = [v[i][: mask[i]].flip(0) for i in range(v.size(0))]
                v = [F.pad(s, [0] * 3 + [lmax - len(s)]) for s in v]
                x[k] = (x[k] + torch.stack(v)) / 2
        elif self.hp.val_flip:
            return self.val_cache.append(x)
        self.fit_forward(x, batch)

    def on_validation_end(self):
        if "loss/V" in self.trainer.logged_metrics:
            l = self.trainer.logged_metrics["loss/V"].item()
            if l < self.hp_metric:
                self.hp_metric = l
        step = self.trainer.global_step - 1
        self.logger.log_metrics({"hp/metric": self.hp_metric}, step)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        Lmax = batch["mask"].sum(-1).max().long()
        batch["mask"] = batch["mask"][:, :Lmax]
        batch["seq"] = batch["seq"][:, :Lmax]
        batch["bpp"] = batch["bpp"][:, :Lmax, :Lmax]
        x, mask = self(batch), batch["mask"].to(torch.bool)
        x["react"] = x["react"].clip(0, 1)
        if dataloader_idx:
            mask = mask.flip(1)
            x = {k: v.flip(1) for k, v in x.items()}
        return {k: v[mask] for k, v in x.items()}
