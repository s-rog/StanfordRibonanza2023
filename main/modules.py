import random
from functools import partial

import torch
import torch.nn.functional as F
import xformers.ops as xops
from torch import Tensor, nn
from x_transformers.x_transformers import RelativePositionBias as RelPB
from apex.normalization import FusedLayerNorm, FusedRMSNorm

compile = partial(torch.compile)


class AttentionBias(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_heads: int,
        p_dropout: float,
        pos_bias_params: tuple[int, int],
    ):
        super().__init__()
        self.pos_bias = RelPB(d_heads**0.5, False, *pos_bias_params, n_heads)
        self.bpp = nn.Sequential(nn.Conv2d(1, n_heads, 1), nn.Dropout(p_dropout, True))

    @compile
    def forward(self, mask: Tensor, bpp: Tensor) -> tuple[Tensor, Tensor]:
        B, L = mask.shape
        pos = self.pos_bias(L, L).unsqueeze(0)
        mask = mask.view(B, 1, 1, L) - 1
        mask[mask < 0] = float("-inf")
        mask = mask.expand(-1, pos.size(1), L, -1)
        bias = mask + pos.expand(B, -1, -1, -1)
        return bias, self.bpp(bpp.unsqueeze(1))


class RNA_Model(nn.Module):
    def __init__(
        self,
        norm_rms: str,
        layer_gru: tuple,
        layer_bpp: tuple,
        pos_bias_params: tuple[int, int],
        **kwargs,
    ):
        super().__init__()
        global Norm
        norm_cls = FusedRMSNorm if norm_rms else FusedLayerNorm
        Norm = partial(norm_cls, memory_efficient=True)
        d_model = (d_heads := kwargs["d_heads"]) * (n_heads := kwargs["n_heads"])
        p_dropout, n_layers = kwargs["p_dropout"], kwargs["n_layers"]
        layers = [kwargs | {"use_gru": g} for g in layer_gru]
        self.layers = nn.ModuleList([EncoderLayer(**k) for k in layers])
        self.emb = nn.Embedding(5, d_model, 0)
        ab_args = n_heads, d_heads, p_dropout, pos_bias_params
        self.att_bias = AttentionBias(*ab_args)
        self.out = nn.Sequential(nn.Dropout(p_dropout, True), nn.Linear(d_model, 2))
        self.layer_opts = (layer_bpp,)

    def forward(self, batch: dict) -> Tensor:
        x = self.emb(batch["seq"])
        res = x * self.layers[0].res_scale
        bias, bpp = self.att_bias(batch["mask"], batch["bpp"])
        for f, use_bpp in zip(self.layers, *self.layer_opts):
            x, res = f(x, res, bias + (bpp if use_bpp else 0))
        return {"react": self.out(x)}


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_heads: int,
        n_heads: int,
        n_layers: int,
        p_dropout: float,
        ffn_multi: int,
        ffn_bias: bool,
        qkv_bias: bool,
        att_fn: str,
        use_gru: int,
        **kwargs,
    ):
        super().__init__()
        d_ffn = (d_model := d_heads * n_heads) * ffn_multi
        self.att = MultiheadAttention(d_model, n_heads, p_dropout, qkv_bias, att_fn)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn, ffn_bias),
            nn.GELU(),
            nn.Dropout(p_dropout, True),
            nn.Linear(d_ffn, d_model, ffn_bias),
            nn.Dropout(p_dropout, True),
        )
        self.ffn = compile(self.ffn)
        self.gru = GRU(d_model, p_dropout) if use_gru else None
        self.norm = nn.ModuleList([Norm(d_model) for _ in range(4)])
        self.res_scale = 0.1

    def forward(self, x: Tensor, res: Tensor, *args, **kwargs) -> tuple[Tensor, Tensor]:
        x_att = self.att(x, *args, **kwargs)
        res = res + x_att * self.res_scale
        x = self.norm[0](x + x_att) + self.norm[1](res)
        x_ffn = self.ffn(x)
        res = res + x_ffn * self.res_scale
        x = self.norm[2](x + x_ffn) + self.norm[3](res)
        if self.gru is not None:
            x = self.gru(x)
        return x, res


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        p_dropout: float,
        qkv_bias: bool,
        att_fn: str,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads
        self.p_dropout = p_dropout
        self.att = self.xmea if att_fn == "xmea" else self.sdpa
        self.qkv = nn.Linear(d_model, d_model * 3, qkv_bias)
        out = [nn.Linear(d_model, d_model), nn.Dropout(p_dropout, True)]
        self.out = nn.Sequential(*out)

    def forward(self, x: Tensor, bias: Tensor) -> Tensor:
        B, L, N, D = (*x.shape[:2], self.n_heads, self.d_heads)
        qkv = [_.view(B, L, N, D) for _ in self.qkv(x).chunk(3, -1)]
        x = self.att(qkv, bias, self.p_dropout if self.training else 0)
        return self.out(x.flatten(2))

    def sdpa(self, qkv: tuple[Tensor, Tensor, Tensor], bias: Tensor, p_dropout: float):
        qkv = [_.transpose(1, 2) for _ in qkv]
        bias = bias.type_as(qkv[0]).contiguous()
        return F.scaled_dot_product_attention(*qkv, bias, p_dropout).transpose(1, 2)

    def xmea(self, qkv: tuple[Tensor, Tensor, Tensor], bias: Tensor, p_dropout: float):
        bias = bias.type_as(qkv[0])
        if (L := qkv[0].size(1)) % 8:
            pad = -(L // -8) * 8 - L
            bias = F.pad(bias, (0, pad, 0, pad))
            bias = bias.contiguous()[..., :L, :L]
        return xops.memory_efficient_attention(*qkv, bias, p_dropout)


class GRU(nn.Module):
    def __init__(self, d_model: int, p_dropout: float):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p_dropout)

    @compile
    def forward(self, x: Tensor) -> Tensor:
        B, L, D = *x.shape[:2], x.size(-1) // 2
        x = x.view(B, L, D, 2).transpose(2, 3).flatten(2)
        x = self.gru(x)[0]
        x = x.view(B, L, 2, D).transpose(2, 3).flatten(2)
        return self.dropout(x)
