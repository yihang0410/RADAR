"""RADAR encoder for RRNCO: SVD features of the distance matrix as initial embedding, and
self-attention whose scores mix (q.k, D, D^T) and are normalised with Sinkhorn. One
embedding per node, returned as both the row and the column embedding of RRNetDecoder.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.common.constructive import AutoregressiveEncoder
from tensordict import TensorDict
from torch import Tensor

from rrnco.models.env_embeddings.rcvrp import ContextualGating, CoordinateExpert


class RadarEncoder(AutoregressiveEncoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        feedforward_hidden: int = 512,
        env_name: str = "rcvrp",
        svd_rank: int = 10,
        ms_hidden_dim: int = 16,
        sinkhorn_iters: int = 10,
        use_coords: bool = False,
        svd_norm: str = "zscore",
        attn_norm: str = "zscore",
        **unused_kwargs,
    ):
        super().__init__()
        if env_name not in ("atsp", "rcvrp", "rcvrptw"):
            raise NotImplementedError(f"RadarEncoder not defined for env '{env_name}'")
        if env_name == "rcvrptw" and not use_coords:
            raise NotImplementedError(
                "RadarEncoder for rcvrptw is only defined with use_coords=True"
            )
        for v in (svd_norm, attn_norm):
            if v not in ("zscore", "minmax"):
                raise ValueError(f"unknown distance normalisation '{v}'")
        self.env_name = env_name
        self.svd_rank = svd_rank
        self.use_coords = use_coords
        self.svd_norm = svd_norm
        self.attn_norm = attn_norm
        if use_coords:
            self.projection = nn.Linear(2 * svd_rank, embed_dim)
            self.gating = ContextualGating(embed_dim)
            if env_name == "atsp":
                self.coord_embed = nn.Linear(2, embed_dim)
            elif env_name == "rcvrp":
                self.coord_expert = CoordinateExpert(3, embed_dim)
                self.demand_init = nn.Linear(1, embed_dim)
                self.combine = nn.Linear(2 * embed_dim, embed_dim)
            else:
                self.coord_expert = CoordinateExpert(3, embed_dim)
                self.attr_init = nn.Linear(4, embed_dim)
                self.combine = nn.Linear(2 * embed_dim, embed_dim)
        else:
            in_dim = 2 * svd_rank + (1 if env_name == "rcvrp" else 0)
            self.projection = nn.Linear(in_dim, embed_dim)
        self.layers = nn.ModuleList(
            [
                RadarEncodingBlock(
                    embed_dim,
                    num_heads,
                    feedforward_hidden,
                    ms_hidden_dim,
                    sinkhorn_iters,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, td: TensorDict, phase: str = "test", mask=None
    ) -> Tuple[Tensor, Tensor]:
        minmax = td["distance_matrix"].float()
        zscore = (minmax - minmax.mean(dim=(1, 2), keepdim=True)) / (
            minmax.std(dim=(1, 2), keepdim=True) + 1e-9
        )
        svd_in = zscore if self.svd_norm == "zscore" else minmax
        cost = zscore if self.attn_norm == "zscore" else minmax
        # svd_lowrank needs fp32
        with torch.autocast(device_type=cost.device.type, enabled=False):
            U, S, V = torch.svd_lowrank(svd_in, q=self.svd_rank)
        sqrt_S = torch.sqrt(S).unsqueeze(1)
        X = torch.cat([U * sqrt_S, V * sqrt_S], dim=-1)  # (B, N, 2k)

        if self.use_coords:
            h = self._embed_with_coords(td, X)
        else:
            if self.env_name == "rcvrp":
                demand = td["demand"].float()
                # depot: 0
                demand = torch.cat([torch.zeros_like(demand[:, :1]), demand], dim=1)
                demand = demand / (demand.sum(dim=1, keepdim=True) + 1e-6)
                X = torch.cat([X, demand.unsqueeze(-1)], dim=-1)
            h = self.projection(X)
        for layer in self.layers:
            h = layer(h, cost)
        return h, h  # RRNetDecoder takes (row_emb, col_emb)

    def _embed_with_coords(self, td: TensorDict, X: Tensor) -> Tensor:
        locs = td["locs"].float()
        svd_emb = self.projection(X)
        if self.env_name == "atsp":
            return self.gating(self.coord_embed(locs), svd_emb)
        h = self.gating(self.coord_expert(locs), svd_emb)
        if self.env_name == "rcvrptw":
            attr = torch.cat(
                [
                    td["demand_linehaul"][..., None],
                    td["time_windows"],
                    td["service_time"][..., None],
                ],
                dim=-1,
            ).float()
            return self.combine(torch.cat([h, self.attr_init(attr)], dim=-1))
        demand = td["demand"].float()
        # depot: 0
        demand = torch.cat([torch.zeros_like(demand[:, :1]), demand], dim=1)
        return self.combine(torch.cat([h, self.demand_init(demand[..., None])], dim=-1))


class RadarEncodingBlock(nn.Module):
    def __init__(
        self, embed_dim, num_heads, feedforward_hidden, ms_hidden_dim, sinkhorn_iters
    ):
        super().__init__()
        qkv_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.Wq = nn.Linear(embed_dim, num_heads * qkv_dim, bias=True)
        self.Wk = nn.Linear(embed_dim, num_heads * qkv_dim, bias=True)
        self.Wv = nn.Linear(embed_dim, num_heads * qkv_dim, bias=False)
        self.mixed_score_mha = MixedScoreSinkhornMHA(
            num_heads, qkv_dim, ms_hidden_dim, sinkhorn_iters
        )
        self.multi_head_combine = nn.Linear(num_heads * qkv_dim, embed_dim)
        self.norm1 = nn.InstanceNorm1d(embed_dim, affine=True, track_running_stats=False)
        self.W1 = nn.Linear(embed_dim, feedforward_hidden)
        self.W2 = nn.Linear(feedforward_hidden, embed_dim)
        self.norm2 = nn.InstanceNorm1d(embed_dim, affine=True, track_running_stats=False)

    def forward(self, h, cost):
        q = _heads(self.Wq(h), self.num_heads)
        k = _heads(self.Wk(h), self.num_heads)
        v = _heads(self.Wv(h), self.num_heads)
        out = self.multi_head_combine(self.mixed_score_mha(q, k, v, cost))
        out1 = self.norm1((h + out).transpose(1, 2)).transpose(1, 2)
        out2 = self.W2(F.relu(self.W1(out1)))
        return self.norm2((out1 + out2).transpose(1, 2)).transpose(1, 2)


class MixedScoreSinkhornMHA(nn.Module):
    def __init__(self, num_heads, qkv_dim, ms_hidden_dim, sinkhorn_iters):
        super().__init__()
        self.num_heads = num_heads
        self.sqrt_qkv_dim = qkv_dim**0.5
        self.sinkhorn_iters = sinkhorn_iters
        mix1_init, mix2_init = 0.5**0.5, (1 / 16) ** 0.5
        U = torch.distributions.Uniform
        self.mix1_weight = nn.Parameter(
            U(-mix1_init, mix1_init).sample((num_heads, 3, ms_hidden_dim))
        )
        self.mix1_bias = nn.Parameter(
            U(-mix1_init, mix1_init).sample((num_heads, ms_hidden_dim))
        )
        self.mix2_weight = nn.Parameter(
            U(-mix2_init, mix2_init).sample((num_heads, ms_hidden_dim, 1))
        )
        self.mix2_bias = nn.Parameter(U(-mix2_init, mix2_init).sample((num_heads, 1)))

    def forward(self, q, k, v, cost):
        # q, k, v: (B, H, N, d); cost: (B, N, N)
        B, H, N, _ = q.shape
        dot = torch.matmul(q, k.transpose(2, 3)) / self.sqrt_qkv_dim
        c = cost[:, None, :, :].expand(B, H, N, N).to(dot.dtype)
        scores = torch.stack((dot, c, c.transpose(2, 3)), dim=4).transpose(1, 2)
        # (B, N, H, N, 3)
        ms1 = F.relu(
            torch.matmul(scores, self.mix1_weight)
            + self.mix1_bias[None, None, :, None, :]
        )
        ms2 = torch.matmul(ms1, self.mix2_weight) + self.mix2_bias[None, None, :, None, :]
        mixed = ms2.transpose(1, 2).squeeze(4)  # (B, H, N, N)
        weights = _sinkhorn(mixed.float(), self.sinkhorn_iters).to(v.dtype)
        out = torch.matmul(weights, v)  # (B, H, N, d)
        return out.transpose(1, 2).reshape(B, N, -1)


def _sinkhorn(scores, n_iter):
    """Alternate row and column normalisation in log space; returns exp of the result."""
    scores = scores - scores.max(dim=-1, keepdim=True)[0]
    for _ in range(n_iter):
        scores = scores - scores.logsumexp(dim=-1, keepdim=True)
        scores = scores - scores.logsumexp(dim=-2, keepdim=True)
    return scores.exp()


def _heads(x, num_heads):
    B, N, _ = x.shape
    return x.reshape(B, N, num_heads, -1).transpose(1, 2)
