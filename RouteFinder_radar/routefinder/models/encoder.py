from typing import Tuple, Union

import torch.nn as nn
import torch
import torch.nn.functional as F

from rl4co.utils.pylogger import get_pylogger
from torch import Tensor

from routefinder.models.env_embeddings.mtvrp import MTVRPInitEmbeddingRouteFinder
from routefinder.models.nn.transformer import Normalization, TransformerBlock

log = get_pylogger(__name__)


class RouteFinderEncoder(nn.Module):
    """
    Encoder for RouteFinder model based on the Transformer Architecture.
    Here we include additional embedding from raw to embedding space, as
    well as more modern architecture options compared to the usual Attention Models
    based on POMO (including multi-task VRP ones).
    """

    def __init__(
        self,
        init_embedding: nn.Module = None,
        num_heads: int = 8,
        embed_dim: int = 128,
        num_layers: int = 6,
        feedforward_hidden: int = 512,
        normalization: str = "instance",
        use_prenorm: bool = False,
        use_post_layers_norm: bool = False,
        parallel_gated_kwargs: dict = None,
        init_type: str = "svd",
        attn_norm: str = "sinkhorn",
        cost_norm: bool = True,
        qk_bias: bool = True,
        **transformer_kwargs,
    ):
        """
        RADAR encoder for asymmetric VRPs (distance matrix in td["locs"]).
        init_type: distance features of the initial node embedding:
            "svd" (RADAR, top-k SVD factors), "nn" (distances to the 50 nearest nodes), "none".
        attn_norm: attention normalization, "sinkhorn" (RADAR) or "softmax" (MatNet).
        cost_norm: feed the per-instance standardized distance matrix to the attention.
        qk_bias: use bias in the query/key projections.
        """
        super(RouteFinderEncoder, self).__init__()

        if init_embedding is None:
            init_embedding = MTVRPInitEmbeddingRouteFinder(embed_dim=embed_dim, init_type=init_type)
        else:
            log.warning("Using custom init_embedding")
        self.init_embedding = init_embedding

        # self.layers = nn.Sequential(
        #     *(
        #         TransformerBlock(
        #             embed_dim=embed_dim,
        #             num_heads=num_heads,
        #             normalization=normalization,
        #             use_prenorm=use_prenorm,
        #             feedforward_hidden=feedforward_hidden,
        #             parallel_gated_kwargs=parallel_gated_kwargs,
        #             **transformer_kwargs,
        #         )
        #         for _ in range(num_layers)
        #     )
        # )

        self.layers = nn.ModuleList(
            [EncodingBlock(embed_dim, num_heads, attn_norm=attn_norm, qk_bias=qk_bias) for _ in range(num_layers)]
        )
        self.cost_norm = cost_norm

        self.post_layers_norm = (
            Normalization(embed_dim, normalization) if use_post_layers_norm else None
        )

    def forward(
        self, td: Tensor, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:

        # Transfer to embedding space
        init_h = self.init_embedding(td)  # [B, N, H]

        cost_mat = td['locs']
        if getattr(self, 'cost_norm', True):
            # attention uses the per-instance standardized distance matrix
            cost_mat = (cost_mat - cost_mat.mean(dim=(1, 2), keepdim=True)) / (
                cost_mat.std(dim=(1, 2), keepdim=True) + 1e-9
            )

        # Process embedding
        h = init_h
        for layer in self.layers:
            h = layer(h, cost_mat, mask)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h)

        # Return latent representation
        return h, init_h  # [B, N, H]

class EncodingBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, attn_norm="sinkhorn", qk_bias=False):
        super().__init__()
        qkv_dim = embedding_dim // head_num
        self.head_num = head_num

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=qk_bias)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=qk_bias)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.W_sym = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.mixed_score_MHA = MixedScore_MultiHeadAttention(head_num, attn_norm=attn_norm)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)
        self.add_n_normalization_2 = AddAndInstanceNormalization(embedding_dim)

        self.g_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, node_emb, cost_mat, mask=None):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.head_num

        q = reshape_by_heads(self.Wq(node_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(node_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(node_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        out1 = self.add_n_normalization_1(node_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3

class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, head_num, attn_norm="sinkhorn"):
        super().__init__()

        ms_hidden_dim = 16
        self.head_num = head_num
        self.attn_norm = attn_norm
        mix1_init = 3**-0.5
        mix2_init = ms_hidden_dim**-0.5

        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, 3, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)

    def forward(self, q, k, v, cost_mat):
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.head_num
        qkv_dim = q.size(-1)
        sqrt_qkv_dim = qkv_dim ** 0.5

        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)

        dot_product_score = dot_product / sqrt_qkv_dim
        # shape: (batch, head_num, row_cnt, col_cnt)

        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt)

        two_scores = torch.stack((dot_product_score, cost_mat_score, cost_mat_score.transpose(2,3)), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)

        two_scores_transposed = two_scores.transpose(1,2)
        # shape: (batch, row_cnt, head_num, col_cnt, 2)

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        mixed_scores = ms2.transpose(1,2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)

        # Sinkhorn normalization (RADAR) or row-wise softmax (MatNet)
        if getattr(self, 'attn_norm', 'sinkhorn') == 'softmax':
            weights = torch.softmax(mixed_scores, dim=3)
        else:
            weights = sinkhorn_normalization(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat

class AddAndInstanceNormalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        ff_hidden_dim = 512

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def sinkhorn_normalization(scores, n_iter=10):
    scores = scores - scores.max(dim=-1, keepdim=True)[0]  # 避免数值溢出
    for _ in range(n_iter):
        scores = scores - scores.logsumexp(dim=-1, keepdim=True)
        scores = scores - scores.logsumexp(dim=-2, keepdim=True)
    return scores.exp()