from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm import Mamba2 as _Mamba2
    _HAS_MAMBA = True
except Exception:
    _Mamba2 = None
    _HAS_MAMBA = False


@dataclass
class TomatoPGFMConfig:
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 9
    n_heads: int = 6
    ff_mult: int = 4
    graph_feature_dim: int = 8
    max_seq_len: int = 2048
    vtp_classes: int = 10
    attn_window: int = 256
    use_moe: bool = False
    moe_experts: int = 0
    moe_top_k: int = 2
    moe_aux_weight: float = 0.01
    use_real_mamba: bool = True
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64


class GraphAdapter(nn.Module):
    """Injects per-node graph *features* (not structure) as a gated residual."""

    def __init__(self, graph_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(graph_dim), nn.Linear(graph_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: torch.Tensor, graph_features: torch.Tensor | None, mode: str = "on") -> torch.Tensor:
        if graph_features is None or mode == "off":
            return h
        if mode == "shuffle":
            # Break the token<->graph-node correspondence while keeping the real
            # feature distribution: permute along the sequence axis per example.
            b, l, _ = graph_features.shape
            perm = torch.stack([torch.randperm(l, device=graph_features.device) for _ in range(b)])
            graph_features = torch.gather(graph_features, 1, perm.unsqueeze(-1).expand_as(graph_features))
        return h + torch.tanh(self.gate) * self.proj(graph_features)


class GraphMessage(nn.Module):
    """GCN-style message passing over the *real* DAG adjacency (edge_index).

    This is what actually makes the model graph-aware: each token/node receives
    the mean of its real predecessors' hidden states. ``edge_index`` is a
    (2, E) tensor of global indices into the flattened (B*L) node set, built
    from each example's ``real_dag_edges``. The shuffle mode permutes destination indices across the flattened batch
    and is used as a perturbation control for the implemented graph-message path.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor | None, mode: str = "on") -> torch.Tensor:
        if edge_index is None or mode == "off" or edge_index.numel() == 0:
            return h
        b, l, d = h.shape
        src, dst = edge_index[0], edge_index[1]
        if mode == "shuffle":
            dst = dst[torch.randperm(dst.numel(), device=dst.device)]
        flat = h.reshape(b * l, d)
        agg = torch.zeros_like(flat)
        agg.index_add_(0, dst, flat.index_select(0, src))
        deg = torch.zeros(b * l, device=h.device).index_add_(0, dst, torch.ones(dst.numel(), device=h.device))
        agg = agg / deg.clamp(min=1.0).unsqueeze(-1)
        msg = self.proj(agg).reshape(b, l, d)
        return h + torch.tanh(self.gate) * msg


class BiMambaLikeBlock(nn.Module):
    """Conv1d fallback for BiMamba2 (bidirectional depthwise convolutions).

    Used only when mamba_ssm is unavailable or cfg.use_real_mamba is False.
    The real production path is BiMamba2Block below.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.rev = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.mix = nn.Linear(d_model, d_model)
        self.dir_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x).transpose(1, 2)
        f = self.fwd(z).transpose(1, 2)
        r = torch.flip(self.rev(torch.flip(z, dims=[2])).transpose(1, 2), dims=[1])
        g = torch.sigmoid(self.dir_gate)
        return x + self.mix(g * f + (1 - g) * r)


class BiMamba2Block(nn.Module):
    """Real bidirectional Mamba2 block. Same interface as BiMambaLikeBlock:
    forward(x: (B, L, D)) -> (B, L, D).

    Mamba2 is causal/unidirectional; bidirectionality is fwd over x plus a second
    pass over the reversed sequence (flipped back), gated by dir_gate -- mirroring
    the fallback's structure so the two are drop-in swappable.
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64):
        super().__init__()
        if not _HAS_MAMBA:
            raise RuntimeError("BiMamba2Block requires mamba_ssm, which is not importable")
        self.norm = nn.LayerNorm(d_model)
        self.fwd = _Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.rev = _Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.mix = nn.Linear(d_model, d_model)
        self.dir_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm(x)
        f = self.fwd(n)
        r = torch.flip(self.rev(torch.flip(n, dims=[1])), dims=[1])
        g = torch.sigmoid(self.dir_gate)
        return x + self.mix(g * f + (1 - g) * r)


def make_sequence_block(cfg: "TomatoPGFMConfig") -> nn.Module:
    """Build a production Mamba2 block or an explicitly requested smoke fallback."""
    if cfg.use_real_mamba:
        if not _HAS_MAMBA:
            raise RuntimeError(
                "This configuration requires mamba_ssm.Mamba2. Install the "
                "production inference dependencies, or set use_real_mamba=false "
                "only for CPU smoke tests."
            )
        return BiMamba2Block(cfg.d_model, cfg.mamba_d_state, cfg.mamba_d_conv,
                             cfg.mamba_expand, cfg.mamba_headdim)
    return BiMambaLikeBlock(cfg.d_model)


class MoEFeedForward(nn.Module):
    """Top-k mixture-of-experts FFN with load-balance and router z-loss.

    Returns (output, aux_loss). The aux loss combines the Switch-Transformer
    load-balancing term (keeps experts evenly used) and the router z-loss
    (keeps router logits from blowing up); both are required by the plan when
    MoE is enabled.
    """

    def __init__(self, d_model: int, ff_mult: int, n_experts: int, top_k: int, aux_weight: float):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.aux_weight = aux_weight
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(), nn.Linear(ff_mult * d_model, d_model)) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, l, d = x.shape
        xf = x.reshape(-1, d)
        t = xf.shape[0]
        logits = self.router(xf)
        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        out = torch.zeros_like(xf)
        for k in range(self.top_k):
            idx = topi[:, k]
            w = topv[:, k : k + 1]
            for e in range(self.n_experts):
                sel = idx == e
                if sel.any():
                    out[sel] = out[sel] + w[sel] * self.experts[e](xf[sel])
        importance = probs.mean(dim=0)
        top1 = probs.argmax(dim=-1)
        load = torch.bincount(top1, minlength=self.n_experts).float() / max(1, t)
        load_balance = (importance * load).sum() * self.n_experts
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        aux = self.aux_weight * (load_balance + 0.001 * z_loss)
        return out.reshape(b, l, d), aux


class SparseAttentionBlock(nn.Module):
    """Local (sliding-window) self-attention with an optional MoE FFN.

    An additive band mask restricts each position to +/- ``window`` tokens. The
    current implementation uses ``nn.MultiheadAttention`` with a dense L-by-L
    mask; the mask is local, but this module does not claim sparse-kernel
    computational complexity.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, window: int = 256, moe: MoEFeedForward | None = None):
        super().__init__()
        self.window = window
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = moe
        self.ff = None if moe is not None else nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(), nn.Linear(ff_mult * d_model, d_model))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.norm1(x)
        y, _ = self.attn(n, n, n, attn_mask=attn_mask, need_weights=False)
        x = x + y
        aux = x.new_zeros(())
        if self.moe is not None:
            ff_out, aux = self.moe(self.norm2(x))
        else:
            ff_out = self.ff(self.norm2(x))
        return x + ff_out, aux


def _local_attention_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    band = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= window
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask = mask.masked_fill(~band, float("-inf"))
    return mask


class TomatoPGFM(nn.Module):
    def __init__(self, cfg: TomatoPGFMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.graph = GraphAdapter(cfg.graph_feature_dim, cfg.d_model)
        self.graph_msg = GraphMessage(cfg.d_model)
        blocks = []
        for i in range(cfg.n_layers):
            if i % 3 == 2:
                moe = MoEFeedForward(cfg.d_model, cfg.ff_mult, cfg.moe_experts, cfg.moe_top_k, cfg.moe_aux_weight) if (cfg.use_moe and cfg.moe_experts > 0) else None
                blocks.append(SparseAttentionBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult, window=cfg.attn_window, moe=moe))
            else:
                blocks.append(make_sequence_block(cfg))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.mlm_head.weight = self.embed.weight
        self.vtp_head = nn.Linear(cfg.d_model, cfg.vtp_classes)
        self.path_head = nn.Linear(cfg.d_model, cfg.d_model)
        self.graph_recon = nn.Linear(cfg.d_model, cfg.graph_feature_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        graph_features: torch.Tensor | None = None,
        graph_mode: str = "on",
        edge_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, l = input_ids.shape
        pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        h = self.embed(input_ids) + self.pos(pos)
        h = self.graph(h, graph_features, graph_mode)
        h = self.graph_msg(h, edge_index, graph_mode)
        attn_mask = _local_attention_mask(l, self.cfg.attn_window, input_ids.device)
        aux_total = h.new_zeros(())
        for block in self.blocks:
            if isinstance(block, SparseAttentionBlock):
                h, aux = block(h, attn_mask=attn_mask)
                aux_total = aux_total + aux
            else:
                h = block(h)
        h = self.norm(h)
        pooled = h.mean(dim=1)
        return {
            "hidden": h,
            "pooled": pooled,
            "mlm_logits": self.mlm_head(h),
            "vtp_logits": self.vtp_head(pooled),
            "path_proj": F.normalize(self.path_head(pooled), dim=-1),
            "graph_recon": self.graph_recon(h),
            "moe_aux": aux_total,
        }


def parameter_report(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def smoke_forward() -> dict:
    cfg = TomatoPGFMConfig(vocab_size=128, d_model=64, n_layers=3, n_heads=4,
                           graph_feature_dim=16, max_seq_len=64,
                           use_real_mamba=False)
    model = TomatoPGFM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    g = torch.randn(2, 32, cfg.graph_feature_dim)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])  # local edges within example 0
    on = model(x, g, "on", edge_index=edges)["pooled"]
    off = model(x, g, "off")["pooled"]
    diff = float((on - off).abs().mean().detach())
    return {**parameter_report(model), "graph_on_off_mean_abs_diff": diff, "status": "pass" if diff > 0 else "fail"}
