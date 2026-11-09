# VERBATIM from Gemini (block 1 of 3): hasher, grafted gated Engram module, layer-2 hook, EngramEdit solver.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

# =====================================================================
# 1. DETERMINISTIC MULTI-HEAD N-GRAM HASHER (Host DRAM Resident)
# =====================================================================
class MultiHeadNgramHasher:
    """
    Computes deterministic (2-gram, 3-gram) x K-head hash indices directly
    from input_ids on CPU so embeddings can be prefetched asynchronously.
    """
    def __init__(self, num_buckets: int = 2_000_000, num_heads: int = 4, orders: Tuple[int, ...] = (2, 3)):
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.orders = orders

        # Distinct odd multipliers primes per (order, head) to minimize collisions
        torch.manual_seed(42)
        self.primes = {
            n: torch.randint(100_003, 999_999_937, (num_heads, n), dtype=torch.int64) | 1
            for n in orders
        }

    @torch.no_grad()
    def hash_sequence(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        input_ids: [B, T] (on CPU)
        Returns dict mapping order n -> [B, T, K] bucket indices in [0, num_buckets)
        """
        B, T = input_ids.shape
        ids_cpu = input_ids.to(torch.int64).cpu()
        indices_by_order = {}

        for n in self.orders:
            # Left-pad sequence with 0s to extract causal n-grams ending at position t
            padded = F.pad(ids_cpu, (n - 1, 0), value=0)
            # Unfold into [B, T, n] windows
            windows = padded.unfold(dimension=1, size=n, step=1)
            # Hash across K heads: [B, T, 1, n] * [1, 1, K, n] -> sum over n -> mod buckets
            hashed = (windows.unsqueeze(2) * self.primes[n].view(1, 1, self.num_heads, n)).sum(dim=-1)
            indices_by_order[n] = (hashed % self.num_buckets).abs()

        return indices_by_order


# =====================================================================
# 2. GATED ENGRAM MODULE (Grafted onto Frozen Layer 2)
# =====================================================================
class GraftedEngramModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        r_mem: int = 512,
        num_buckets: int = 2_000_000,
        num_heads: int = 4,
        orders: Tuple[int, ...] = (2, 3),
        conv_kernel: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_model = d_model
        self.r_mem = r_mem
        self.orders = orders
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype

        # Each order gets half the memory bottleneck dimension (e.g., 256 for 2-gram, 256 for 3-gram)
        assert r_mem % len(orders) == 0
        self.r_per_order = r_mem // len(orders)

        self.hasher = MultiHeadNgramHasher(num_buckets, num_heads, orders)

        # Host-RAM (CPU) Pinned Embedding Tables + Frequency Counters for EngramEdit specificity
        self.cpu_tables = {
            n: torch.randn(num_buckets, self.r_per_order, dtype=dtype).mul_(0.02).pin_memory()
            for n in orders
        }
        self.bucket_counts = {
            n: torch.ones(num_buckets, dtype=torch.float32)
            for n in orders
        }

        # GPU-resident Context-Aware Gating & Fusion projections
        self.q_proj = nn.Linear(d_model, r_mem, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(r_mem, r_mem, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(r_mem, d_model, bias=False, device=device, dtype=dtype)

        self.q_norm = nn.RMSNorm(r_mem, eps=1e-6, device=device, dtype=dtype)
        self.k_norm = nn.RMSNorm(r_mem, eps=1e-6, device=device, dtype=dtype)

        # Causal depthwise 1D convolution over gated values
        self.conv_kernel = conv_kernel
        self.short_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            groups=d_model,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # STAPLE Zero-Initialization: ensures bit-exact identity with frozen backbone at step 0
        nn.init.zeros_(self.v_proj.weight)
        nn.init.dirac_(self.short_conv.weight)

        # Runtime state for async prefetch & EngramEdit perturbation hooks
        self._prefetched_e: torch.Tensor = None
        self._active_perturbation: Tuple[torch.Tensor, List[int]] = None  # (delta, pos_list)

    @torch.no_grad()
    def prefetch_from_host(self, input_ids: torch.Tensor):
        """
        Called right after tokenization. Gathers N-gram embeddings on CPU
        and streams only the [B, T, r_mem] tensor over PCIe.
        """
        idx_map = self.hasher.hash_sequence(input_ids)
        order_chunks = []

        for n in self.orders:
            idx = idx_map[n]  # [B, T, K] on CPU
            # Gather [B, T, K, r_per_order] and average over K hash heads
            gathered = self.cpu_tables[n][idx].mean(dim=2)
            order_chunks.append(gathered)

        # Concatenate 2-gram and 3-gram sub-vectors -> [B, T, r_mem] and copy non-blocking to GPU
        e_cpu = torch.cat(order_chunks, dim=-1)
        self._prefetched_e = e_cpu.to(self.device, dtype=self.dtype, non_blocking=True)

    def forward_fusion(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Computes gated Engram residual update given Layer 2 hidden_states: [B, T, d_model].
        """
        e_t = self._prefetched_e  # [B, T, r_mem]

        # Inject learnable perturbation delta_i during EngramEdit Stage 1
        if self._active_perturbation is not None:
            delta, anchor_positions = self._active_perturbation
            e_t = e_t.clone()
            for b_idx, pos in enumerate(anchor_positions):
                e_t[b_idx, pos] = e_t[b_idx, pos] + delta

        # Context-aware gating: Q from backbone hidden state, K/V from Engram memory
        q = self.q_norm(self.q_proj(hidden_states))  # [B, T, r_mem]
        k = self.k_norm(self.k_proj(e_t))            # [B, T, r_mem]
        v = self.v_proj(e_t)                         # [B, T, d_model]

        # Scaled dot-product per token -> scalar trust gate alpha_t in (0, 1)
        scale = self.r_mem ** -0.5
        alpha = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * scale)  # [B, T, 1]

        gated_v = alpha * v  # [B, T, d_model]

        # Causal depthwise ShortConv along sequence dimension T
        # Transpose to [B, d_model, T], left-pad by (kernel_size - 1) to prevent future leakage
        v_conv = F.pad(gated_v.transpose(1, 2), (self.conv_kernel - 1, 0))
        v_conv = self.short_conv(v_conv).transpose(1, 2)

        return hidden_states + v_conv


# =====================================================================
# 3. NON-INTRUSIVE LAYER-2 PYTHORCH HOOK WRAPPER
# =====================================================================
class Layer2EngramGraft:
    """Attaches GraftedEngramModule to any HuggingFace / custom DecoderLayer via forward hooks."""
    def __init__(self, model: nn.Module, target_layer: nn.Module, engram: GraftedEngramModule):
        self.model = model
        self.engram = engram

        # Hook 1: Pre-hook on model root to trigger CPU N-gram prefetch as soon as input_ids arrive
        self.pre_hook = model.register_forward_pre_hook(self._on_model_start, with_kwargs=True)
        # Hook 2: Forward hook on Layer 2 to splice the gated residual branch
        self.layer_hook = target_layer.register_forward_hook(self._on_layer_forward)

    def _on_model_start(self, module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
        if input_ids is not None and isinstance(input_ids, torch.Tensor):
            self.engram.prefetch_from_host(input_ids)

    def _on_layer_forward(self, module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h_aug = self.engram.forward_fusion(h)
            return (h_aug,) + output[1:]
        return self.engram.forward_fusion(output)


# =====================================================================
# 4. ENGRAM-EDIT SOLVER (Target Perturbation + Joint Ridge Allocation)
# =====================================================================
class EngramEditSolver:
    """
    Implements the two-stage EngramEdit update on the Host-RAM N-gram tables
    with 100% of GPU weights (backbone + fusion gates) frozen.
    """
    def __init__(self, model: nn.Module, engram: GraftedEngramModule):
        self.model = model
        self.engram = engram

    def apply_edit(
        self,
        prompt_ids_batch: torch.Tensor,   # [M, T] M paraphrased expressions of the fact/API
        anchor_positions: List[int],      # Index q_i(x) of last subject/symbol token per expression
        target_mask: torch.Tensor,        # [M, T] 1 where loss is evaluated (target fact tokens), 0 elsewhere
        steps: int = 20,
        lr: float = 0.08,
        clamp_rho: float = 0.4,           # Max norm ratio ||delta|| <= rho * ||e_q||
        order_penalties: Dict[int, float] = {2: 10.0, 3: 1.0},  # Penalize 2-grams 10x more than 3-grams
    ) -> float:
        # Freeze everything on GPU
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.engram.parameters():
            p.requires_grad_(False)

        # Prefetch baseline N-gram vectors for the M expressions to measure baseline norm
        self.engram.prefetch_from_host(prompt_ids_batch)
        base_e = self.engram._prefetched_e.detach()
        anchor_norms = torch.stack([base_e[m, pos].norm() for m, pos in enumerate(anchor_positions)]).mean()
        max_delta_norm = (clamp_rho * anchor_norms).clamp(min=0.1)

        # -----------------------------------------------------------------
        # STAGE 1: Optimize shared perturbation delta_i* in Layer-2 Memory Space
        # -----------------------------------------------------------------
        delta = torch.zeros(self.engram.r_mem, device=self.engram.device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            self.engram._active_perturbation = (delta.to(self.engram.dtype), anchor_positions)

            logits = self.model(input_ids=prompt_ids_batch)
            if hasattr(logits, "logits"):
                logits = logits.logits

            # Compute NLL only over the target completion tokens o_i*
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = prompt_ids_batch[:, 1:].contiguous()
            shift_mask = target_mask[:, 1:].contiguous().float()

            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)

            nll_loss = (token_loss * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)
            reg_loss = 0.01 * delta.pow(2).sum()
            loss = nll_loss + reg_loss

            loss.backward()
            optimizer.step()

            # Trust-region L2 norm clamping (Equation 4 in EngramEdit)
            with torch.no_grad():
                if delta.norm() > max_delta_norm:
                    delta.mul_(max_delta_norm / (delta.norm() + 1e-8))

        self.engram._active_perturbation = None
        delta_star = delta.detach().cpu().to(self.engram.dtype)

        # -----------------------------------------------------------------
        # STAGE 2: Map Expressions to N-grams & Solve Weighted Ridge Update
        # -----------------------------------------------------------------
        idx_map = self.engram.hasher.hash_sequence(prompt_ids_batch)
        M = prompt_ids_batch.size(0)
        K = self.engram.num_heads

        # Solve independently for each N-gram order slice (2-gram slice vs 3-gram slice)
        delta_slices = torch.split(delta_star, self.engram.r_per_order, dim=-1)

        with torch.no_grad():
            for order_idx, n in enumerate(self.engram.orders):
                target_slice = delta_slices[order_idx].float()  # [r_per_order]

                # Collect active bucket IDs at anchor positions across all M expressions: [M, K]
                active_buckets = torch.stack([
                    idx_map[n][m, anchor_positions[m]] for m in range(M)
                ])
                unique_buckets, inverse_idx = torch.unique(active_buckets, return_inverse=True)
                U = unique_buckets.numel()

                # Build design matrix A in R^{M x U} where A[m, u] = count(u in expr m) / K
                A = torch.zeros(M, U, dtype=torch.float32)
                for m in range(M):
                    for k_head in range(K):
                        u_col = inverse_idx[m, k_head]
                        A[m, u_col] += 1.0 / K

                # Frequency & N-gram order penalty matrix Gamma (diagonal U x U)
                freqs = self.engram.bucket_counts[n][unique_buckets]
                gamma_diag = order_penalties[n] * (1.0 + torch.log1p(freqs)) * 1e-2
                Gamma = torch.diag(gamma_diag)

                # Closed-form Ridge solution: (A^T A + Gamma) * Delta_E = A^T * 1_M * target_slice^T
                rhs = A.sum(dim=0, keepdim=True).T @ target_slice.unsqueeze(0)  # [U, r_per_order]
                lhs = A.T @ A + Gamma                                           # [U, U]
                delta_E = torch.linalg.solve(lhs, rhs)                          # [U, r_per_order]

                # Write directly into Host-RAM pinned table & increment usage counter
                self.engram.cpu_tables[n][unique_buckets] += delta_E.to(self.engram.dtype)
                self.engram.bucket_counts[n][unique_buckets] += 1.0

        return float(nll_loss.item())
