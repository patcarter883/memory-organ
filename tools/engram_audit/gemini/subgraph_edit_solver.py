# VERBATIM from Gemini (block 3 of 3): EngramEdit Stage 1 on a 7-layer slice (layers 0..6) with a
# context-enriched layer-6 teacher, then the same ridge Stage 2.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class SubgraphEngramEditSolver:
    """
    Two-stage EngramEdit solver that operates strictly on a 7-layer slice (Layers 0..6).
    - Pre-caches Layers 0..2 outside the optimization loop.
    - Optimizes delta_i* by running ONLY Layer 2 Fusion + Layers 3..6 against a
      Context-Enriched Layer-6 Teacher trajectory.
    """
    def __init__(
        self,
        calibrator,  # Instance of SubgraphGateCalibrator holding sub_layers[0..6] & engram
    ):
        self.calibrator = calibrator
        self.engram = calibrator.engram
        self.embed_tokens = calibrator.embed_tokens
        self.rotary_emb = calibrator.rotary_emb
        self.sub_layers = calibrator.sub_layers
        self.graft_idx = calibrator.graft_idx    # 2
        self.target_idx = calibrator.target_idx  # 6

    @torch.no_grad()
    def _extract_teacher_targets(
        self,
        teacher_ids_batch: torch.Tensor,  # [M, T_prefix + T_expr]
        prefix_len: int,
    ) -> torch.Tensor:
        """
        Runs [Context_Spec + Expression] through Layers 0..6 (Engram OFF)
        and extracts the contextualized Layer-6 hidden states for the expression slice.
        """
        _, h6_full, _ = self.calibrator._run_subgraph(teacher_ids_batch, use_engram=False)
        # Slice out only the expression tokens so shape matches bare student_ids_batch: [M, T_expr, d_model]
        return h6_full[:, prefix_len:, :].detach()

    @torch.no_grad()
    def _cache_student_up_to_graft(
        self,
        student_ids_batch: torch.Tensor,  # [M, T_expr] (No context prefix!)
    ):
        """
        Runs Layers 0..2 ONCE before the optimization loop so the 20-step Adam loop
        only executes Layer 2 Engram Fusion + Layers 3..6.
        """
        M, T_expr = student_ids_batch.shape
        device = student_ids_batch.device

        h = self.embed_tokens(student_ids_batch)
        pos_ids = torch.arange(T_expr, device=device).unsqueeze(0).expand(M, T_expr)
        pos_emb = self.rotary_emb(h, pos_ids) if self.rotary_emb is not None else None

        for idx in range(self.graft_idx + 1):
            kw = {"position_ids": pos_ids}
            if pos_emb is not None:
                kw["position_embeddings"] = pos_emb
            out = self.sub_layers[idx](h, **kw)
            h = out[0] if isinstance(out, tuple) else out

        # Prefetch baseline N-gram vectors into self.engram._prefetched_e once
        self.engram.prefetch_from_host(student_ids_batch)
        return h.detach(), pos_ids, pos_emb

    def apply_subgraph_edit(
        self,
        teacher_ids_batch: torch.Tensor,  # [M, T_prefix + T_expr] (With docstring/spec prefix)
        student_ids_batch: torch.Tensor,  # [M, T_expr] (Bare expressions without prefix)
        prefix_len: int,                  # Length of prepended spec in teacher_ids_batch
        anchor_positions: List[int],      # Index q_i(x) in student_ids_batch for each expression
        lookahead_window: int = 8,        # Number of tokens after q_i(x) to match at Layer 6
        steps: int = 20,
        lr: float = 0.08,
        clamp_rho: float = 0.45,
        order_penalties: Dict[int, float] = {2: 10.0, 3: 1.0},
    ) -> float:
        M, T_expr = student_ids_batch.shape
        device = self.engram.device

        # Ensure all GPU weights are frozen
        for p in self.engram.parameters():
            p.requires_grad_(False)

        # 1. Capture Layer-6 Teacher representations (1 forward pass of Layers 0..6)
        h6_teacher = self._extract_teacher_targets(teacher_ids_batch, prefix_len)

        # 2. Pre-cache Student Layer-2 hidden states & baseline N-gram prefetch (1 pass of Layers 0..2)
        h2_cached, pos_ids, pos_emb = self._cache_student_up_to_graft(student_ids_batch)
        base_e = self.engram._prefetched_e.detach()

        anchor_norms = torch.stack([base_e[m, pos].norm() for m, pos in enumerate(anchor_positions)]).mean()
        max_delta_norm = (clamp_rho * anchor_norms).clamp(min=0.1)

        # Build spatial weight mask [M, T_expr, 1] emphasizing [q_i(x) .. q_i(x) + lookahead_window]
        # The anchor token gets weight 2.0; the lookahead tokens attending to it get exponentially decaying weights
        window_mask = torch.zeros(M, T_expr, 1, device=device, dtype=torch.float32)
        for m, q_pos in enumerate(anchor_positions):
            end_pos = min(T_expr, q_pos + lookahead_window + 1)
            span_len = end_pos - q_pos
            decay = torch.exp(-0.25 * torch.arange(span_len, device=device, dtype=torch.float32))
            decay[0] = 2.0  # Anchor token priority
            window_mask[m, q_pos:end_pos, 0] = decay

        # Pre-normalize teacher target across d_model for stable MSE + Cosine matching
        t_float = h6_teacher.float()
        t_normed = F.layer_norm(t_float, (self.engram.d_model,))

        # -----------------------------------------------------------------
        # STAGE 1: Optimize shared delta_i* across Layers 3..6 ONLY
        # -----------------------------------------------------------------
        delta = torch.zeros(self.engram.r_mem, device=device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            self.engram._active_perturbation = (delta.to(self.engram.dtype), anchor_positions)

            # Inject perturbed Engram memory into cached Layer 2 state
            h = self.engram.forward_fusion(h2_cached)

            # Propagate strictly through Layers 3..6
            for idx in range(self.graft_idx + 1, self.target_idx + 1):
                kw = {"position_ids": pos_ids}
                if pos_emb is not None:
                    kw["position_embeddings"] = pos_emb
                out = self.sub_layers[idx](h, **kw)
                h = out[0] if isinstance(out, tuple) else out

            s_float = h.float()
            s_normed = F.layer_norm(s_float, (self.engram.d_model,))

            # Compute Window-Weighted Layer-6 Representation Loss (MSE + Direction Cosine)
            token_mse = (s_normed - t_normed).pow(2).mean(dim=-1, keepdim=True)  # [M, T_expr, 1]
            token_cos = 1.0 - F.cosine_similarity(s_float, t_float, dim=-1).unsqueeze(-1)

            distill_loss = ((token_mse + 0.5 * token_cos) * window_mask).sum() / window_mask.sum()
            reg_loss = 0.005 * delta.pow(2).sum()
            loss = distill_loss + reg_loss

            loss.backward()
            optimizer.step()

            # Trust-region L2 norm clamping
            with torch.no_grad():
                if delta.norm() > max_delta_norm:
                    delta.mul_(max_delta_norm / (delta.norm() + 1e-8))

        self.engram._active_perturbation = None
        delta_star = delta.detach().cpu().to(self.engram.dtype)

        # -----------------------------------------------------------------
        # STAGE 2: Closed-Form Frequency-Penalized Ridge Allocation (CPU)
        # -----------------------------------------------------------------
        idx_map = self.engram.hasher.hash_sequence(student_ids_batch)
        K = self.engram.num_heads
        delta_slices = torch.split(delta_star, self.engram.r_per_order, dim=-1)

        with torch.no_grad():
            for order_idx, n in enumerate(self.engram.orders):
                target_slice = delta_slices[order_idx].float()
                active_buckets = torch.stack([
                    idx_map[n][m, anchor_positions[m]] for m in range(M)
                ])
                unique_buckets, inverse_idx = torch.unique(active_buckets, return_inverse=True)
                U = unique_buckets.numel()

                A = torch.zeros(M, U, dtype=torch.float32)
                for m in range(M):
                    for k_head in range(K):
                        A[m, inverse_idx[m, k_head]] += 1.0 / K

                freqs = self.engram.bucket_counts[n][unique_buckets]
                gamma_diag = order_penalties[n] * (1.0 + torch.log1p(freqs)) * 1e-2
                Gamma = torch.diag(gamma_diag)

                rhs = A.sum(dim=0, keepdim=True).T @ target_slice.unsqueeze(0)
                lhs = A.T @ A + Gamma
                delta_E = torch.linalg.solve(lhs, rhs)

                self.engram.cpu_tables[n][unique_buckets] += delta_E.to(self.engram.dtype)
                self.engram.bucket_counts[n][unique_buckets] += 1.0

        return float(distill_loss.item())
