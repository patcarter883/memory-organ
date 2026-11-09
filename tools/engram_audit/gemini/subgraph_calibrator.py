# VERBATIM from Gemini (block 2 of 3): subgraph slicing + two-phase gate calibration (Layers 0..6 of a 27B).
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SubgraphGateCalibrator:
    """
    Calibrates the GraftedEngramModule (W_Q, W_K, W_V, short_conv, and initial N-gram rows)
    using only Layers 0..target_layer (e.g., Layer 6) of a frozen 27B model.
    Requires < 4 GB VRAM instead of 60+ GB for full-model backprop.
    """
    def __init__(
        self,
        full_model: nn.Module,
        engram_module: nn.Module,  # Instance of GraftedEngramModule from previous step
        graft_layer_idx: int = 2,
        target_layer_idx: int = 6,
    ):
        self.engram = engram_module
        self.graft_idx = graft_layer_idx
        self.target_idx = target_layer_idx

        # 1. Extract ONLY the embedding layer, rotary emb, and layers [0 .. target_layer_idx]
        # (Works with standard Llama / Qwen2 / Qwen3 HuggingFace PreTrainedModel structures)
        base = full_model.model if hasattr(full_model, "model") else full_model
        self.embed_tokens = base.embed_tokens
        self.rotary_emb = getattr(base, "rotary_emb", None)

        # Slice out only the first (target_layer_idx + 1) layers and free the rest!
        self.sub_layers = nn.ModuleList(
            [base.layers[i] for i in range(target_layer_idx + 1)]
        )

        # Freeze 100% of the sliced backbone parameters
        self.embed_tokens.eval()
        for p in self.embed_tokens.parameters():
            p.requires_grad_(False)

        for layer in self.sub_layers:
            layer.eval()
            for p in layer.parameters():
                p.requires_grad_(False)

    def _run_subgraph(
        self,
        input_ids: torch.Tensor,
        use_engram: bool = False,
        record_ffn_accumulation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs strictly Layers 0 .. target_idx.
        Returns:
          h_graft: hidden state right after Layer `graft_idx` (before Engram)
          h_target: hidden state exiting `target_idx`
          ffn_delta: sum of FFN outputs from (graft_idx + 1) to target_idx
        """
        B, T = input_ids.shape
        device = input_ids.device

        h = self.embed_tokens(input_ids)
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        # Compute RoPE cos/sin once if model uses centralized rotary_emb
        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(h, position_ids)

        if use_engram:
            self.engram.prefetch_from_host(input_ids)

        h_at_graft = None
        ffn_accum = torch.zeros_like(h) if record_ffn_accumulation else None

        for idx, layer in enumerate(self.sub_layers):
            # Track pre-layer state to isolate the FFN contribution in layers > graft_idx
            h_in = h

            layer_kwargs = {"position_ids": position_ids}
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            out = layer(h, **layer_kwargs)
            h = out[0] if isinstance(out, tuple) else out

            if idx == self.graft_idx:
                h_at_graft = h.detach()
                if use_engram:
                    h = self.engram.forward_fusion(h)
            elif idx > self.graft_idx and record_ffn_accumulation:
                # Accumulate the residual trajectory across layers 3..6
                ffn_accum = ffn_accum + (h - h_in).detach()

        return h_at_graft, h, ffn_accum

    # =====================================================================
    # PHASE 1: ZERO-BACKPROP SUBSPACE ALIGNMENT (Layer 2 Only)
    # =====================================================================
    def warmup_subspace_alignment(
        self,
        calib_batches: list,  # List of [B, T] token tensors
        lr: float = 1e-3,
        lookahead_scale: float = 0.25,
    ):
        """
        Trains W_Q, W_K, W_V, and short_conv directly at Layer 2 so the Engram
        output basis aligns with the cumulative residual delta of Layers 3..6.
        Zero gradients ever pass through ANY transformer layer!
        """
        for p in self.engram.parameters():
            p.requires_grad_(True)

        # In Phase 1, temporarily perturb v_proj with tiny orthogonal weights so gradients flow to Q/K
        with torch.no_grad():
            nn.init.orthogonal_(self.engram.v_proj.weight, gain=0.01)

        opt = torch.optim.AdamW(self.engram.parameters(), lr=lr, weight_decay=1e-2)

        for step, input_ids in enumerate(calib_batches):
            opt.zero_grad()

            # 1. Forward-only pass through Layers 0..6 to capture h_2 and the L3..L6 delta
            with torch.no_grad():
                h_2, _, lookahead_delta = self._run_subgraph(
                    input_ids, use_engram=False, record_ffn_accumulation=True
                )
                self.engram.prefetch_from_host(input_ids)

            # 2. Run ONLY the Engram fusion module on h_2
            h_2_aug = self.engram.forward_fusion(h_2)
            engram_injection = h_2_aug - h_2  # [B, T, d_model]

            # 3. Cosine + L2 alignment with the scaled L3..L6 lookahead trajectory
            target_delta = lookahead_scale * lookahead_delta
            cos_loss = 1.0 - F.cosine_similarity(engram_injection, target_delta, dim=-1).mean()
            l2_loss = F.smooth_l1_loss(engram_injection, target_delta)

            loss = cos_loss + l2_loss
            loss.backward()
            opt.step()

    # =====================================================================
    # PHASE 2: CONTEXT-DROP SUBGRAPH DISTILLATION (4-Layer Local Backprop)
    # =====================================================================
    def calibrate_context_distillation(
        self,
        calib_batches: list,  # List of [B, T] token tensors (e.g., T=512)
        prefix_len: int = 256,
        lr_gates: float = 5e-4,
        lr_table: float = 5e-2,
    ):
        """
        Splits each sequence into [Prefix (256), Continuation (256)].
        - Teacher runs full 512 tokens through Layers 0..6 (no Engram).
        - Student runs ONLY the 256 Continuation tokens through Layers 0..6 WITH Engram.
        Forces Layer 2's Engram gates to inject the missing prefix lexical/entity priors
        so Layer 6's hidden state matches the full-context teacher!
        """
        for p in self.engram.parameters():
            p.requires_grad_(True)

        opt = torch.optim.AdamW(self.engram.parameters(), lr=lr_gates, weight_decay=1e-2)

        for step, full_ids in enumerate(calib_batches):
            opt.zero_grad()
            cont_ids = full_ids[:, prefix_len:]  # [B, T - prefix_len]

            # 1. TEACHER PASS (No grad, full prefix + continuation, Engram OFF)
            with torch.no_grad():
                _, h6_teacher_full, _ = self._run_subgraph(full_ids, use_engram=False)
                # Slice out only the continuation positions at Layer 6
                h6_teacher = h6_teacher_full[:, prefix_len:, :].detach()

            # 2. STUDENT PASS (Grad enabled through Layers 3..6, continuation ONLY, Engram ON)
            # We also track grad on the prefetched N-gram embeddings so the CPU table warms up!
            self.engram.prefetch_from_host(cont_ids)
            self.engram._prefetched_e.requires_grad_(True)

            # Run Layers 0..2 without grad
            with torch.no_grad():
                B, T_c = cont_ids.shape
                h = self.embed_tokens(cont_ids)
                pos_ids = torch.arange(T_c, device=cont_ids.device).unsqueeze(0).expand(B, T_c)
                pos_emb = self.rotary_emb(h, pos_ids) if self.rotary_emb is not None else None

                for idx in range(self.graft_idx + 1):
                    kw = {"position_ids": pos_ids}
                    if pos_emb is not None:
                        kw["position_embeddings"] = pos_emb
                    out = self.sub_layers[idx](h, **kw)
                    h = out[0] if isinstance(out, tuple) else out

            # Apply Grafted Engram at Layer 2 (with gradients!)
            h = self.engram.forward_fusion(h)

            # Run Layers 3..6 so gradients flow back from Layer 6 -> Layer 2 Engram gates
            for idx in range(self.graft_idx + 1, self.target_idx + 1):
                kw = {"position_ids": pos_ids}
                if pos_emb is not None:
                    kw["position_embeddings"] = pos_emb
                out = self.sub_layers[idx](h, **kw)
                h = out[0] if isinstance(out, tuple) else out

            h6_student = h

            # 3. Representation Distillation Loss at Layer 6 (Normalized MSE + Cosine)
            norm_t = F.layer_norm(h6_teacher.float(), (self.engram.d_model,))
            norm_s = F.layer_norm(h6_student.float(), (self.engram.d_model,))

            mse_loss = F.mse_loss(norm_s, norm_t)
            cos_loss = 1.0 - F.cosine_similarity(norm_s, norm_t, dim=-1).mean()
            loss = mse_loss + 0.5 * cos_loss

            loss.backward()
            opt.step()

            # 4. Sparse SGD step on the Host-RAM N-gram rows that fired in this batch
            with torch.no_grad():
                e_grad_cpu = self.engram._prefetched_e.grad.cpu()
                grad_slices = torch.split(e_grad_cpu, self.engram.r_per_order, dim=-1)
                idx_map = self.engram.hasher.hash_sequence(cont_ids)

                for order_idx, n in enumerate(self.engram.orders):
                    buckets = idx_map[n].reshape(-1)  # [B * T_c * K]
                    # Expand grad across K heads and scatter-add into pinned CPU table
                    g_slice = grad_slices[order_idx].unsqueeze(2).expand(-1, -1, self.engram.num_heads, -1)
                    g_flat = g_slice.reshape(-1, self.engram.r_per_order) / self.engram.num_heads
                    self.engram.cpu_tables[n].index_add_(0, buckets, -lr_table * g_flat.to(self.engram.dtype))
