"""Reserved-id aliases for token-keyed tables (DESIGN.md §4.2 "Unigram problem -> alias ids"). Minimal version.

A Gemma-PLE row fires on every occurrence of its token. An alias is a reserved id (E4B ships 6,227 `<unusedN>`
ids) whose main-embedding and per-layer rows start as exact copies of the source token's, so installing it is an
exact no-op; edits then go to the alias row only, and a serving-loop matcher decides which occurrences become the
alias. The tied output head must never emit an alias id (`mask_alias_logits`).

Not done here (open): matcher precision/recall on the alias / surface-variant / lexical-neighbour cohorts (§6),
tokenizer-level `<unusedN>` discovery, and a longest-match rule set. `ContextAliasRule` is a plain
(preceding tokens, token) -> alias rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn

from .gemma4_ple import gemma_text_model


@dataclass
class ContextAliasRule:
    token: int
    alias: int
    context: tuple[int, ...] = ()  # tokens that must immediately precede `token` (oldest first); () = always


@dataclass
class AliasMatcher:
    rules: list[ContextAliasRule] = field(default_factory=list)

    def add(self, token: int, alias: int, context: Sequence[int] = ()) -> None:
        self.rules.append(ContextAliasRule(int(token), int(alias), tuple(int(c) for c in context)))

    @property
    def alias_ids(self) -> list[int]:
        return sorted({r.alias for r in self.rules})

    def rewrite(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, T] -> [B, T] with matched occurrences replaced by their alias (rules matched on the ORIGINAL ids;
        longer contexts win)."""
        out = ids.clone()
        for r in sorted(self.rules, key=lambda r: len(r.context)):
            hit = ids == r.token
            k = len(r.context)
            if k:
                ok = torch.zeros_like(hit)
                if ids.shape[1] > k:
                    m = torch.ones_like(hit[:, k:])
                    for j, c in enumerate(r.context):
                        m &= ids[:, j:ids.shape[1] - k + j] == c
                    ok[:, k:] = m
                hit &= ok
            out[hit] = r.alias
        return out


@torch.no_grad()
def install_alias(model: nn.Module, token: int, alias: int) -> None:
    """Copy `token`'s main-embedding and per-layer rows into the reserved id `alias` (an exact no-op install)."""
    tm = gemma_text_model(model)
    if alias == token or not (0 < alias < tm.embed_tokens.num_embeddings):
        raise ValueError("alias must be a distinct, in-vocabulary reserved id (not 0/pad)")
    tm.embed_tokens.weight[alias] = tm.embed_tokens.weight[token]
    tm.embed_tokens_per_layer.weight[alias] = tm.embed_tokens_per_layer.weight[token]


def mask_alias_logits(logits: torch.Tensor, alias_ids: Sequence[int]) -> torch.Tensor:
    """The output head never emits an alias id."""
    if not alias_ids:
        return logits
    out = logits.clone()
    out[..., list(alias_ids)] = float("-inf")
    return out


__all__ = ["AliasMatcher", "ContextAliasRule", "install_alias", "mask_alias_logits"]
