"""Row census and trainable-row policies (§I.1, §I.5): which (head, row) entries a corpus reads, count >= k and
enrichment policies, per-row reuse weights omega_j, collision statistics, n-gram loss-weight caps.

Numpy-only. Row ids are GLOBAL table rows (head offsets included), so a set of global ids is a set of (head, row)
pairs; "distinct rows" below always means distinct global ids = the sum over heads of distinct rows per head
(the census definition of S1/ingest-algo/scripts/count_rows.py).

Two different "count >= k" definitions exist in the proofs and both are implemented:
* row-level (`policy kind="row_count"`): a (head, row) read >= k times (count_rows.py: 2,385,780 rows for k=2 on
  all 250 files);
* n-gram-level (`kind="count"`): rows read by positions whose exact n-gram of the head's order occurs >= k times
  (row_policy.py: 2,040,221 rows for k=2 on the 85/15 shuffle split's train files).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..hashing import hasher_for
from ..spec import KeySpec


# ------------------------------------------------------------------------------------------------- index
class NGramIndex:
    """Per-position global rows [N, H] and exact n-gram keys for a list of documents (each document is its own
    EOS segment, hashed from a fresh history unless `contexts` gives its true preceding tokens)."""

    def __init__(self, key: KeySpec, docs: Sequence[np.ndarray], contexts: Sequence[np.ndarray] | None = None,
                 rows: bool = True):
        self.key = key
        self.V = int(key.vocab_size)
        if max(key.orders) > 1 and float(self.V) ** max(key.orders) >= 2.0 ** 63:
            raise ValueError("exact n-gram keys need V**n < 2**63")
        docs = [np.asarray(d, dtype=np.int64) for d in docs]
        self.doc_lengths = np.array([d.size for d in docs], dtype=np.int64)
        self.doc_offsets = np.concatenate([[0], np.cumsum(self.doc_lengths)])
        h = hasher_for(key)
        R, K = [], {o: [] for o in sorted(set(key.orders))}
        for i, d in enumerate(docs):
            if d.size == 0:
                continue
            ctx = None if contexts is None else np.asarray(contexts[i], dtype=np.int64)
            hist = np.concatenate([ctx if ctx is not None else np.full(key.context_len, key.eos_token_id), d])[None]
            xs = [h.shifted(hist, s)[0, -d.size:] for s in range(key.ngram_size)]
            for o in K:
                k = xs[0].copy()
                for s in range(1, o):
                    k = k + xs[s] * (self.V ** s)
                K[o].append(k)
            if rows:
                R.append(h.rows_for(d, context=ctx))
        self.n = int(self.doc_lengths.sum())
        self.keys = {o: (np.concatenate(v) if v else np.zeros(0, np.int64)) for o, v in K.items()}
        self.rows = (np.concatenate(R) if R else np.zeros((0, key.n_heads), np.int64)) if rows else None
        self.ids = np.concatenate(docs) if docs else np.zeros(0, np.int64)

    def head_keys(self, h: int) -> np.ndarray:
        return self.keys[self.key.head_orders[h]]

    def read_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """(unique global rows, read counts)."""
        assert self.rows is not None
        return np.unique(self.rows.reshape(-1), return_counts=True)


def _lookup_counts(u: np.ndarray, c: np.ndarray, q: np.ndarray) -> np.ndarray:
    if u.size == 0:
        return np.zeros(q.shape, dtype=np.int64)
    pos = np.clip(np.searchsorted(u, q), 0, u.size - 1)
    return np.where(u[pos] == q, c[pos], 0)


# ------------------------------------------------------------------------------------------------ census
def row_census(ix: NGramIndex, ks: Sequence[int] = (2, 5, 20)) -> dict:
    """The count_rows.py::census quantities for one corpus."""
    assert ix.rows is not None
    key = ix.key
    d: dict = {"tokens": ix.n}
    for o, k in ix.keys.items():
        if o == 1:
            continue
        u, c = np.unique(k, return_counts=True)
        d[f"distinct_{o}gram"] = int(u.size)
        d[f"{o}gram_singleton_frac"] = float((c == 1).mean()) if c.size else 0.0
        for kk in ks:
            d[f"distinct_{o}gram_count>={kk}"] = int((c >= kk).sum())
    per, per2, per5 = [], [], []
    for h in range(key.n_heads):
        u, c = np.unique(ix.rows[:, h], return_counts=True)
        per.append(int(u.size)); per2.append(int((c >= 2).sum())); per5.append(int((c >= 5).sum()))
    d["distinct_rows_per_head"] = per
    d["distinct_rows_per_head_count>=2"] = per2
    d["distinct_rows_per_head_count>=5"] = per5
    d["distinct_rows_total"] = int(sum(per))
    d["rows_count>=2_total"] = int(sum(per2))
    d["rows_count>=5_total"] = int(sum(per5))
    d["rows_frac_of_table"] = sum(per) / key.total_rows
    return d


def overlay_bytes(n_rows: int, w: int) -> dict:
    """Bytes of an overlay of n_rows rows of width w (count_rows.py conventions; int6-g32 for w = 160 is 130 B)."""
    i6 = (6 * w + 7) // 8 + 2 * ((w + 31) // 32)
    return dict(rows=n_rows, fp32_values=n_rows * w * 4, fp32_plus_int64_ids=n_rows * (w * 4 + 8),
                bf16_values=n_rows * w * 2, bf16_plus_int64_ids=n_rows * (w * 2 + 8), int6g32_plus_ids=n_rows * (i6 + 8),
                rowwise_adagrad_fp32_state=n_rows * (w * 4 + 4 + 8), sparseadam_fp32_state=n_rows * (w * 4 * 3 + 8))


# ------------------------------------------------------------------------------------------------ policy
@dataclass(frozen=True)
class RowPolicy:
    """kind: 'all' (every row read), 'row_count' (row read >= k times), 'count' (exact n-gram count >= k),
    'enrich' (enrichment >= r vs a reference, optionally & n-gram count >= k), 'specific' (n-gram absent from the
    reference, & count >= k)."""

    kind: str = "count"
    k: int = 2
    r: float = 4.0

    def name(self) -> str:
        return {"all": "all", "row_count": f"row_count>={self.k}", "count": f"count>={self.k}",
                "enrich": f"enrich>={self.r:g}" + (f" & count>={self.k}" if self.k > 1 else ""),
                "specific": "specific" + (f" & count>={self.k}" if self.k > 1 else "")}[self.kind]


def enrichment(ix: NGramIndex, ref: NGramIndex, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(unique n-gram keys, project counts, enrichment) with add-one smoothing on the reference (row_policy.py:
    enrich = (c/N) / (c_ref/N_ref + 1/N_ref))."""
    u, c = np.unique(ix.keys[order], return_counts=True)
    ur, cr = np.unique(ref.keys[order], return_counts=True)
    cref = _lookup_counts(ur, cr, u)
    rate_p = c / max(ix.n, 1)
    rate_r = cref / max(ref.n, 1)
    return u, c, rate_p / (rate_r + 1.0 / max(ref.n, 1))


def selected_ngrams(ix: NGramIndex, policy: RowPolicy, ref: NGramIndex | None = None) -> dict[int, np.ndarray]:
    """Sorted unique exact n-gram keys selected per order (n-gram-level policies)."""
    out = {}
    for o in ix.keys:
        u, c = np.unique(ix.keys[o], return_counts=True)
        if policy.kind == "all":
            m = np.ones(u.size, bool)
        elif policy.kind == "count":
            m = c >= policy.k
        elif policy.kind in ("enrich", "specific"):
            if ref is None:
                raise ValueError(f"policy {policy.kind} needs a reference index")
            if policy.kind == "enrich":
                _, _, e = enrichment(ix, ref, o)
                m = e >= policy.r
            else:
                ur, cr = np.unique(ref.keys[o], return_counts=True)
                m = _lookup_counts(ur, cr, u) == 0
            if policy.k > 1:
                m &= c >= policy.k
        else:
            raise ValueError(policy.kind)
        out[o] = u[m]
    return out


def select_rows(ix: NGramIndex, policy: RowPolicy, ref: NGramIndex | None = None) -> np.ndarray:
    """Sorted unique global row ids a policy makes trainable (rows read by positions with a selected n-gram)."""
    assert ix.rows is not None
    key = ix.key
    if policy.kind == "row_count":
        u, c = ix.read_counts()
        return u[c >= policy.k]
    if policy.kind == "all":
        return np.unique(ix.rows.reshape(-1))
    sel = selected_ngrams(ix, policy, ref)
    parts = []
    for o in sorted(set(key.orders)):
        m = np.isin(ix.keys[o], sel[o])
        hs = list(key.heads_of_order(o))
        parts.append(np.unique(ix.rows[m][:, hs].reshape(-1)))
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)


def per_head_counts(key: KeySpec, row_ids: np.ndarray) -> list[int]:
    off = np.asarray(key.head_offsets + (key.total_rows,), dtype=np.int64)
    return [int(x) for x in np.diff(np.searchsorted(row_ids, off))]


def reach(row_ids: np.ndarray, other: NGramIndex) -> dict:
    """Share of `other`'s positions that read at least one of `row_ids`, overall and on the first head of each
    order (row_policy.py 'any_row' column)."""
    assert other.rows is not None
    hit = np.isin(other.rows, row_ids)
    d = {"any_row": float(hit.any(axis=1).mean()) if other.n else 0.0}
    for o in sorted(set(other.key.orders)):
        h0 = other.key.heads_of_order(o)[0]
        d[f"head{h0}_order{o}"] = float(hit[:, h0].mean()) if other.n else 0.0
    return d


# --------------------------------------------------------------------------------------------- collisions
def collision_stats(ix: NGramIndex, row_ids: np.ndarray | None = None) -> dict:
    """Per head: distinct n-grams of its order, distinct rows, occupancy (rows / head size; row_policy.py's
    'collision rate per head'), the within-head collision gap (distinct n-grams - distinct rows), rows shared by
    >= 2 distinct n-grams and the share of positions reading such a shared row. Restricted to `row_ids` if given."""
    assert ix.rows is not None
    key = ix.key
    out = []
    for h in range(key.n_heads):
        r, k = ix.rows[:, h], ix.head_keys(h)
        if row_ids is not None:
            m = np.isin(r, row_ids)
            r, k = r[m], k[m]
        pairs = np.unique(np.stack([r, k], 1), axis=0) if r.size else np.zeros((0, 2), np.int64)
        ur, n_ng = np.unique(pairs[:, 0], return_counts=True)
        shared = ur[n_ng >= 2]
        out.append(dict(head=h, order=key.head_orders[h], distinct_ngrams=int(np.unique(k).size),
                        distinct_rows=int(ur.size), occupancy=ur.size / key.head_sizes[h],
                        collision_gap=int(np.unique(k).size - ur.size), shared_rows=int(shared.size),
                        positions_reading_shared_row=float(np.isin(r, shared).mean()) if r.size else 0.0))
    return {"per_head": out,
            "occupancy_first_head_per_order": {o: out[key.heads_of_order(o)[0]]["occupancy"] for o in set(key.orders)}}


# ------------------------------------------------------------------------------------------- reuse weights
def reuse_weights(row_ids: np.ndarray, project: NGramIndex, reference: NGramIndex, gamma: float = 4.0,
                  pi_other: float = 0.45) -> np.ndarray:
    """omega_j = 1 + gamma * r_j / (p_j + r_j) (§I.1) aligned to `row_ids`: p_j = reads of row j per project
    position, r_j = pi_other * reads per reference position (pi_other = share of non-project text inside project
    requests; 0.3-0.6 est.). Rows with p_j = r_j = 0 get 1."""
    up, cp = project.read_counts()
    ur, cr = reference.read_counts()
    p = _lookup_counts(up, cp, row_ids) / max(project.n, 1)
    r = pi_other * _lookup_counts(ur, cr, row_ids) / max(reference.n, 1)
    den = p + r
    frac = np.divide(r, den, out=np.zeros_like(r, dtype=np.float64), where=den > 0)
    return (1.0 + gamma * frac).astype(np.float32)


# ------------------------------------------------------------------------------------- n-gram weight cap
def ngram_loss_weights(ix: NGramIndex, order: int | None = None, pct: float = 99.9) -> tuple[np.ndarray, float]:
    """Per-position loss weights that cap each n-gram's total weight at the pct-th percentile of n-gram counts
    (§I.4 step 2): a position whose exact n-gram (default: the highest order) occurs c > cap times gets cap / c.
    Returns (weights [N] float32, cap)."""
    o = max(ix.keys) if order is None else order
    u, inv, c = np.unique(ix.keys[o], return_inverse=True, return_counts=True)
    cap = float(np.percentile(c, pct)) if c.size else 1.0
    w = np.minimum(1.0, cap / c[inv]).astype(np.float32)
    return w, cap


# ----------------------------------------------------------------------------------------------- summary
def policy_table(train: NGramIndex, policies: Sequence[RowPolicy], ref: NGramIndex | None = None,
                 evals: dict[str, NGramIndex] | None = None) -> dict:
    """Rows, per-head occupancy and reach on evaluation corpora for each policy."""
    res = {}
    for pol in policies:
        rows = select_rows(train, pol, ref)
        ph = per_head_counts(train.key, rows)
        d = dict(rows=int(rows.size), per_head=ph,
                 occupancy_first_head_per_order={o: ph[train.key.heads_of_order(o)[0]] /
                                                 train.key.head_sizes[train.key.heads_of_order(o)[0]]
                                                 for o in sorted(set(train.key.orders))})
        for name, ix in (evals or {}).items():
            d[f"reach[{name}]"] = reach(rows, ix)
        res[pol.name()] = d
    return res


__all__ = ["NGramIndex", "row_census", "overlay_bytes", "RowPolicy", "enrichment", "selected_ngrams", "select_rows",
           "per_head_counts", "reach", "collision_stats", "reuse_weights", "ngram_loss_weights", "policy_table"]
