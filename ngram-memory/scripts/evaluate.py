"""Stage-1 evaluation driver (§6.A): score a host and its arms on the project, retention and Split-B sets at the
fixed context, paired against the base, with file-cluster CIs, token classes and wall-clock per 10k tokens.

    python scripts/evaluate.py --splits SPLITS.json --out DIR \
        --sets dev[,test,splitb,r1,r2,r2ml] --conditions base,countlm,rag1k,rag4k,knm,ngm,rows:P.pt,perm:P.pt,\
random:P.pt,constbias:P.pt,lora[,rag4k+rows:P.pt] [--max-files 2] [--threads 4]

Tuned arms (countlm lambdas, kNM lam/T, NGM ratio) are tuned on Split-A dev files that are NOT scored here when
possible (`--tune-files`, taken after the scored dev files); the chosen values are written to the summary.
Rows files are SparseRowStore state dicts (torch.save(store.state_dict())); their FormatSpec comes from --fmt
(FormatSpec JSON) or defaults to the Qwen2.5 proxy graft (graft_format(hidden, qwen25_proxy_key_spec())).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QWEN25 = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--host", default=QWEN25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="dev")
    ap.add_argument("--conditions", default="base")
    ap.add_argument("--fmt", default=None, help="FormatSpec JSON for rows-based arms")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--min-context", type=int, default=1024)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None, help="per set (smoke)")
    ap.add_argument("--retention-tokens", type=int, default=64_000)
    ap.add_argument("--tune-files", type=int, default=1, help="dev files (after the scored ones) used for tuning")
    ap.add_argument("--tune-max-windows", type=int, default=None)
    ap.add_argument("--knm-tokens", type=int, default=50_000)
    ap.add_argument("--ngm-ratios", default="0.01,0.038,0.1")
    ap.add_argument("--lora-tokens", type=int, default=49_152)
    ap.add_argument("--lora-window", type=int, default=512)
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--lora-epochs", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--control-seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-classes", action="store_true")
    ap.add_argument("--r3-items", type=int, default=0, help="R3: items per task (MMLU, ARC-C, HellaSwag); 0 = off")
    ap.add_argument("--ref-vocab-cache", default=None)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    import torch
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    from ngram_memory.eval.metrics import class_breakdown, paired, retention, summary as nll_summary
    from ngram_memory.eval.report import render
    from ngram_memory.eval.stage1 import datasets as ds
    from ngram_memory.eval.stage1 import tokclass as tc
    from ngram_memory.eval.stage1.scoring import Composite, Condition, ScoreResult, plan_windows, score
    from ngram_memory.eval.stage1.splits import Splits

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    splits = Splits.load(a.splits)
    tok = AutoTokenizer.from_pretrained(a.host)
    model = AutoModelForCausalLM.from_pretrained(a.host, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    eos = int(model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else model.config.eos_token_id[0])
    V = int(model.config.vocab_size)
    S: dict = dict(host=a.host, splits=a.splits, split_summary=splits.summary(), args=vars(a), sets={}, tuning={})

    def save():
        S["wall_s_total"] = time.time() - T0
        S["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))

    def windows_for(docs, set_name, max_windows=None):
        mc = 1 if set_name == "splitb" else a.min_context
        w = plan_windows(docs, a.context, a.stride, mc)
        mw = max_windows if max_windows is not None else a.max_windows
        return w[:mw] if mw else w

    # ------------------------------------------------------------------------------------------- eval sets
    want = a.sets.split(",")
    sets: dict[str, list] = {}
    long_enough = lambda docs: [d for d in docs if len(d) > a.min_context]  # noqa: E731  (others have no window)
    dev_all = long_enough(ds.project_docs(splits, tok, "a", "dev"))
    n_scored_dev = a.max_files if a.max_files is not None else len(dev_all)
    if "dev" in want:
        sets["dev"] = dev_all[:n_scored_dev]
    if "test" in want:
        sets["test"] = long_enough(ds.project_docs(splits, tok, "a", "test"))[: a.max_files]
    if "splitb" in want:
        sets["splitb"] = ds.split_b_docs(splits, tok, a.max_files)
    if "r1" in want:
        sets["r1"] = ds.take_tokens(ds.wikitext_docs(tok), a.retention_tokens, a.min_context)
    if "r2" in want:
        sets["r2"] = ds.take_tokens(ds.stdlib_docs(tok), a.retention_tokens, a.min_context)
    if "r2ml" in want:
        sets["r2ml"] = ds.take_tokens(ds.ml_codebase_docs(tok), a.retention_tokens, a.min_context)
    tune_docs = dev_all[n_scored_dev:n_scored_dev + a.tune_files] if "dev" in sets else dev_all[: a.tune_files]
    if not tune_docs:
        tune_docs = dev_all[: a.tune_files]
        S["tuning"]["warning"] = "no unscored dev files left: tuning on scored dev files"
    S["tuning"]["files"] = [d.name for d in tune_docs]
    tune_w = windows_for(tune_docs, "dev", a.tune_max_windows)
    log("sets", {k: (len(v), sum(len(d) for d in v)) for k, v in sets.items()}, "tune files", S["tuning"]["files"])
    if "splitb" in sets:
        S.setdefault("warnings", []).append(
            "splitb: data-dependent arms (countlm, rag*, knm, lora) are built from Split-A train files at HEAD, not "
            "from the tree at T_cut (§6.A.4); their Split-B numbers are contaminated by post-cut code")

    # ------------------------------------------------------------------------------------------ conditions
    fmt = None

    def get_fmt():
        nonlocal fmt
        if fmt is None:
            from ngram_memory.spec import FormatSpec, graft_format, qwen25_proxy_key_spec
            fmt = FormatSpec.from_json(Path(a.fmt).read_text()) if a.fmt else graft_format(model.config.hidden_size,
                                                                                            qwen25_proxy_key_spec())
        return fmt

    cache: dict = {}

    def train_stream():
        if "stream" not in cache:
            cache["stream"] = ds.train_stream(splits, tok, eos)
        return cache["stream"]

    def build(spec: str) -> Condition:
        if "+" in spec:
            return Composite([build(x) for x in spec.split("+")], name=spec.replace(":", "="))
        kind, _, arg = spec.partition(":")
        if kind == "base":
            return Condition()
        if kind == "countlm":
            from ngram_memory.eval.stage1.baselines import countlm as cl
            if "kn" not in cache:
                t = time.time()
                docs_tr = [tok(splits.document(p), add_special_tokens=False)["input_ids"] for p in splits.files("a", "train")]
                cache["kn"] = cl.KNTrigram(docs_tr, V, eos)
                S["tuning"]["countlm_build_s"] = time.time() - t
            kn = cache["kn"]
            base_t = score(model, tune_docs, tune_w)
            pk, pc = cl.component_target_probs(kn, tune_docs, base_t)
            tu = cl.tune_lambdas(base_t, pk, pc)
            S["tuning"]["countlm"] = {k: v for k, v in tu.items() if k != "grid"}
            S["tuning"]["countlm_headroom_tune_docs"] = cl.headroom(kn, tune_docs, base_t)
            return cl.CountLMCondition(kn, tu["lam_n"], tu["lam_c"])
        if kind in ("rag1k", "rag4k"):
            from ngram_memory.eval.stage1.baselines.bm25_rag import BM25Retriever, BM25RAGCondition
            if "bm25" not in cache:
                cache["bm25"] = BM25Retriever({p: splits.text(p) for p in splits.files("a", "train")})
            return BM25RAGCondition(cache["bm25"], tok, budget=1024 if kind == "rag1k" else 4096)
        if kind == "knm":
            from ngram_memory.eval.stage1.baselines import knm
            from ngram_memory.eval.stage1.scoring import Doc
            t = time.time()
            tw = ds.train_windows(train_stream(), a.lora_window, n=max(1, a.knm_tokens // a.lora_window), seed=0)
            docs_k = [Doc(f"trainwin{i}", w) for i, w in enumerate(tw)]
            from ngram_memory.eval.stage1.scoring import Window
            store = knm.KNMDatastore.build(model, docs_k, [Window(i, 0, 1, len(w)) for i, w in enumerate(tw)])
            tu = knm.tune(model, store, tune_docs, tune_w)
            S["tuning"]["knm"] = {k: v for k, v in tu.items() if k != "grid"} | dict(build_s=time.time() - t, **store.meta)
            return knm.KNMCondition(store, tu["lam"], tu["temp"])
        if kind == "ngm":
            from ngram_memory.eval.stage1.baselines.ngm import NGMCondition
            ratios = [float(x) for x in (arg or a.ngm_ratios).split(",")]
            best = None
            res_t = {}
            base_t = score(model, tune_docs, tune_w)
            for r in ratios:
                c = NGMCondition(model, layer_idx=1, ratio=r, eos=eos)
                nll = float(score(model, tune_docs, tune_w, c).nll.mean())
                res_t[r] = nll
                if best is None or nll < best[1]:
                    best = (r, nll)
            S["tuning"]["ngm"] = dict(base_dev_nll=float(base_t.nll.mean()), by_ratio=res_t, ratio=best[0])
            return NGMCondition(model, layer_idx=1, ratio=best[0], eos=eos)
        if kind in ("rows", "perm", "random", "constbias"):
            import torch as _t
            from ngram_memory.eval.stage1.baselines import controls as ct
            from ngram_memory.eval.stage1.conditions import RowsCondition
            from ngram_memory.tables import SparseRowStore
            f = get_fmt()
            st = SparseRowStore.from_state_dict(_t.load(arg, map_location="cpu", weights_only=True))
            stem = Path(arg).stem
            if kind == "rows":
                return RowsCondition(f, st, name=f"rows[{stem}]")
            if kind == "perm":
                return RowsCondition(f, ct.permuted_keys_store(st, f.key, a.control_seed), name=f"permkeys[{stem}]")
            if kind == "random":
                return RowsCondition(f, ct.random_rows_store(st, a.control_seed), name=f"randrows[{stem}]")
            rc = RowsCondition(f, st)
            bias = ct.mean_graft_output(model, rc.graft, tune_docs, tune_w)
            S["tuning"][f"constbias[{stem}]"] = dict(bias_norm=float(bias.norm()), calibrated_on=S["tuning"]["files"])
            return ct.ConstantBiasCondition(bias, layer_idx=f.module.layer_idx, name=f"constbias[{stem}]")
        if kind == "lora":
            import torch as _t
            from ngram_memory.eval.stage1.baselines import lora as lo
            if not lo.lora_modules(model):
                lo.add_lora(model, r=a.lora_r, alpha=2.0 * a.lora_r)
            if arg:
                lo.load_lora_state_dict(model, _t.load(arg, map_location="cpu", weights_only=True))
                S["tuning"]["lora"] = dict(loaded=arg, params=lo.n_lora_params(model))
            else:
                n = max(1, a.lora_tokens // a.lora_window)
                tw = ds.train_windows(train_stream(), a.lora_window, n=n, seed=0)
                info = lo.train_lora(model, list(tw), epochs=a.lora_epochs, lr=a.lora_lr,
                                     log=lambda r: log("lora", r) if r["step"] % 10 == 0 else None)
                _t.save(lo.lora_state_dict(model), out / "lora_adapter.pt")
                S["tuning"]["lora"] = {k: v for k, v in info.items() if k != "history"} | dict(
                    r=a.lora_r, lr=a.lora_lr, window=a.lora_window, adapter=str(out / "lora_adapter.pt"),
                    loss_first=info["history"][0]["loss"] if info["history"] else None)
            return lo.LoRACondition(model)
        raise ValueError(f"unknown condition {spec}")

    # ------------------------------------------------------------------------------------- classes per set
    def classes_for(set_name, docs, res: ScoreResult):
        if a.no_classes or set_name not in ("dev", "test", "splitb"):
            return None
        if "ref" not in cache:
            cache["ref"] = tc.reference_vocab(tc.default_reference_dirs(), a.ref_vocab_cache or out / "ref_vocab.json")
            allp = splits.files("a", "train") + splits.files("a", "dev") + splits.files("a", "test") + splits.files("a", "near_dup")
            cache["defs"] = tc.project_definitions({p: splits.text(p) for p in allp})
        sk = "b" if set_name == "splitb" else "a"  # Split B: seen n-grams / 16-grams of the tree at T_cut
        if f"seen_{sk}" not in cache:
            at = splits.data["split_b"]["t_cut"] if sk == "b" else None
            tr = [np.asarray(tok(splits.document(p, at=at), add_special_tokens=False)["input_ids"])
                  for p in splits.files(sk, "train")]
            cache[f"seen_{sk}"] = tc.NGramSeen(tr, V, eos)
            cache[f"dup_{sk}"] = tc.NGramIndex(tr, 16)
        per_doc = []
        for d in docs:
            c = tc.token_classes(d.text, d.offsets, cache["ref"], cache["defs"], d.name)
            c["cat"] = cache[f"seen_{sk}"].categories(d.ids)
            c["dup16"] = cache[f"dup_{sk}"].in_train(d.ids)
            per_doc.append(c)
        return tc.class_masks(tc.gather_targets(per_doc, res.doc, res.pos))

    def label(spec: str) -> str:
        return "+".join(p.split(":")[0] + (f"[{Path(p.split(':', 1)[1]).stem}]" if ":" in p else "")
                        for p in spec.split("+"))

    # -------------------------------------------------------------------------------------------- run
    specs = a.conditions.split(",")
    if "base" not in specs:
        specs = ["base"] + specs
    import gc
    plans = {k: windows_for(v, k) for k, v in sets.items()}
    base_res: dict[str, ScoreResult] = {}
    masks: dict = {}
    for spec in specs:  # build, score on every set, free (one arm in memory at a time)
        t = time.time()
        try:
            cond = build(spec)
        except Exception as e:  # noqa: BLE001  (record and continue: one broken arm must not kill the run)
            log("build failed", spec, repr(e))
            S["tuning"][f"error:{label(spec)}"] = repr(e)
            continue
        S["tuning"].setdefault("build_s", {})[label(spec)] = time.time() - t
        log("built", label(spec), json.dumps(cond.describe(), default=str)[:300])
        is_base = spec == "base"
        for set_name, docs in sets.items():
            R = S["sets"].setdefault(set_name, {})
            with_kl = set_name.startswith("r") and not is_base and not cond.provides_prefix
            res = score(model, docs, plans[set_name], cond, with_kl=with_kl)
            res.save(out / f"{set_name}__{label(spec)}.npz")
            groups = np.asarray([res.groups[i] for i in res.doc])
            r = dict(wall_s=res.meta["wall_s"], wall_s_per_10k=res.wall_per_10k(), forward_tokens=res.meta["forward_tokens"],
                     threads=res.meta["threads"], describe=cond.describe())
            if is_base:
                base_res[set_name] = res
                r.update(nll_summary(res.nll, groups, res.hit))
                masks[set_name] = classes_for(set_name, docs, res)
                if masks[set_name] is not None:
                    R["_class_fractions"] = {k: float(v.mean()) for k, v in masks[set_name].items()}
            else:
                b = base_res[set_name]
                if not np.array_equal(b.key(), res.key()):
                    raise RuntimeError("targets differ from base")
                r["paired"] = (retention(b.nll, res.nll, groups, res.kl, res.agree) if with_kl else
                               paired(b.nll, res.nll, groups, base_hit=b.hit, cond_hit=res.hit))
                if masks.get(set_name) is not None:
                    r["classes"] = class_breakdown(b.nll, res.nll, groups, masks[set_name])
            R[label(spec)] = r
            p = r.get("paired", {})
            log(set_name, label(spec), f"tok={len(res)}", f"nll={float(res.nll.mean()):.4f}",
                f"rel%={p.get('rel_pct', 0):+.3f}", f"s/10k={r['wall_s_per_10k']:.1f}")
            save()
        hook_only = not cond.provides_prefix and type(cond).adjust is Condition.adjust and not isinstance(cond, Composite)
        if a.r3_items and hook_only:  # R3 capability (forward-only MC): arms acting only through hooks
            from ngram_memory.eval.stage1 import capability as cap
            if "r3_items" not in cache:
                cache["r3_items"] = [x for t in ("mmlu", "arc_challenge", "hellaswag") for x in cap.load_items(t, a.r3_items)]
            t = time.time()
            sc = cap.score_mc(model, tok, cache["r3_items"], cond)
            R3 = S["sets"].setdefault("r3", {})
            if is_base:
                cache["r3_base"] = sc
                R3["base"] = dict(n=len(sc), acc=float(np.mean([x["acc"] for x in sc])),
                                  acc_norm=float(np.mean([x["acc_norm"] for x in sc])), wall_s=time.time() - t)
            else:
                R3[label(spec)] = dict(paired_mc=cap.paired_mc(cache["r3_base"], sc), wall_s=time.time() - t)
            log("r3", label(spec), f"{time.time() - t:.0f}s")
            save()
        if not spec.startswith("lora"):
            del cond
            gc.collect()
    save()
    (out / "report.md").write_text(render(out / "summary.json"))
    log("done", f"{time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
