"""Repository QA (§6.A.5; report-only in Stage 1): AST-generated items and teacher-forced scoring.

Item schema (JSONL, one object per line; the owner may append lines):
    kind        defines_module | signature | env_var_reader | import_completion | constant_value
    name        the identifier / env var the item is about
    question    natural-language prompt ending where the answer starts (e.g. "...?\\nA: `")
    cue         code-form prompt (None if the kind has no code form)
    answer      canonical answer text (appended to `question` / `cue` when scoring; NL answers close with "`")
    aliases     other accepted answers (EM only)
    evidence    "path:line"
    commit      git commit the item was generated at
    split       train | dev | test | near_dup (Split-A membership of the evidence file)
    reach_class unreachable (defines_module, env_var_reader: cue-after-answer order, §I.3) | partial
    distractors three same-package distractors for 4-way MC
Scoring per item and form (nl / cue): answer NLL (sum and per token), teacher-forced EM (greedy argmax equals
every answer token), 4-way MC by total answer log-likelihood. Conditions act through `activate` only (rows,
LoRA, injections); prefix/adjust conditions (RAG, count-LM, kNM) are not supported here.
"""
from __future__ import annotations

import ast
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .scoring import Condition

KINDS = ("defines_module", "signature", "env_var_reader", "import_completion", "constant_value")
REACH = dict(defines_module="unreachable", env_var_reader="unreachable", signature="partial",
             import_completion="partial", constant_value="partial")


def module_name(path: str) -> str:
    m = path[:-3] if path.endswith(".py") else path
    m = m.replace("/", ".")
    return m[: -len(".__init__")] if m.endswith(".__init__") else m


def _pkg(x: str, sep: str = ".") -> str:
    return x.rsplit(sep, 1)[0] if sep in x else x


def _distractors(ans: str, pool: Sequence[str], sep: str = ".", k: int = 3) -> list[str]:
    pkg = _pkg(ans, sep)
    same = [m for m in pool if m != ans and m.startswith(pkg)]
    other = [m for m in pool if m != ans and m not in same]
    r = random.Random(ans)
    r.shuffle(same)
    r.shuffle(other)
    return (same + other)[:k]


def generate_items(texts: Mapping[str, str], *, split_of: Mapping[str, str] | None = None, commit: str = "",
                   ref_vocab: Mapping[str, int] | None = None, project: str = "the codebase",
                   env_prefix: str | None = None, min_name_len: int = 6) -> list[dict]:
    """Items from {path relative to the package root's parent (e.g. 'minisgl/ple/state.py'): source}."""
    ref = ref_vocab or {}
    split_of = split_of or {}
    toplevel: dict[str, set[str]] = defaultdict(set)
    sigs: dict[tuple[str, str], tuple[str, int]] = {}
    consts: dict[tuple[str, str], tuple[str, int]] = {}
    imports: list[tuple[str, str, str, int]] = []  # (module, name, file, line)
    for f, s in sorted(texts.items()):
        try:
            tree = ast.parse(s)
        except SyntaxError:
            continue
        lines = s.splitlines()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                toplevel[node.name].add(f)
                if not isinstance(node, ast.ClassDef):
                    try:
                        sigs[(f, node.name)] = (f"def {node.name}({ast.unparse(node.args)})", node.lineno)
                    except Exception:  # noqa: BLE001
                        pass
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                tgs = node.targets if isinstance(node, ast.Assign) else [node.target]
                for tg in tgs:
                    if isinstance(tg, ast.Name):
                        toplevel[tg.id].add(f)
                        v = node.value
                        if (isinstance(tg, ast.Name) and tg.id.isupper() and isinstance(v, ast.Constant)
                                and isinstance(v.value, (int, float, str)) and not isinstance(v.value, bool)
                                and v.lineno == v.end_lineno and isinstance(node, ast.Assign)):
                            src = ast.get_source_segment(s, v)
                            line = lines[node.lineno - 1]
                            if src and line.lstrip().startswith(f"{tg.id} = "):
                                consts[(f, tg.id)] = (src, node.lineno)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for al in node.names:
                    if al.name != "*":
                        imports.append((node.module, al.name, f, node.lineno))
    mods = sorted({module_name(f) for f in texts})
    files = sorted(texts)
    items: list[dict] = []

    def add(**kw):
        f = kw.pop("file")
        kw.setdefault("aliases", [])
        kw.update(evidence=f"{f}:{kw.pop('line', 1)}", commit=commit, split=split_of.get(f, "unknown"),
                  reach_class=REACH[kw["kind"]])
        items.append(kw)

    for n, fs in sorted(toplevel.items()):
        if len(fs) != 1 or len(n) < min_name_len or n.startswith("_") or ref.get(n, 0) > 0:
            continue
        f = next(iter(fs))
        m = module_name(f)
        add(kind="defines_module", name=n, file=f, question=f"Q: In {project}, which module defines `{n}`?\nA: `",
            cue=None, answer=m, distractors=_distractors(m, mods))
        if (f, n) in sigs and sigs[(f, n)][0].count(",") >= 1:
            sig, ln = sigs[(f, n)]
            pool = [v[0] for (ff, _), v in sigs.items() if v[0].count(",") >= 1]
            add(kind="signature", name=n, file=f, line=ln, question=f"Q: What is the signature of `{n}` in {m}?\nA: `",
                cue=f"# {f}\n", answer=sig, distractors=_distractors(sig, sorted(set(pool)), sep="("))
    for (f, n), (val, ln) in sorted(consts.items()):
        if len(n) < 4 or ref.get(n, 0) > 0:
            continue
        pool = sorted({v for (_, nn), (v, _) in consts.items() if v != val})
        add(kind="constant_value", name=n, file=f, line=ln,
            question=f"Q: In {project}, what is `{n}` set to in {module_name(f)}?\nA: `",
            cue=f"# {f}\n{n} =", answer=" " + val, distractors=[" " + x for x in _distractors(val, pool, sep="\0")])
    first_import: dict[tuple[str, str], tuple[str, int]] = {}
    proj_mods = set(mods)
    for mod, n, f, ln in imports:
        if mod in proj_mods and (mod, n) not in first_import:
            first_import[(mod, n)] = (f, ln)
    names_by_mod: dict[str, list[str]] = defaultdict(list)
    for (mod, n) in first_import:
        names_by_mod[_pkg(mod)].append(n)
    for (mod, n), (f, ln) in sorted(first_import.items()):
        if len(n) < 4:
            continue
        pool = sorted(set(names_by_mod[_pkg(mod)]) - {n}) or sorted({x for (_, x) in first_import} - {n})
        r = random.Random(mod + n)
        r.shuffle(pool)
        add(kind="import_completion", name=n, file=f, line=ln,
            question=f"Q: In {project}, complete the import: `from {mod} import ...`\nA: `",
            cue=f"from {mod} import", answer=" " + n, distractors=[" " + x for x in pool[:3]])
    if env_prefix:
        pat = re.compile(r"""["'](""" + re.escape(env_prefix) + r"""[A-Z0-9_]+)["']""")
        env_files: dict[str, set[str]] = defaultdict(set)
        env_line: dict[tuple[str, str], int] = {}
        for f in files:
            for i, line in enumerate(texts[f].splitlines(), 1):
                for e in pat.findall(line):
                    env_files[e].add(f)
                    env_line.setdefault((e, f), i)
        for e, fs in sorted(env_files.items()):
            if len(fs) == 1:
                f = next(iter(fs))
                add(kind="env_var_reader", name=e, file=f, line=env_line[(e, f)],
                    question=f"Q: Which file in {project} reads the environment variable {e}?\nA: `", cue=None,
                    answer=f, distractors=_distractors(f, files, sep="/"))
    return items


def write_jsonl(items: Iterable[dict], path: str | Path) -> None:
    with open(path, "w") as fo:
        for it in items:
            fo.write(json.dumps(it) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


@torch.no_grad()
def answer_logprob(model, tok, prompt: str, answer: str) -> tuple[float, int, bool]:
    """(sum log p(answer | prompt), n answer tokens, teacher-forced EM)."""
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    if not a:
        return 0.0, 0, True
    x = torch.tensor([p + a])
    lg = model(input_ids=x, use_cache=False, logits_to_keep=len(a) + 1).logits[0, :-1].float()
    lp = torch.log_softmax(lg, -1)
    y = torch.tensor(a)
    return float(lp.gather(1, y[:, None]).sum()), len(a), bool((lg.argmax(-1) == y).all())


def score_items(model, tok, items: Sequence[dict], condition: Condition | None = None, forms=("nl", "cue")) -> list[dict]:
    """Per-item scores. NL answers are closed with a backtick (the prompt opens one)."""
    cond = condition or Condition()
    out = []
    with cond.activate(model):
        for it in items:
            r = dict(kind=it["kind"], name=it["name"], split=it.get("split"), reach_class=it.get("reach_class"))
            for form in forms:
                prompt = it["question"] if form == "nl" else it.get("cue")
                if not prompt:
                    continue
                close = "`" if form == "nl" else ""
                lp, n, em = answer_logprob(model, tok, prompt, it["answer"] + close)
                ems = [em] + [answer_logprob(model, tok, prompt, al + close)[2] for al in it.get("aliases", [])]
                cands = [lp] + [answer_logprob(model, tok, prompt, d + close)[0] for d in it.get("distractors", [])]
                r[form] = dict(answer_nll=-lp, answer_tokens=n, nll_per_token=-lp / max(n, 1), em=bool(any(ems)),
                               mc4=bool(len(cands) == 4 and int(np.argmax(cands)) == 0), mc_n=len(cands))
            out.append(r)
    return out


def summarize(scores: Sequence[dict]) -> dict:
    """Per kind and form: n, EM, MC4 accuracy (items with 4 candidates), mean answer NLL."""
    agg: dict = defaultdict(lambda: defaultdict(list))
    for s in scores:
        for form in ("nl", "cue"):
            if form in s:
                for key in (s["kind"], "all"):
                    a = agg[f"{key}/{form}"]
                    a["em"].append(s[form]["em"])
                    a["nll"].append(s[form]["answer_nll"])
                    if s[form]["mc_n"] == 4:
                        a["mc4"].append(s[form]["mc4"])
    return {k: dict(n=len(v["em"]), em=float(np.mean(v["em"])), mc4=float(np.mean(v["mc4"])) if v["mc4"] else None,
                    n_mc4=len(v["mc4"]), answer_nll=float(np.mean(v["nll"]))) for k, v in sorted(agg.items())}


def paired_summary(base: Sequence[dict], arm: Sequence[dict]) -> dict:
    """Per kind/form: arm minus base for EM and MC4 (points) with McNemar p, and answer NLL delta."""
    from ..metrics import mcnemar
    out = {}
    keys = sorted({(b["kind"], f) for b in base for f in ("nl", "cue") if f in b})
    for kind, form in keys + [("all", "nl"), ("all", "cue")]:
        idx = [i for i, b in enumerate(base) if form in b and (kind == "all" or b["kind"] == kind)]
        if not idx:
            continue
        be = np.array([base[i][form]["em"] for i in idx])
        ae = np.array([arm[i][form]["em"] for i in idx])
        mi = [i for i in idx if base[i][form]["mc_n"] == 4]
        bm = np.array([base[i][form]["mc4"] for i in mi], bool)
        am = np.array([arm[i][form]["mc4"] for i in mi], bool)
        out[f"{kind}/{form}"] = dict(n=len(idx), d_em_pts=100 * float(ae.mean() - be.mean()),
                                     em_mcnemar=mcnemar(be, ae),
                                     d_mc4_pts=100 * float(am.mean() - bm.mean()) if len(mi) else None,
                                     mc4_mcnemar=mcnemar(bm, am) if len(mi) else None,
                                     d_answer_nll=float(np.mean([arm[i][form]["answer_nll"] - base[i][form]["answer_nll"]
                                                                 for i in idx])))
    return out


__all__ = ["KINDS", "generate_items", "score_items", "summarize", "paired_summary", "write_jsonl", "read_jsonl",
           "answer_logprob", "module_name"]
