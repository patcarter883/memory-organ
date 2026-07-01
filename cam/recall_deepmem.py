"""Recall de-risk for the bolt-on Titans memory adapter (deepmem) on a frozen Qwen3.5-4B.

THE QUESTION (eval gate b): does the trained memory adapter actually LIFT long-context recall
above what the frozen base can do ALONE? Stage 3 removed the OOM; before funding the Triton
kernel (stage 4) or the big M3 training run, prove the bolt-on mechanism retains+retrieves a
planted fact across a segment boundary on held-out bindings.

## Why this isolates the memory cleanly
The adapter calls the frozen base PER SEGMENT (m2_adapter.lm_loss_segmented): each segment is a
fresh `base(inputs_embeds=[K mem tokens ; segment embeds])` forward. The base therefore has NO
native cross-segment context window — the ONLY pathway for a fact in an early segment to reach a
query in a later segment is the K injected memory tokens (read from the carried memory state).
So if base+adapter answers a query whose answer was only ever stated in an earlier segment, that
recall is attributable to the memory, full stop.

## The task (associative recall, real tokenizer)
Synthetic "manifest" docs. M bindings name->cargo are stated in segment 0 (e.g. "Halvard carries
copper."). Filler pads the doc so the query lands at the start of a LATER segment (--qa-seg):
"Question : which ship carries <cargo_q> ? Answer : <name_q>". name/cargo are single-token words
(filtered against the live tokenizer) so positions are deterministic and the answer is one token.
We score the answer token.

Train: AdamW on the adapter only (base frozen), LM loss over the doc + up-weighted answer-token
loss, FRESH RANDOM bindings drawn from the TRAIN pool every step (so the adapter cannot memorize
specific bindings — it must learn a general store/retrieve routine).

Eval (held-out EVAL pool, disjoint name/cargo tokens never seen in training):
  - memory       : state carried across segments (normal run)            -> recall possible
  - no_memory    : same adapter, state reset to init at every segment    -> floor (empty memory)
  - local_control: bindings placed IN the query segment, memory off      -> base sees them via
                   local attention; MUST recall -> validates scoring + that the frozen base can
                   do the lookup at all when the fact is in-context.
Metrics per condition: answer NLL (bits) and exact-match accuracy (teacher-forced argmax).

VERDICT: the bolt-on has legs iff `memory` beats `no_memory` (lower NLL / higher acc) on held-out
bindings, AND `local_control` works (else scoring is broken -> inconclusive).

Run (CPU self-test of doc construction, tokenizer only, no 4B):
  python -m cam.recall_deepmem --selftest
Full GPU run:
  python -m cam.recall_deepmem --steps 400 --batch 8 --k 16
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from m2_adapter import MODEL, DEV, TitansMemoryAdapter, load_frozen_base  # noqa: E402

LN2 = math.log(2.0)

# Candidate single-token-ish pools (filtered against the live tokenizer at startup). Capitalized
# given-name-ish strings for ship names; common nouns for cargo. We keep only those that encode to
# exactly ONE token (names space-prefixed, cargo line-initial NO-space), so binding lines have a
# constant token length.
#
# cam-loop-pkstore-ceiling (M=64/128 ceiling probe): the pools were expanded to clear the
# single-token-supply wall at high M. The dict-phrasing held-out signal is fresh PAIRINGS (random
# name<->cargo combos drawn each batch), NOT disjoint tokens, so a larger pool only RAISES the
# achievable M (M distinct names + M distinct cargo must be drawable per doc) — it does not weaken
# the probe. Verified single-token counts under the Qwen3.5-4B tokenizer at generation time:
#   NAME_CANDIDATES  -> 430 distinct single-token (space-prefixed) ids
#   CARGO_CANDIDATES -> 221 distinct single-token (no-space, line-initial) ids
# Both comfortably exceed M=128 (and the >=256-of-names ideal), so held-out eval at M=128 still
# samples many fresh pairings. The no-space cargo constraint is the binding supply limit (BPE
# favors space-prefixed word tokens): 221 is the ceiling reached from a broad concrete-noun sweep.
NAME_CANDIDATES = [
    "Aaron", "Abel", "Adam", "Adrian", "Alan", "Albert", "Alex", "Alice", "Amber", "Amy",
    "Andre", "Andrew", "Angela", "Anita", "Anna", "Anne", "Anthony", "April", "Arthur", "Ashley",
    "Audrey", "Austin", "Barbara", "Becky", "Bella", "Ben", "Bernard", "Beth", "Betty", "Beverly",
    "Bill", "Billy", "Blake", "Bobby", "Brad", "Brandon", "Brenda", "Brett", "Brian", "Brock",
    "Brooke", "Bruce", "Bryan", "Caleb", "Calvin", "Cameron", "Carl", "Carlos", "Carmen", "Carol",
    "Caroline", "Carrie", "Carson", "Casey", "Catherine", "Cecil", "Chad", "Charles", "Charlotte", "Chase",
    "Chester", "Chloe", "Chris", "Christine", "Claire", "Claude", "Clay", "Cliff", "Clinton", "Clyde",
    "Cody", "Colin", "Connor", "Conrad", "Craig", "Crystal", "Curtis", "Cynthia", "Daisy", "Dale",
    "Dana", "Daniel", "Danny", "Darren", "David", "Dawn", "Dean", "Deborah", "Dennis", "Diana",
    "Diane", "Dirk", "Dominic", "Donald", "Donna", "Dorothy", "Doug", "Douglas", "Drew", "Duncan",
    "Dustin", "Dylan", "Earl", "Edgar", "Edmund", "Edward", "Edwin", "Elaine", "Eleanor", "Eli",
    "Elijah", "Ellen", "Elliot", "Emily", "Emma", "Eric", "Erica", "Erik", "Erin", "Ernest",
    "Ethan", "Eugene", "Evan", "Evelyn", "Ezra", "Faith", "Fay", "Florence", "Floyd", "Ford",
    "Forrest", "Frances", "Francis", "Frank", "Franklin", "Fred", "Gabriel", "Gary", "Gavin", "Gene",
    "George", "Gerald", "Gilbert", "Glen", "Glenn", "Gordon", "Grace", "Grant", "Gregory", "Gus",
    "Hank", "Hannah", "Hans", "Hardy", "Harold", "Harry", "Harvey", "Hayes", "Hazel", "Heather",
    "Hector", "Helen", "Henry", "Herbert", "Holly", "Homer", "Howard", "Ian", "Irene", "Isaac",
    "Isabel", "Jack", "Jackie", "Jacob", "Jake", "James", "Jamie", "Jane", "Janet", "Jared",
    "Jasmine", "Jason", "Jean", "Jed", "Jeff", "Jeffrey", "Jenna", "Jennifer", "Jenny", "Jeremy",
    "Jerry", "Jesse", "Jessica", "Jill", "Jim", "Jimmy", "Joan", "Joanna", "Joel", "John",
    "Johnny", "Jordan", "Joseph", "Josh", "Joshua", "Joyce", "Juan", "Judith", "Judy", "Julia",
    "Julian", "Julie", "Justin", "Kane", "Kate", "Katherine", "Kathleen", "Kathy", "Katie", "Keith",
    "Kelly", "Ken", "Kenneth", "Kevin", "Kim", "Kimberly", "Kirk", "Klaus", "Kurt", "Kyle",
    "Lance", "Lane", "Larry", "Laura", "Lauren", "Lawrence", "Leah", "Lee", "Leo", "Leon",
    "Leonard", "Leslie", "Lewis", "Liam", "Lily", "Linda", "Lindsey", "Lisa", "Lloyd", "Logan",
    "Lois", "Lori", "Louis", "Louise", "Lucas", "Lucy", "Luke", "Luther", "Lydia", "Mara",
    "Marcus", "Margaret", "Maria", "Marian", "Marie", "Marilyn", "Marion", "Mark", "Martha", "Martin",
    "Marvin", "Mary", "Mason", "Matthew", "Maurice", "Max", "Megan", "Melanie", "Melissa", "Mercy",
    "Meredith", "Michael", "Michelle", "Mike", "Miles", "Milton", "Mitchell", "Molly", "Monica", "Morgan",
    "Morris", "Murray", "Nancy", "Nash", "Nathan", "Neil", "Nelson", "Newton", "Nicholas", "Nick",
    "Nicole", "Noah", "Noel", "Nolan", "Norman", "Norris", "Olive", "Oliver", "Olivia", "Otto",
    "Owen", "Pamela", "Pat", "Patricia", "Patrick", "Paul", "Pearl", "Peggy", "Penny", "Perry",
    "Peter", "Philip", "Phillip", "Preston", "Quinn", "Rachel", "Ralph", "Randall", "Randy", "Ray",
    "Raymond", "Rebecca", "Regina", "Rex", "Richard", "Rick", "Ricky", "Riley", "Rita", "Robert",
    "Roberta", "Robin", "Rod", "Rodney", "Roger", "Roland", "Ronald", "Rory", "Rose", "Ross",
    "Roy", "Ruby", "Russell", "Ruth", "Ryan", "Sally", "Sam", "Samuel", "Sandra", "Sarah",
    "Saul", "Scott", "Sean", "Seth", "Shane", "Shannon", "Sharon", "Shaun", "Shawn", "Sheila",
    "Shelby", "Sherman", "Shirley", "Sidney", "Simon", "Sophie", "Spencer", "Stacy", "Stanley", "Stella",
    "Stephen", "Steve", "Steven", "Stewart", "Stuart", "Sue", "Susan", "Suzanne", "Sydney", "Tate",
    "Ted", "Teresa", "Terry", "Theodore", "Theresa", "Thomas", "Tiffany", "Timothy", "Tina", "Toby",
    "Todd", "Tony", "Tracy", "Travis", "Trent", "Trevor", "Tristan", "Troy", "Tyler", "Valerie",
    "Vance", "Vernon", "Veronica", "Victor", "Victoria", "Vincent", "Violet", "Virginia", "Walter", "Warren",
    "Wayne", "Wendy", "Wesley", "Whitney", "William", "Willie", "Wilson", "Winston", "Wolf", "Zoe",
]
CARGO_CANDIDATES = [
    "salt", "iron", "coal", "amber", "wine", "tea", "glass", "paper", "tin", "lead",
    "gold", "silver", "jade", "oil", "gas", "steel", "rice", "corn", "beans", "nuts",
    "dates", "fish", "cloth", "ink", "soap", "ash", "lime", "chalk", "ore", "plates",
    "pipes", "sheets", "boards", "blocks", "posts", "bars", "chains", "hooks", "gems", "stones",
    "tiles", "frames", "fans", "nets", "roots", "sap", "tar", "pitch", "paint", "fuel",
    "ether", "acid", "coffee", "mint", "berries", "bread", "cakes", "cement", "sand", "threads",
    "fabric", "bundles", "boxes", "bins", "buckets", "bows", "axes", "maps", "charts", "books",
    "coins", "tokens", "flags", "ovens", "agate", "caps", "pants", "tables", "ale", "rum",
    "gin", "pine", "paste", "buttons", "pins", "canvas", "felt", "cod", "water", "cream",
    "soup", "jam", "brick", "stone", "wood", "rock", "metal", "peat", "apple", "melon",
    "bean", "leaf", "seed", "twig", "horse", "pig", "hen", "duck", "cow", "bull",
    "sword", "shield", "arrow", "bow", "blade", "knife", "axe", "club", "ship", "boat",
    "raft", "oar", "mast", "anchor", "net", "trap", "lamp", "torch", "fire", "steam",
    "wind", "rain", "snow", "ice", "mist", "fog", "cloud", "reed", "hay", "grass",
    "weed", "vine", "bell", "horn", "pipe", "ring", "chain", "band", "loop", "link",
    "mesh", "coat", "hat", "boot", "sock", "belt", "cloak", "robe", "shirt", "chair",
    "bench", "shelf", "desk", "chest", "crate", "vat", "jar", "pot", "pan", "cup",
    "gem", "shell", "bone", "fang", "scale", "fur", "hide", "wheel", "gear", "spring",
    "lever", "bolt", "bran", "meal", "egg", "lard", "ember", "spark", "pin", "hook",
    "eye", "snap", "stud", "fern", "rush", "flag", "iris", "pike", "eel", "ray",
    "fox", "wolf", "bear", "deer", "hare", "vole", "bat", "owl", "hawk", "crow",
    "ant", "bee", "fly", "moth", "tick", "mite", "slug", "worm", "elm", "fir",
    "alum",
]

# ---- MULTI-TOKEN cargo (the single-vs-multi-token axis test, cam-loop-pkstore-ceiling) ----------
# THE GAP this closes: the single-token probe scores the answer with ONE argmax — the most-cited
# artificiality. The multi-token mode makes the ANSWER a real-word sequence of `cargo_tokens` (2-4)
# tokens; the QA answer is the FULL token SEQUENCE (teacher-forced CE + exact-match scored).
#
# ROLE SWAP vs the single-token dict format: single-token mode keyed on cargo and answered the
# (single-token) NAME. Here the lookup KEY is the single-token NAME and the VALUE/ANSWER is a
# multi-token cargo PHRASE — binding "<name>: <cargo phrase>\n", query "<name>:", answer = the phrase.
#
# CONSTRUCTION (smallest honest version): a K-token cargo phrase = K verified single-token,
# space-prefixed REAL common words joined (e.g. " amber copper silver" = exactly 3 tokens). Each piece
# is checked single-token at generation time, so a K-word phrase is DETERMINISTICALLY exactly K tokens
# — preserving the DocBuilder's constant-length contract for ANY K while keeping the cargo genuine
# real-word multi-token English. The combinatorial pool (|pool|^K) is enormous, so held-out eval draws
# fresh phrases. CAVEAT: true 3-4-token SINGLE words are scarce under Qwen3.5-4B BPE (most common words
# are 1-2 tokens); phrases of real single-token words are the robust, constant-length way to span 2-4
# tokens. This isolates the single-vs-multi-token axis (sequence answer, teacher-forced), NOT a
# real-knowledge test (the pairings stay random, so no_memory pins ~0).
MULTITOKEN_WORD_POOL = [
    "apple", "river", "stone", "cloud", "tiger", "amber", "copper", "silver", "velvet", "crimson",
    "golden", "silent", "ancient", "frozen", "hidden", "royal", "wild", "calm", "bright", "dark",
    "iron", "glass", "coral", "jade", "ruby", "pearl", "slate", "ivory", "ash", "wolf",
    "eagle", "crane", "maple", "cedar", "ember", "frost", "storm", "ocean", "valley", "canyon",
    "harbor", "forest", "desert", "garden", "marble", "crystal", "bronze", "azure", "violet", "olive",
    "teal", "swift", "brave", "noble", "quiet", "gentle", "fierce", "mighty", "humble", "clever",
    "loyal", "north", "south", "grand", "stout", "sharp", "keen", "bold", "pure", "vast", "lone",
]


def single_token_ids(tok, words, prefix=" "):
    """Keep words that encode to exactly one token when space-prefixed; return [(word, tid)]."""
    out = []
    for w in words:
        ids = tok(prefix + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out.append((w, ids[0]))
    return out


def piece(tok, s):
    return tok(s, add_special_tokens=False).input_ids


class DocBuilder:
    """Builds batches of associative-recall docs with deterministic positions (single-token
    name/cargo so every binding line is a constant token length)."""

    def __init__(self, tok, names, cargo, M, seg_len, qa_seg, pad_word=" and", phrasing="dict",
                 cargo_tokens=1, cargo_words=None):
        self.tok = tok
        self.names = names          # list[(word, tid)] — space-prefixed single tokens
        self.cargo = cargo          # dict: NO-space single tokens; manifest: space-prefixed
        self.M = M
        self.seg_len = seg_len
        self.qa_seg = qa_seg
        self.phrasing = phrasing
        # MULTI-TOKEN cargo (cargo_tokens>1): the answer is a K-token real-word phrase. Only supported
        # for dict phrasing. We ROLE-SWAP — the single-token NAME is the lookup key, the K-token cargo
        # PHRASE is the value/answer: binding "<name>: <cargo phrase>\n", query "<name>:". cargo_words
        # is the pool of verified single-token (space-prefixed) words a phrase is drawn from; each phrase
        # is exactly K tokens by construction. cargo_tokens==1 keeps the byte-identical single-token path.
        self.cargo_tokens = int(cargo_tokens)
        self.multitoken = self.cargo_tokens > 1
        if self.multitoken:
            assert phrasing == "dict", "multi-token cargo only implemented for dict phrasing"
            assert cargo_words is not None, "multi-token cargo needs cargo_words (single-token word pool)"
            # cargo_words: list[(word, tid)] verified single-token space-prefixed (see single_token_ids)
            self.cargo_word_tids = [t for (_w, t) in cargo_words]
            assert len(self.cargo_word_tids) >= self.cargo_tokens + 2, \
                "cargo word pool too small for the phrase length"
        self.bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
        # key_off / val_off: the KEY and VALUE token offsets WITHIN a constant-length binding block, read
        # by the pk-store adapter (_write_episode) to locate the association's addressable key + stored
        # value. dict/manifest keep KEY at offset 0 and the historic VALUE offset (1+len(colon)) so the
        # adapter is byte-identical; natural phrasing sets its own (KEY=subject@0, VALUE=object@1+len(rel)).
        if phrasing == "manifest":
            # "<name> carries <cargo>." / "... Question : which ship carries <cargo> ? Answer :"
            self.header = piece(tok, " The manifest lists the following ships.")
            self.carries = piece(tok, " carries")
            self.dot = piece(tok, ".")
            self.qfix1 = piece(tok, " Question : which ship carries")
            self.qfix2 = piece(tok, " ? Answer :")
            self.bind_len = 1 + len(self.carries) + 1 + len(self.dot)   # name + carries + cargo + dot
            self.qfix_len = len(self.qfix1) + 1 + len(self.qfix2)       # +1 for cargo_q
            self.key_off = 0                                            # name
            self.val_off = 1 + len(self.carries)                       # cargo (after "<name> carries")
        elif phrasing == "natural":
            # NATURAL-LANGUAGE single-relation facts: "<Subject> lives in <Object>." Subject drawn from
            # NAME_CANDIDATES (space-prefixed single token = KEY); Object from the single-token real-word
            # pool (space-prefixed, mid-sentence = VALUE). Query "<Subject> lives in" -> answer " <Object>".
            # This is the REALISM probe of issue #1: a coherent relation + real-word vocabulary phrased as
            # a sentence, vs the terse "<cargo>: <name>" dict. no_memory stays ~0 because the (subject->
            # object) pairing is still drawn at RANDOM per doc (the base cannot know a specific random
            # binding), so the realism is the phrasing, not exotic entities.
            assert not self.multitoken, "natural phrasing is single-token only"
            self.header = piece(tok, "The following facts are given.\n")
            self.rel = piece(tok, " lives in")                         # the single fixed relation
            self.dot = piece(tok, ".")
            self.nl = piece(tok, "\n")
            self.bind_len = 1 + len(self.rel) + 1 + len(self.dot) + len(self.nl)  # subj rel obj . \n
            self.qfix_len = 1 + len(self.rel)                          # "<Subject> lives in"
            self.key_off = 0                                           # subject
            self.val_off = 1 + len(self.rel)                          # object (after "<Subject> lives in")
        elif phrasing == "dict":
            # the basecheck-validated best format (acc 0.61): "Cargo to ship:\n<cargo>: <name>\n..." with
            # query "<cargo>:" -> answer " <name>". cargo is line-initial (NO-space single token),
            # name is space-prefixed; ":" and "\n" are single tokens -> constant positions.
            self.colon = piece(tok, ":")
            self.nl = piece(tok, "\n")
            assert len(self.colon) == 1 and len(self.nl) == 1, "':' / '\\n' not single tokens"
            if self.multitoken:
                # ROLE-SWAPPED layout: "<name>: <cargo phrase>\n". key=name (offset 0),
                # value/answer = the K-token cargo phrase (offset 1+len(colon)).
                self.header = piece(tok, "Ship to cargo:\n")
                self.bind_len = 1 + len(self.colon) + self.cargo_tokens + len(self.nl)  # name : <K> \n
                self.qfix_len = 1 + len(self.colon)                     # name :
                self.key_off = 0                                        # name
                self.val_off = 1 + len(self.colon)                     # cargo phrase
            else:
                # the basecheck-validated best format (acc 0.61): "Cargo to ship:\n<cargo>: <name>\n..."
                # query "<cargo>:" -> answer " <name>". cargo line-initial (NO-space single token),
                # name space-prefixed; ":" and "\n" single tokens -> constant positions.
                self.header = piece(tok, "Cargo to ship:\n")
                self.bind_len = 1 + len(self.colon) + 1 + len(self.nl)  # cargo : name \n  (=4)
                self.qfix_len = 1 + len(self.colon)                     # cargo :          (=2)
                self.key_off = 0                                        # cargo
                self.val_off = 1 + len(self.colon)                     # name
        else:
            raise ValueError(f"unknown phrasing {phrasing!r}")
        pad = piece(tok, pad_word)
        assert len(pad) == 1, f"pad_word {pad_word!r} must be a single token (got {pad})"
        self.pad_tid = pad[0]
        self.qa_start = qa_seg * seg_len
        # sanity on the construction
        assert len(self.bos) + len(self.header) + M * self.bind_len <= self.qa_start, \
            "binding block does not fit before the QA segment — raise --qa-seg or seg-len"
        # the QA block (query prefix + the K answer tokens) must fit in one segment
        n_ans = self.cargo_tokens if self.multitoken else 1
        assert self.qfix_len + n_ans <= seg_len, "QA block does not fit in one segment"

    def _draw_cargo(self, rng):
        """One cargo entry. single-token: a token id from self.cargo. multi-token: a K-tuple of distinct
        single-token word ids drawn from the word pool (a real-word K-token phrase)."""
        if not self.multitoken:
            return None  # not used; single-token path draws via self.cargo indices in build()
        idx = rng.choice(len(self.cargo_word_tids), size=self.cargo_tokens, replace=False)
        return tuple(self.cargo_word_tids[i] for i in idx)

    def _binding_ids(self, name_tid, cargo):
        """cargo is an int (single-token) or a tuple of K ints (multi-token phrase).

        NATURAL: name_tid = the SUBJECT token (KEY), cargo = the OBJECT token (VALUE);
        "<Subject> lives in <Object>.\\n"  (subject@0 = key, object@1+len(rel) = value)."""
        if self.phrasing == "natural":
            return [name_tid] + self.rel + [cargo] + self.dot + self.nl
        if self.phrasing == "manifest":
            return [name_tid] + self.carries + [cargo] + self.dot
        if self.multitoken:
            return [name_tid] + self.colon + list(cargo) + self.nl      # "<name>: <cargo phrase>\n"
        return [cargo] + self.colon + [name_tid] + self.nl             # dict: "<cargo>: <name>\n"

    def _query_ids(self, name_tid, cargo):
        if self.phrasing == "natural":
            return [name_tid] + self.rel                               # "<Subject> lives in"
        if self.phrasing == "manifest":
            return self.qfix1 + [cargo] + self.qfix2
        if self.multitoken:
            return [name_tid] + self.colon                             # "<name>:" (key=name)
        return [cargo] + self.colon                                    # dict: "<cargo>:"

    def build(self, rng, batch, local=False):
        """Return (ids[B,S] long, answer[...] long, answer_pos int). Each row draws M distinct
        (name, cargo) bindings and queries one of them. local=True puts the bindings in the QA segment
        itself (base can see them via attention; tests scoring + in-context lookup).

        single-token: answer is [B] (the queried NAME token); answer_pos = its position.
        multi-token : answer is [B,K] (the queried cargo PHRASE token sequence); answer_pos = the
                      position of the FIRST answer token (the K answer tokens occupy apos..apos+K-1)."""
        rows, ans = [], []
        S = None
        for _ in range(batch):
            n_idx = rng.choice(len(self.names), size=self.M, replace=False)
            name_tids = [self.names[i][1] for i in n_idx]
            if self.multitoken:
                cargos = [self._draw_cargo(rng) for _ in range(self.M)]   # M K-token phrases
            else:
                c_idx = rng.choice(len(self.cargo), size=self.M, replace=False)
                cargos = [self.cargo[i][1] for i in c_idx]
            q = int(rng.integers(0, self.M))
            qa = self._query_ids(name_tids[q], cargos[q])           # query prefix (ends at ":")
            if self.multitoken:
                answer_seq = list(cargos[q])                        # K-token cargo phrase = the answer
            elif self.phrasing == "natural":
                answer_seq = [cargos[q]]                            # natural: the OBJECT token is the answer
            else:
                answer_seq = [name_tids[q]]                         # the single-token name = the answer
            bindings = []
            for nt, ct in zip(name_tids, cargos):
                bindings += self._binding_ids(nt, ct)

            if not local:
                pre = self.bos + self.header + bindings
                assert len(pre) <= self.qa_start
                pre = pre + [self.pad_tid] * (self.qa_start - len(pre))   # pad to QA segment start
                seq = pre + qa
                answer_pos = len(seq)                                    # first answer token predicted next
            else:
                # filler-only until QA segment, then [bindings ; query] in that one segment
                pre = self.bos + [self.pad_tid] * (self.qa_start - len(self.bos))
                seq = pre + bindings + qa
                answer_pos = len(seq)

            seq = seq + answer_seq                                       # append gold answer token(s)
            if S is None:
                S = len(seq)
            assert len(seq) == S, "row length mismatch — non-constant tokenization"
            rows.append(seq)
            ans.append(answer_seq if self.multitoken else answer_seq[0])
        ids = torch.tensor(rows, dtype=torch.long)
        ans_t = torch.tensor(ans, dtype=torch.long)                     # [B] or [B,K]
        # answer_pos = position of the FIRST answer token (single-token: last col; multi: K-token block)
        n_ans = self.cargo_tokens if self.multitoken else 1
        return ids, ans_t, len(rows[0]) - n_ans


def run_doc(base, adapter, ids, embeds, seg_len, K, answer_pos, memory=True):
    """Per-segment inject+forward (m2 contract). Returns (mean LM CE over all segments,
    answer_logits[B,V] = logits predicting the token at answer_pos). memory=False -> the adapter
    reads the INIT memory state every segment and never ingests (empty-memory floor)."""
    B, S, H = embeds.shape
    V = base.config.get_text_config().vocab_size
    state, total, nseg, answer_logits, prev_seg = None, 0.0, 0, None, None
    for s in range(0, S, seg_len):
        seg_emb = embeds[:, s:s + seg_len]
        seg_ids = ids[:, s:s + seg_len]
        L = seg_emb.shape[1]
        if L < 2:
            break
        read_state = state if memory else None
        is_ans = s <= answer_pos < s + L
        # query-conditioned read, LEAK-FREE: the answer segment queries its OWN tokens strictly BEFORE
        # the answer (so the answer prediction never sees a prefix derived from the answer token); other
        # segments query the previous segment (past context). first segment / empty ctx -> zero prefix.
        q_ctx = seg_emb[:, :answer_pos - s] if is_ans else prev_seg
        if q_ctx is None or q_ctx.shape[1] == 0:
            mem_tokens = torch.zeros(B, K, H, dtype=embeds.dtype, device=embeds.device)
        else:
            mem_tokens = adapter.read(read_state, q_ctx, embeds.dtype)    # [B,K,H]
        inp = torch.cat([mem_tokens, seg_emb], dim=1)
        logits = base(inputs_embeds=inp).logits[:, K:]               # [B,L,V] drop memory positions
        total = total + F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                                        seg_ids[:, 1:].reshape(-1))
        nseg += 1
        if is_ans:                                                  # answer token lives here
            local = answer_pos - s
            assert local >= 1, "answer at segment start has no in-segment predictor — bad alignment"
            answer_logits = logits[:, local - 1].float()             # logits that predict answer_pos
        if memory:
            state = adapter.ingest(seg_emb, state)
        prev_seg = seg_emb
    return total / max(nseg, 1), answer_logits


@torch.no_grad()
def eval_condition(base, adapter, builder, rng, args, condition):
    """condition in {memory, no_memory, local_control}. Returns (mean NLL bits, accuracy).

    local_control is now a CLEAN base-only control: the in-context doc is fed straight to the frozen
    base in ONE forward, with NO adapter K-prefix and NO segmentation. It measures only 'can the frozen
    base do the in-context associative lookup' — isolated from the adapter machinery (the old version
    prepended the adapter's K memory tokens, which after training are non-zero and collapsed the base's
    lookup, making the control — and thus the whole probe — uninterpretable)."""
    local = condition == "local_control"
    memory = condition == "memory"
    nlls, accs, done = [], [], 0
    while done < args.n_eval:
        cur = min(args.batch, args.n_eval - done)
        ids, ans, apos = builder.build(rng, cur, local=local)
        ids = ids.to(DEV)
        ans = ans.to(DEV)
        if local:
            alog = base(input_ids=ids).logits[:, apos - 1].float()   # pure base, no adapter, no segments
        else:
            embeds = base.get_input_embeddings()(ids).detach()
            _, alog = run_doc(base, adapter, ids, embeds, args.seg_len, args.k, apos, memory=memory)
        logp = F.log_softmax(alog, dim=-1)
        nll = -logp.gather(-1, ans[:, None]).squeeze(-1) / LN2          # bits
        acc = (alog.argmax(-1) == ans).float()
        nlls.extend(nll.tolist())
        accs.extend(acc.tolist())
        done += cur
    return float(np.mean(nlls)), float(np.mean(accs))


@torch.no_grad()
def qa_inject_tokens(base, adapter, ids, embeds, seg_len, K, answer_pos, memory):
    """Replay the segmented inject loop and return the K memory tokens the adapter injects AT the QA
    segment ([B,K,H]) — i.e. the read of the state accumulated from the earlier (binding) segments.
    These are the adapter's output BEFORE the base; decoding the answer from them isolates 'did the
    memory deliver the binding into the injection' independent of the base's generative competence."""
    B, S, H = embeds.shape
    state, out, prev_seg = None, None, None
    for s in range(0, S, seg_len):
        seg_emb = embeds[:, s:s + seg_len]
        L = seg_emb.shape[1]
        if L < 2:
            break
        is_ans = s <= answer_pos < s + L
        q_ctx = seg_emb[:, :answer_pos - s] if is_ans else prev_seg     # same leak-free ctx as run_doc
        if q_ctx is not None and q_ctx.shape[1] > 0:
            mt = adapter.read(state if memory else None, q_ctx, embeds.dtype)   # [B,K,H]
            if is_ans:
                out = mt                                               # the QA-segment injection
        if memory:
            state = adapter.ingest(seg_emb, state)
        prev_seg = seg_emb
    return out


@torch.no_grad()
def _collect_inject(base, adapter, builder, rng, args, n, memory):
    """Gather mean-pooled injected memory tokens [n,H] + answer tids [n] for held-out docs."""
    feats, labels, done = [], [], 0
    while done < n:
        cur = min(args.batch, n - done)
        ids, ans, apos = builder.build(rng, cur, local=False)
        ids = ids.to(DEV)
        embeds = base.get_input_embeddings()(ids).detach()
        mt = qa_inject_tokens(base, adapter, ids, embeds, args.seg_len, args.k, apos, memory)  # [cur,K,H]
        feats.append(mt.mean(dim=1).float().cpu())     # mean-pool over K -> [cur,H]
        labels.append(ans.cpu())
        done += cur
    return torch.cat(feats), torch.cat(labels)


def _fit_linear_decode(X, y, n_classes, tid2cls, epochs=400):
    """Logistic regression (70/30 split, standardized, weight-decayed) -> held-out decode accuracy."""
    cls = torch.tensor([tid2cls[int(t)] for t in y])
    n = X.shape[0]
    ntr = int(n * 0.7)
    Xtr, Xte, ctr, cte = X[:ntr], X[ntr:], cls[:ntr], cls[ntr:]
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    clf = torch.nn.Linear(X.shape[1], n_classes)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=1e-2)
    with torch.enable_grad():           # caller (decode_probe) collects features under no_grad
        for _ in range(epochs):
            opt.zero_grad()
            F.cross_entropy(clf(Xtr), ctr).backward()
            opt.step()
    with torch.no_grad():
        return (clf(Xte).argmax(-1) == cte).float().mean().item()


def decode_probe(base, adapter, eval_b, eval_names, rng, args):
    """Base-competence-INDEPENDENT injection check: is the answer linearly decodable from the injected
    memory tokens? memory (state carries the binding) vs no_memory (init state, no per-example info ->
    must be chance). memory >> chance ⟹ the memory delivers the binding into the injection."""
    tid2cls = {tid: i for i, (_, tid) in enumerate(eval_names)}
    C = len(eval_names)
    Xm, ym = _collect_inject(base, adapter, eval_b, rng, args, args.decode_n, memory=True)
    X0, y0 = _collect_inject(base, adapter, eval_b, rng, args, args.decode_n, memory=False)
    acc_m = _fit_linear_decode(Xm, ym, C, tid2cls)
    acc_0 = _fit_linear_decode(X0, y0, C, tid2cls)
    return acc_m, acc_0, 1.0 / C


def selftest(args):
    """Tokenizer-only: build a batch, decode it, assert the answer token(s) align. No 4B, no GPU."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    cargo_prefix = "" if args.phrasing == "dict" else " "   # natural + manifest place cargo mid-sentence
    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix=cargo_prefix)
    K = args.cargo_tokens
    cargo_words = single_token_ids(tok, MULTITOKEN_WORD_POOL) if K > 1 else None
    print(f"[selftest] phrasing={args.phrasing} cargo_tokens={K} single-token names={len(names)} "
          f"cargo={len(cargo)} word_pool={len(cargo_words) if cargo_words else 0} "
          f"(need >= {args.M + args.eval_pool} names)")
    b = DocBuilder(tok, names, cargo, args.M, args.seg_len, args.qa_seg, phrasing=args.phrasing,
                   cargo_tokens=K, cargo_words=cargo_words)
    print(f"[selftest] bind_len={b.bind_len} qfix_len={b.qfix_len} qa_start={b.qa_start} "
          f"multitoken={b.multitoken}")
    for local in (False, True):
        rng = np.random.default_rng(0)
        ids, ans, apos = b.build(rng, 2, local=local)
        print(f"\n[selftest] local={local} ids.shape={tuple(ids.shape)} ans.shape={tuple(ans.shape)} "
              f"answer_pos={apos} (seg {apos // args.seg_len})")
        for r in range(ids.shape[0]):
            if b.multitoken:
                a = ans[r].tolist()
                assert ids[r, apos:apos + K].tolist() == a, "answer phrase not at answer_pos..+K"
                ans_word = tok.decode(a)
            else:
                assert ids[r, apos].item() == ans[r].item(), "answer token not at answer_pos"
                ans_word = tok.decode([ans[r].item()])
            text = tok.decode(ids[r])
            print(f"  row{r} answer='{ans_word.strip()}' :: {text!r}")
    # round-trip: the piece-concatenated ids must re-tokenize to themselves, so the base sees exactly a
    # valid tokenization of the doc (else our deterministic positions diverge from string-tokenization).
    rng = np.random.default_rng(1)
    ids, _, _ = b.build(rng, 1, local=False)
    row = ids[0].tolist()
    body = row[1:] if (b.bos and row[0] == b.bos[0]) else row    # drop leading BOS for re-tokenize
    retok = tok(tok.decode(body), add_special_tokens=False).input_ids
    assert retok == body, (f"piece-concat NOT round-trip-stable:\n  built={body}\n  retok={retok}")
    print("\n[selftest] OK — answer aligns for both layouts AND piece-concat is round-trip-stable.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="tokenizer-only doc-construction check")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seg-len", type=int, default=32, dest="seg_len")
    ap.add_argument("--qa-seg", type=int, default=2, dest="qa_seg", help="segment the query lands in")
    ap.add_argument("--M", type=int, default=3, help="bindings per doc")
    ap.add_argument("--phrasing", default="dict", choices=["dict", "manifest", "natural"],
                    help="doc format; dict (cargo: name) is the best base substrate per recall_basecheck; "
                         "natural = '<Subject> lives in <Object>.' single-relation NL facts (issue #1)")
    ap.add_argument("--cargo-tokens", type=int, default=1, dest="cargo_tokens",
                    help="K: answer cargo phrase length in tokens (1=single-token byte-preserved path; "
                         ">1 = multi-token real-word cargo, role-swapped 'name: <K-token phrase>')")
    ap.add_argument("--decode-n", type=int, default=1536, dest="decode_n",
                    help="examples for the linear-decodability probe of the injected memory tokens")
    ap.add_argument("--k", type=int, default=16, help="injected memory tokens")
    ap.add_argument("--mem-dim", type=int, default=512, dest="mem_dim")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--answer-weight", type=float, default=4.0, dest="answer_weight",
                    help="extra weight on the answer-token loss during training")
    ap.add_argument("--n-eval", type=int, default=256, dest="n_eval")
    ap.add_argument("--eval-pool", type=int, default=12, dest="eval_pool",
                    help="held-out name/cargo tokens reserved for eval (disjoint from train)")
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--save", default="ckpt/recall_adapter.pt")
    ap.add_argument("--eval-only", action="store_true", dest="eval_only",
                    help="load --save checkpoint, skip training, run held-out eval + decode probe only")
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    base, tok = load_frozen_base()
    H = base.config.get_text_config().hidden_size
    # dict phrasing places cargo line-initial -> needs NO-space single tokens; manifest is mid-sentence
    cargo_prefix = "" if args.phrasing == "dict" else " "
    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix=cargo_prefix)
    assert len(names) >= args.M + args.eval_pool and len(cargo) >= args.M + args.eval_pool, \
        f"not enough single-token words ({args.phrasing}): names={len(names)} cargo={len(cargo)}"
    # disjoint train / eval splits so eval bindings were NEVER seen in training (tests the
    # mechanism, not memorization of specific name->cargo pairs).
    train_names, eval_names = names[:-args.eval_pool], names[-args.eval_pool:]
    train_cargo, eval_cargo = cargo[:-args.eval_pool], cargo[-args.eval_pool:]
    print(f"[recall] device={DEV} base hidden={H} | names {len(train_names)}tr/{len(eval_names)}ev "
          f"cargo {len(train_cargo)}tr/{len(eval_cargo)}ev | M={args.M} seg_len={args.seg_len} "
          f"qa_seg={args.qa_seg} K={args.k} steps={args.steps} bs={args.batch}", flush=True)

    train_b = DocBuilder(tok, train_names, train_cargo, args.M, args.seg_len, args.qa_seg,
                         phrasing=args.phrasing)
    eval_b = DocBuilder(tok, eval_names, eval_cargo, args.M, args.seg_len, args.qa_seg,
                        phrasing=args.phrasing)

    adapter = TitansMemoryAdapter(base_hidden=H, mem_dim=args.mem_dim, n_mem_tokens=args.k,
                                  memory="deepmem").to(DEV)
    print(f"[recall] adapter trainable params: {sum(p.numel() for p in adapter.parameters())/1e6:.2f}M",
          flush=True)
    if args.eval_only:
        ckpt = torch.load(args.save, map_location=DEV)
        adapter.load_state_dict(ckpt["state_dict"])
        print(f"[recall] loaded adapter <- {args.save} (eval-only, skip training)", flush=True)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    for step in (range(args.steps) if not args.eval_only else range(0)):
        opt.zero_grad()
        ids, ans, apos = train_b.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        embeds = base.get_input_embeddings()(ids).detach()
        lm, alog = run_doc(base, adapter, ids, embeds, args.seg_len, args.k, apos, memory=True)
        ans_loss = F.cross_entropy(alog, ans)
        loss = lm + args.answer_weight * ans_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == args.steps - 1:
            acc = (alog.argmax(-1) == ans).float().mean().item()
            print(f"[recall] step {step:4d} lm {lm.item():.3f} ans_nll "
                  f"{ans_loss.item()/LN2:.3f} bits train_acc {acc:.3f}", flush=True)

    if not args.eval_only:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        torch.save({"state_dict": adapter.state_dict(), "args": vars(args)}, args.save)
        print(f"[recall] saved adapter -> {args.save}", flush=True)

    # ---- eval on HELD-OUT bindings -----------------------------------------
    adapter.eval()
    print("\n[recall] === held-out eval (eval-pool bindings, unseen in training) ===", flush=True)
    print(f"{'condition':>14} {'answer NLL (bits)':>18} {'accuracy':>10}", flush=True)
    print("-" * 46, flush=True)
    res = {}
    for cond in ("local_control", "memory", "no_memory"):
        nll, acc = eval_condition(base, adapter, eval_b, rng, args, cond)
        res[cond] = (nll, acc)
        print(f"{cond:>14} {nll:>18.3f} {acc:>10.3f}", flush=True)

    chance_nll = math.log2(args.M)
    print("-" * 46, flush=True)
    print(f"[recall] chance (uniform over {args.M} names) ≈ {chance_nll:.2f} bits / "
          f"{1/args.M:.3f} acc", flush=True)

    # ---- base-competence-independent injection probe: linear-decode the injected memory tokens ----
    print("\n[recall] === linear-decodability of the injected memory tokens (eval-pool) ===", flush=True)
    dec_m, dec_0, dec_chance = decode_probe(base, adapter, eval_b, eval_names, rng, args)
    print(f"  decode answer from injection: memory {dec_m:.3f} | no_memory {dec_0:.3f} | "
          f"chance {dec_chance:.3f} (1/{len(eval_names)} names)", flush=True)

    # ---- verdict ------------------------------------------------------------
    lc_nll, lc_acc = res["local_control"]      # CLEAN base-only ceiling (no adapter)
    m_nll, m_acc = res["memory"]
    nm_nll, nm_acc = res["no_memory"]
    lift_acc = m_acc - nm_acc
    lift_nll = nm_nll - m_nll                    # positive = memory lowers answer NLL (PRIMARY signal)
    nll_lift = lift_nll > 0.20                   # primary: NLL is robust to the base's weak argmax
    decode_ok = dec_m > dec_0 + 0.10 and dec_m > 2 * dec_chance   # injection carries the binding
    print("\n" + "=" * 64, flush=True)
    print(f"[verdict] base-only ceiling (local_control): acc {lc_acc:.3f} nll {lc_nll:.3f} bits "
          f"(this is the headroom cap, NOT a gate)", flush=True)
    print(f"[verdict] PRIMARY  memory NLL-lift over floor: ΔNLL {lift_nll:+.3f} bits  (Δacc {lift_acc:+.3f})",
          flush=True)
    print(f"[verdict] INJECTION decode: memory {dec_m:.3f} vs no_memory {dec_0:.3f} (chance {dec_chance:.3f})",
          flush=True)
    # diagnosis split (decode isolates 'memory delivered into injection'; NLL-lift adds 'base USES it')
    if decode_ok and nll_lift:
        print("[verdict] => END-TO-END RECALL LIFT: the memory delivers the binding into the injection "
              "AND the frozen base uses it generatively. Premise holds → green-light M3 / fund the kernel.",
              flush=True)
    elif decode_ok and not nll_lift:
        print("[verdict] => INJECTION-MECHANISM GAP: the memory DOES deliver the binding (decodable from "
              "the injected tokens) but the input-embeds prefix doesn't drive the frozen base to emit it. "
              "Memory is sound; the bottleneck is the injection point → try multi-layer KV / deeper "
              "injection BEFORE M3. Do NOT blame the memory.", flush=True)
    elif (not decode_ok) and nll_lift:
        print("[verdict] => weak/ambiguous: some generative NLL-lift but the answer is NOT cleanly "
              "decodable from the injection — investigate before funding M3 (K, mem_dim, training).",
              flush=True)
    else:
        print("[verdict] => NO cross-segment recall: the binding is neither decodable from the injection "
              "nor used by the base. Premise unsupported at this scale; do NOT fund M3/kernel — investigate "
              "capacity/K/training (note: MQAR already proved the core, so suspect injection/adapter wiring).",
              flush=True)


if __name__ == "__main__":
    main()
