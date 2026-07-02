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

# flat package: sibling imports resolve relatively when imported as cam.X and fall back to a
# path-hacked absolute import when run as a file.
try:
    from .m2_adapter import MODEL, DEV, TitansMemoryAdapter, load_frozen_base
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
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

# ---- VARIED-RELATION natural phrasing (issue #1, heterogeneous facts) --------------------------
# The `natural` phrasing uses ONE fixed relation (" lives in") for every fact — so the document is M
# repetitions of the same template. `varied` extends it: each fact independently draws from a SMALL
# SET of relation templates, so one document mixes fact STRUCTURES (subject lives-in / works-as /
# was-born-in / owns-a / studies an object). This tests whether the memory handles diverse fact
# shapes in a single document, not just one repeated template.
#
# CONTRACT each relation string must satisfy so the deterministic-position addressing holds:
#   - space-prefixed (the SUBJECT that precedes it is a space-prefixed single token = KEY@offset 0);
#   - the OBJECT immediately follows the relation (space-prefixed single token = VALUE@offset
#     1+len(rel)), then "." and "\n". A binding block is [subj, *rel, obj, ".", "\n"], whose length
#     is 1+len(rel)+1+len(dot)+len(nl) and VARIES per relation (hence per-binding positions, below).
# Relations are assigned DETERMINISTICALLY per binding slot (slot m -> relations[m % R]) so the
# per-binding token positions are batch-uniform (the store reads keys/values by a single [B,pos]
# gather) AND identical across tokenizers (cross-base transfer draws the same fact structure).
VARIED_RELATIONS = [
    " lives in",
    " works as",
    " was born in",
    " owns a",
    " studies",
]

# ---- COUNTERFACTUAL knowledge-editing facts (issue #1, real-knowledge-editing) -----------------
# THE TEST this enables: real (country -> capital) facts the FROZEN BASE ALREADY KNOWS parametrically,
# but the MEMORY holds a COUNTERFACTUAL (deranged) capital. Does the memory make the frozen base emit
# the WRONG (counterfactual) capital, OVERRIDING its own prior? Unlike every other phrasing (where the
# name<->cargo pairing is RANDOM so no_memory pins ~0 and the memory only has to teach a fact the base
# cannot know), here the base HAS a prior — so the probe measures genuine knowledge EDITING (override),
# not knowledge INSERTION.
#
# A prior attempt was INVALID because the base did NOT actually hold the priors it was tested on
# (no_mem prior-acc 0.107). The fix (recall_mag runs it when --phrasing counterfactual): PROBE the
# frozen base FIRST over the candidate facts (no memory, tap off), KEEP ONLY the facts it answers
# correctly, and bind the counterfactual values on that FILTERED set. Then no_mem prior-acc is high
# BY CONSTRUCTION (we kept only known facts) and the counterfactual-acc measures true override.
#
# CONTRACT (so the deterministic-position, single-token-KEY store machinery is reused byte-for-byte):
#   - EVERY country encodes to a SINGLE space-prefixed token (the addressable KEY at qa_start), and
#   - EVERY capital encodes to a SINGLE space-prefixed token (the VALUE / one-token answer),
#     both verified at generation time against the live Qwen3.5-4B tokenizer (single_token_ids). Facts
#     whose country OR capital is multi-token are dropped at startup (multi-token country names are
#     "handled" by exclusion — they cannot be a single-token KEY). Capitals are all DISTINCT so a
#     derangement exists. The doc format is natural-language "The capital of <Country> is <Capital>."
#     with "The capital of" folded into the HEADER (the query FORMAT prefix), so the query region is
#     "<Country> is" (KEY=country@qa_start, addr-sup intact) and the base context reconstructs
#     "...The capital of <Country> is" -> predicts the capital. seg_len ~48 fits the doc-format that
#     elicits the base's parametric recall (the value that worked in the prior agent's run).
COUNTERFACTUAL_FACTS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Austria", "Vienna"), ("Norway", "Oslo"), ("Egypt", "Cairo"), ("Japan", "Tokyo"),
    ("Ireland", "Dublin"), ("Portugal", "Lisbon"), ("Poland", "Warsaw"), ("Greece", "Athens"),
    ("Finland", "Helsinki"), ("Switzerland", "Bern"), ("Sweden", "Stockholm"),
    ("Denmark", "Copenhagen"), ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"),
    ("Hungary", "Budapest"), ("Turkey", "Ankara"), ("Russia", "Moscow"), ("China", "Beijing"),
    ("India", "Delhi"), ("Canada", "Ottawa"), ("Peru", "Lima"), ("Chile", "Santiago"),
    ("Cuba", "Havana"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Israel", "Jerusalem"),
    ("Kenya", "Nairobi"), ("Thailand", "Bangkok"), ("Bulgaria", "Sofia"), ("Tunisia", "Tunis"),
    ("Venezuela", "Caracas"), ("Afghanistan", "Kabul"), ("Pakistan", "Islamabad"),
    ("Indonesia", "Jakarta"), ("Philippines", "Manila"), ("Australia", "Canberra"),
]

# The natural-language fact format, factored so the store reuses the single-token-KEY natural machinery.
# HEADER carries the query FORMAT prefix ("The capital of") so the query region starts AT the country
# (the single-token KEY at qa_start, so addr-sup q_tok=ids[:,qa_start] is the country). REL=" is".
COUNTERFACTUAL_HEADER = "The following facts are given.\nThe capital of"
COUNTERFACTUAL_REL = " is"


def counterfactual_single_token(tok):
    """Filter COUNTERFACTUAL_FACTS to those whose country AND capital are each a single space-prefixed
    token under `tok`. Returns [(country, capital, country_tid, capital_tid)] — the candidate table the
    probe/filter narrows further. Multi-token country/capital facts are excluded (a multi-token country
    cannot be the single-token addressable KEY)."""
    out = []
    for country, capital in COUNTERFACTUAL_FACTS:
        c_ids = tok(" " + country, add_special_tokens=False).input_ids
        k_ids = tok(" " + capital, add_special_tokens=False).input_ids
        if len(c_ids) == 1 and len(k_ids) == 1:
            out.append((country, capital, c_ids[0], k_ids[0]))
    return out


def derange_capitals(rng, n):
    """A derangement of range(n): a permutation with NO fixed point (perm[i] != i for all i). Used to
    assign each country a COUNTERFACTUAL capital that is NOT its true one. Sattolo-style single-cycle
    shuffle guarantees no fixed point in one pass (requires n >= 2)."""
    assert n >= 2, "derangement needs at least 2 facts"
    perm = list(range(n))
    # Sattolo's algorithm: produces a uniformly-random single cycle -> guaranteed derangement.
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))     # 0 <= j < i (strictly below i -> no fixed point)
        perm[i], perm[j] = perm[j], perm[i]
    assert all(perm[i] != i for i in range(n)), "derangement produced a fixed point"
    return perm


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
                 cargo_tokens=1, cargo_words=None, facts=None):
        self.tok = tok
        self.names = names          # list[(word, tid)] — space-prefixed single tokens
        self.cargo = cargo          # dict: NO-space single tokens; manifest: space-prefixed
        self.M = M
        self.seg_len = seg_len
        self.qa_seg = qa_seg
        self.phrasing = phrasing
        # COUNTERFACTUAL: a FIXED fact table (country -> capital) the base already knows. `facts` is a
        # list of (country, capital, country_tid, capital_tid_TRUE); the counterfactual (memory) capital
        # per fact is carried in self.cf_tid (set by set_counterfactual, filled by the probe/filter step).
        # build() places the COUNTERFACTUAL capital as the binding VALUE (what the memory teaches) and
        # returns it as the answer; build_cf() additionally returns the TRUE (prior) capital answer.
        self.facts = facts
        self.cf_tid = None          # per-fact counterfactual capital tid (parallel to self.facts)
        # MULTI-TOKEN cargo (cargo_tokens>1): the answer is a K-token real-word phrase. Only supported
        # for dict phrasing. We ROLE-SWAP — the single-token NAME is the lookup key, the K-token cargo
        # PHRASE is the value/answer: binding "<name>: <cargo phrase>\n", query "<name>:". cargo_words
        # is the pool of verified single-token (space-prefixed) words a phrase is drawn from; each phrase
        # is exactly K tokens by construction. cargo_tokens==1 keeps the byte-identical single-token path.
        self.cargo_tokens = int(cargo_tokens)
        self.multitoken = self.cargo_tokens > 1
        if self.multitoken:
            assert phrasing in ("dict", "natural"), \
                "multi-token cargo implemented for dict + natural phrasing (not manifest/varied)"
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
            # NAME_CANDIDATES (space-prefixed single token = KEY); Object from the real-word pool
            # (space-prefixed, mid-sentence = VALUE). Query "<Subject> lives in" -> answer " <Object>".
            # This is the REALISM probe of issue #1: a coherent relation + real-word vocabulary phrased as
            # a sentence, vs the terse "<cargo>: <name>" dict. no_memory stays ~0 because the (subject->
            # object) pairing is still drawn at RANDOM per doc (the base cannot know a specific random
            # binding), so the realism is the phrasing, not exotic entities.
            #
            # MULTI-TOKEN natural (cargo_tokens=K>1): the OBJECT is a K-token real-word phrase drawn from
            # the same verified single-token, space-prefixed word pool the disjoint DICT experiments used
            # ("<Subject> lives in <w0 w1>."). The subject stays the single-token KEY; the K-token object
            # phrase is the VALUE (answer). Because each object word is verified single-token and
            # space-prefixed at generation time, a K-word phrase is DETERMINISTICALLY exactly K tokens, so
            # the binding block "<Subject> lives in <K-token object>.\n" is constant length (bind_len holds)
            # and the disjoint per-position store reads object token t at val_off+t — the SAME generic
            # (key_off, val_off, bind_len, cargo_tokens) contract the DICT multi-token path uses. This
            # combines natural phrasing (issue #1 realism) with multi-token answers (real facts are often
            # multi-token: "lives in New York"). K==1 keeps the byte-identical single-token natural path.
            self.header = piece(tok, "The following facts are given.\n")
            self.rel = piece(tok, " lives in")                         # the single fixed relation
            self.dot = piece(tok, ".")
            self.nl = piece(tok, "\n")
            if self.multitoken:
                # OBJECT is a K-token phrase drawn from the single-token word pool (like DICT multi-token);
                # self.cargo_word_tids was already validated+set in the common multitoken block above.
                # subj rel <K object tokens> . \n
                self.bind_len = 1 + len(self.rel) + self.cargo_tokens + len(self.dot) + len(self.nl)
            else:
                self.bind_len = 1 + len(self.rel) + 1 + len(self.dot) + len(self.nl)  # subj rel obj . \n
            self.qfix_len = 1 + len(self.rel)                          # "<Subject> lives in"
            self.key_off = 0                                           # subject
            self.val_off = 1 + len(self.rel)                          # object (after "<Subject> lives in")
        elif phrasing == "varied":
            # VARIED-RELATION natural facts: each binding slot m draws a relation from VARIED_RELATIONS
            # DETERMINISTICALLY (slot m -> relations[m % R]) so per-binding positions are batch-uniform
            # and tokenizer-agnostic. A binding is "<Subject><rel> <Object>.\n" with the subject the
            # KEY (offset 0 within its block) and the object the VALUE (offset 1+len(rel_m)). Because
            # rel length VARIES, bind blocks have DIFFERENT lengths -> we precompute per-binding token
            # lengths and cumulative bases; the constant `bind_len` no longer applies (the pk-store
            # adapter reads per-binding key/val positions via builder.binding_positions()).
            assert not self.multitoken, "varied phrasing is single-token only"
            self.header = piece(tok, "The following facts are given.\n")
            self.dot = piece(tok, ".")
            self.nl = piece(tok, "\n")
            self.rels = [piece(tok, r) for r in VARIED_RELATIONS]     # per-relation token pieces
            self.R = len(self.rels)
            # per binding SLOT m: which relation, its token piece, its block length, its val offset
            self.slot_rel = [m % self.R for m in range(M)]
            self.bind_lens = [1 + len(self.rels[r]) + 1 + len(self.dot) + len(self.nl)
                              for r in self.slot_rel]                  # subj rel obj . \n  (per slot)
            self.bind_vals = [1 + len(self.rels[r]) for r in self.slot_rel]  # object offset per slot
            # cumulative base offset of slot m WITHIN the binding block (after bos+header)
            self.bind_bases = [sum(self.bind_lens[:m]) for m in range(M)]
            self.bind_total = sum(self.bind_lens)                     # total binding-block token length
            # key_off/val_off kept for API compat (fixed-length fallbacks); the varied path uses the
            # PER-BINDING positions from binding_positions(). key is always subject@0 within a block.
            self.key_off = 0
            self.val_off = None                                        # varies per slot -> not scalar
            self.bind_len = None                                      # NOT constant for varied
            # qfix_len (query "<Subject><rel>") and the answer position depend on the QUERIED slot,
            # which is drawn per build() and shared across the batch (self._q for this build).
            self.qfix_len = None                                     # set per-build in build()
        elif phrasing == "counterfactual":
            # COUNTERFACTUAL knowledge-editing facts: "<Country> is <Capital>." drawn from a FIXED fact
            # table (self.facts). Structurally identical to `natural`, but the query FORMAT prefix
            # "The capital of" is folded into the HEADER so the query region begins AT the country (the
            # single-token KEY at qa_start -> addr-sup q_tok=ids[:,qa_start] is the country) while the
            # base context reconstructs "...The capital of <Country> is" -> predicts the capital. The
            # binding VALUE is the COUNTERFACTUAL capital (set via set_counterfactual); the true (prior)
            # capital is returned by build_cf() for the prior-recall metrics. KEY=country@0, REL=" is",
            # VALUE=capital@1+len(rel). All countries/capitals are single-token (constant bind_len).
            assert not self.multitoken, "counterfactual phrasing is single-token only"
            assert facts is not None and len(facts) >= M, \
                "counterfactual phrasing needs a fact table with >= M facts (set via facts=)"
            self.header = piece(tok, COUNTERFACTUAL_HEADER)            # "...The capital of"
            self.rel = piece(tok, COUNTERFACTUAL_REL)                  # " is"
            self.dot = piece(tok, ".")
            self.nl = piece(tok, "\n")
            self.bind_len = 1 + len(self.rel) + 1 + len(self.dot) + len(self.nl)  # ctry is cap . \n
            self.qfix_len = 1 + len(self.rel)                         # "<Country> is"
            self.key_off = 0                                          # country
            self.val_off = 1 + len(self.rel)                         # capital (after "<Country> is")
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
        # sanity on the construction. varied has PER-BINDING lengths (bind_len is None) -> use the
        # precomputed total; the query length is the LONGEST possible queried relation (worst case).
        if self.phrasing == "varied":
            bind_block = len(self.bos) + len(self.header) + self.bind_total
            worst_qfix = 1 + max(len(r) for r in self.rels)          # "<Subject><rel>" (longest rel)
        else:
            bind_block = len(self.bos) + len(self.header) + M * self.bind_len
            worst_qfix = self.qfix_len
        assert bind_block <= self.qa_start, \
            "binding block does not fit before the QA segment — raise --qa-seg or seg-len"
        # the QA block (query prefix + the K answer tokens) must fit in one segment
        n_ans = self.cargo_tokens if self.multitoken else 1
        assert worst_qfix + n_ans <= seg_len, "QA block does not fit in one segment"

    def _draw_cargo(self, rng):
        """One cargo entry. single-token: a token id from self.cargo. multi-token: a K-tuple of distinct
        single-token word ids drawn from the word pool (a real-word K-token phrase)."""
        if not self.multitoken:
            return None  # not used; single-token path draws via self.cargo indices in build()
        idx = rng.choice(len(self.cargo_word_tids), size=self.cargo_tokens, replace=False)
        return tuple(self.cargo_word_tids[i] for i in idx)

    def binding_positions(self, hstart):
        """VARIED: absolute (key_pos, val_pos) for each of the M bindings given the binding block's
        start offset hstart (=len(bos)+len(header)). key = subject (block offset 0), value = object
        (block offset 1+len(rel_m)). The pk-store adapter uses these instead of the constant-bind_len
        arithmetic (which does not hold when relation lengths vary)."""
        assert self.phrasing == "varied", "binding_positions is varied-only"
        keys, vals = [], []
        for m in range(self.M):
            base = hstart + self.bind_bases[m]
            keys.append(base + 0)                        # subject (KEY)
            vals.append(base + self.bind_vals[m])        # object (VALUE), after "<Subject><rel_m>"
        return keys, vals

    def _binding_ids(self, name_tid, cargo, slot=None):
        """cargo is an int (single-token) or a tuple of K ints (multi-token phrase).

        NATURAL: name_tid = the SUBJECT token (KEY), cargo = the OBJECT token (VALUE);
        "<Subject> lives in <Object>.\\n"  (subject@0 = key, object@1+len(rel) = value).
        VARIED: same, but the relation is self.rels[self.slot_rel[slot]] (per-slot)."""
        if self.phrasing == "varied":
            rel = self.rels[self.slot_rel[slot]]                       # per-slot relation piece
            return [name_tid] + rel + [cargo] + self.dot + self.nl     # "<Subj><rel> <Obj>.\n"
        if self.phrasing == "natural":
            # single-token: cargo is an int object; multi-token: cargo is a K-tuple object phrase.
            obj = list(cargo) if self.multitoken else [cargo]
            return [name_tid] + self.rel + obj + self.dot + self.nl    # "<Subj> lives in <Obj(s)>.\n"
        if self.phrasing == "counterfactual":
            # name_tid = country (KEY), cargo = capital tid (VALUE, the counterfactual capital in memory)
            return [name_tid] + self.rel + [cargo] + self.dot + self.nl  # "<Country> is <Capital>.\n"
        if self.phrasing == "manifest":
            return [name_tid] + self.carries + [cargo] + self.dot
        if self.multitoken:
            return [name_tid] + self.colon + list(cargo) + self.nl      # "<name>: <cargo phrase>\n"
        return [cargo] + self.colon + [name_tid] + self.nl             # dict: "<cargo>: <name>\n"

    def _query_ids(self, name_tid, cargo, slot=None):
        if self.phrasing == "varied":
            rel = self.rels[self.slot_rel[slot]]                       # queried slot's relation
            return [name_tid] + rel                                    # "<Subject><rel>"
        if self.phrasing == "natural":
            return [name_tid] + self.rel                               # "<Subject> lives in"
        if self.phrasing == "counterfactual":
            return [name_tid] + self.rel                               # "<Country> is"
        if self.phrasing == "manifest":
            return self.qfix1 + [cargo] + self.qfix2
        if self.multitoken:
            return [name_tid] + self.colon                             # "<name>:" (key=name)
        return [cargo] + self.colon                                    # dict: "<cargo>:"

    def set_counterfactual(self, cf_tid):
        """Install the per-fact COUNTERFACTUAL capital tids (parallel to self.facts) that build() places
        as the binding VALUE / answer. cf_tid[i] is the (deranged) capital token for fact i; the TRUE
        capital (self.facts[i][3]) is used only by build_cf() for the prior-recall answer. Called by
        recall_mag AFTER the probe/filter narrows self.facts to the base-known set + deranges."""
        assert self.phrasing == "counterfactual", "set_counterfactual is counterfactual-only"
        assert len(cf_tid) == len(self.facts), "cf_tid must be parallel to self.facts"
        self.cf_tid = list(cf_tid)

    def _build_counterfactual(self, rng, batch, local=False):
        """COUNTERFACTUAL doc builder. Each row draws M distinct facts from self.facts and queries one.
        The binding VALUE = the COUNTERFACTUAL capital (self.cf_tid); the answer returned by build() is
        that counterfactual capital. Also returns the parallel PRIOR (true) capital answer so build_cf()
        can score both. Returns (ids[B,S], ans_cf[B], ans_prior[B], answer_pos)."""
        assert self.cf_tid is not None, "call set_counterfactual(cf_tid) before building counterfactual docs"
        rows, ans_cf, ans_prior = [], [], []
        S = None
        for _ in range(batch):
            f_idx = rng.choice(len(self.facts), size=self.M, replace=False)
            country_tids = [self.facts[i][2] for i in f_idx]        # KEY (country) tids
            cf_caps = [self.cf_tid[i] for i in f_idx]               # counterfactual capital -> memory VALUE
            true_caps = [self.facts[i][3] for i in f_idx]           # true capital -> prior answer
            q = int(rng.integers(0, self.M))
            qa = self._query_ids(country_tids[q], cf_caps[q], slot=q)   # "<Country> is"
            bindings = []
            for nt, ct in zip(country_tids, cf_caps):
                bindings += self._binding_ids(nt, ct)              # "<Country> is <CF-Capital>.\n"
            if not local:
                pre = self.bos + self.header + bindings
                assert len(pre) <= self.qa_start
                pre = pre + [self.pad_tid] * (self.qa_start - len(pre))
                seq = pre + qa
                answer_pos = len(seq)
            else:
                pre = self.bos + [self.pad_tid] * (self.qa_start - len(self.bos))
                seq = pre + bindings + qa
                answer_pos = len(seq)
            seq = seq + [cf_caps[q]]                                # gold = counterfactual capital
            if S is None:
                S = len(seq)
            assert len(seq) == S, "row length mismatch — non-constant tokenization"
            rows.append(seq)
            ans_cf.append(cf_caps[q])
            ans_prior.append(true_caps[q])
        ids = torch.tensor(rows, dtype=torch.long)
        return (ids, torch.tensor(ans_cf, dtype=torch.long),
                torch.tensor(ans_prior, dtype=torch.long), len(rows[0]) - 1)

    def build_cf(self, rng, batch, local=False):
        """Counterfactual build exposing BOTH answers: (ids, ans_cf[B], ans_prior[B], answer_pos).
        ans_cf = the counterfactual capital the memory teaches; ans_prior = the base's TRUE prior."""
        assert self.phrasing == "counterfactual", "build_cf is counterfactual-only"
        return self._build_counterfactual(rng, batch, local=local)

    def build(self, rng, batch, local=False):
        """Return (ids[B,S] long, answer[...] long, answer_pos int). Each row draws M distinct
        (name, cargo) bindings and queries one of them. local=True puts the bindings in the QA segment
        itself (base can see them via attention; tests scoring + in-context lookup).

        counterfactual: draws from the fixed fact table; answer = the COUNTERFACTUAL capital (the value
                        the memory teaches). build_cf() additionally returns the true prior capital.
        single-token: answer is [B] (the queried NAME token); answer_pos = its position.
        multi-token : answer is [B,K] (the queried cargo PHRASE token sequence); answer_pos = the
                      position of the FIRST answer token (the K answer tokens occupy apos..apos+K-1)."""
        if self.phrasing == "counterfactual":
            ids, ans_cf, _ans_prior, apos = self._build_counterfactual(rng, batch, local=local)
            return ids, ans_cf, apos
        rows, ans = [], []
        S = None
        # VARIED: the queried slot must be BATCH-UNIFORM (its relation sets the query length, so all
        # rows must share it for a rectangular batch). Drawn once here; the queried FACT still varies
        # per row (different subject/object), and across build() calls the slot varies too.
        q_fixed = int(rng.integers(0, self.M)) if self.phrasing == "varied" else None
        for _ in range(batch):
            n_idx = rng.choice(len(self.names), size=self.M, replace=False)
            name_tids = [self.names[i][1] for i in n_idx]
            if self.multitoken:
                cargos = [self._draw_cargo(rng) for _ in range(self.M)]   # M K-token phrases
            else:
                c_idx = rng.choice(len(self.cargo), size=self.M, replace=False)
                cargos = [self.cargo[i][1] for i in c_idx]
            q = q_fixed if q_fixed is not None else int(rng.integers(0, self.M))
            qa = self._query_ids(name_tids[q], cargos[q], slot=q)   # query prefix (ends at ":"/rel)
            if self.multitoken:
                answer_seq = list(cargos[q])                        # K-token cargo phrase = the answer
            elif self.phrasing in ("natural", "varied"):
                answer_seq = [cargos[q]]                            # natural/varied: OBJECT token = answer
            else:
                answer_seq = [name_tids[q]]                         # the single-token name = the answer
            bindings = []
            for m, (nt, ct) in enumerate(zip(name_tids, cargos)):
                bindings += self._binding_ids(nt, ct, slot=m)

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


def selftest_counterfactual(args, tok):
    """Counterfactual doc-construction selftest: fact-table single-token verification, derangement (no
    fixed point), doc building, per-binding KEY/VALUE position check, and build_cf's dual answer set.
    Tokenizer only — no 4B, no GPU."""
    facts = counterfactual_single_token(tok)
    print(f"[selftest] phrasing=counterfactual candidate facts (single-token country+capital): "
          f"{len(facts)} / {len(COUNTERFACTUAL_FACTS)} in table")
    assert len(facts) >= args.M, f"need >= M={args.M} single-token facts, have {len(facts)}"
    for country, capital, ctid, ktid in facts[:5]:
        assert len(tok(" " + country, add_special_tokens=False).input_ids) == 1
        assert len(tok(" " + capital, add_special_tokens=False).input_ids) == 1
    # derangement: no country keeps its true capital
    rng = np.random.default_rng(args.seed)
    perm = derange_capitals(rng, len(facts))
    cf_tid = [facts[perm[i]][3] for i in range(len(facts))]     # counterfactual capital tid per fact
    for i in range(len(facts)):
        assert cf_tid[i] != facts[i][3], f"fact {i} kept its true capital — not deranged"
    print(f"[selftest] derangement OK: {len(facts)} facts, no fact keeps its true capital "
          f"(e.g. {facts[0][0]}->{tok.decode([cf_tid[0]]).strip()} instead of {facts[0][1]})")
    b = DocBuilder(tok, None, None, args.M, args.seg_len, args.qa_seg, phrasing="counterfactual",
                   facts=facts)
    b.set_counterfactual(cf_tid)
    print(f"[selftest] bind_len={b.bind_len} qfix_len={b.qfix_len} qa_start={b.qa_start} "
          f"header_len={len(b.header)} rel={b.rel}")
    fact_ctids = {f[2] for f in facts}
    ctid2cf = {facts[i][2]: cf_tid[i] for i in range(len(facts))}     # country tid -> counterfactual cap
    ctid2true = {facts[i][2]: facts[i][3] for i in range(len(facts))}  # country tid -> TRUE capital
    for local in (False, True):
        r = np.random.default_rng(0)
        ids, ans_cf, ans_prior, apos = b.build_cf(r, 2, local=local)
        print(f"\n[selftest] local={local} ids.shape={tuple(ids.shape)} apos={apos} "
              f"(seg {apos // args.seg_len})")
        # per-binding KEY(country)/VALUE(counterfactual capital) positions (non-local addressing layout)
        if not local:
            hstart = len(b.bos) + len(b.header)
            for row in range(ids.shape[0]):
                for m in range(b.M):
                    base = hstart + m * b.bind_len
                    kt = ids[row, base + b.key_off].item()
                    vt = ids[row, base + b.val_off].item()
                    assert kt in fact_ctids, f"row{row} slot{m}: KEY@{base+b.key_off} not a country token"
                    rel_ids = ids[row, base + 1:base + 1 + len(b.rel)].tolist()
                    assert rel_ids == b.rel, f"row{row} slot{m}: rel {rel_ids} != {b.rel}"
                    # VALUE must be THIS country's counterfactual capital, and NOT its true one
                    assert vt == ctid2cf[kt], "VALUE is not the counterfactual capital for this country"
                    assert vt != ctid2true[kt], "VALUE equals the TRUE capital — derangement leaked"
            print(f"[selftest] counterfactual KEY(country)/VALUE(cf-capital) positions VERIFIED "
                  f"for all M={b.M} slots (VALUE = deranged capital != true capital)")
        for row in range(ids.shape[0]):
            assert ids[row, apos].item() == ans_cf[row].item(), "answer token not at answer_pos"
            # queried country: the token at qa_start (the KEY position of the query region)
            q_country = ids[row, b.qa_start].item()
            assert q_country in fact_ctids, "query key at qa_start is not a country token"
            print(f"  row{row} query='{tok.decode([q_country]).strip()}' "
                  f"cf_ans='{tok.decode([ans_cf[row].item()]).strip()}' "
                  f"prior_ans='{tok.decode([ans_prior[row].item()]).strip()}' :: {tok.decode(ids[row])!r}")
    # round-trip stability
    r = np.random.default_rng(1)
    ids, _, _, _ = b.build_cf(r, 1, local=False)
    row = ids[0].tolist()
    body = row[1:] if (b.bos and row[0] == b.bos[0]) else row
    retok = tok(tok.decode(body), add_special_tokens=False).input_ids
    assert retok == body, (f"piece-concat NOT round-trip-stable:\n  built={body}\n  retok={retok}")
    print("\n[selftest] OK — counterfactual answer aligns (cf + prior) AND piece-concat is round-trip-stable.")


def selftest(args):
    """Tokenizer-only: build a batch, decode it, assert the answer token(s) align. No 4B, no GPU."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if args.phrasing == "counterfactual":
        selftest_counterfactual(args, tok)
        return
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
    if args.phrasing == "varied":
        print(f"[selftest] varied: R={b.R} slot_rel={b.slot_rel} bind_lens={b.bind_lens} "
              f"bind_vals={b.bind_vals} bind_bases={b.bind_bases} bind_total={b.bind_total}")
    for local in (False, True):
        rng = np.random.default_rng(0)
        ids, ans, apos = b.build(rng, 2, local=local)
        print(f"\n[selftest] local={local} ids.shape={tuple(ids.shape)} ans.shape={tuple(ans.shape)} "
              f"answer_pos={apos} (seg {apos // args.seg_len})")
        # VARIED: verify EACH binding's KEY (subject) and VALUE (object) land at the per-binding
        # positions the pk-store adapter will read — for the non-local layout only (local shifts the
        # bindings into the QA segment, exercised for round-trip but not the store's hstart addressing).
        if args.phrasing == "varied" and not local:
            hstart = len(b.bos) + len(b.header)
            key_pos, val_pos = b.binding_positions(hstart)
            for r in range(ids.shape[0]):
                for m in range(b.M):
                    rel = VARIED_RELATIONS[b.slot_rel[m]]
                    kt = ids[r, key_pos[m]].item(); vt = ids[r, val_pos[m]].item()
                    # KEY must be a subject (space-prefixed NAME); VALUE an object (space-prefixed word)
                    kw = tok.decode([kt]); vw = tok.decode([vt])
                    assert kt in {t for _, t in names}, \
                        f"row{r} slot{m} rel'{rel}': KEY@{key_pos[m]} '{kw}' not a subject token"
                    assert vt in {t for _, t in cargo}, \
                        f"row{r} slot{m} rel'{rel}': VALUE@{val_pos[m]} '{vw}' not an object token"
                    # and the token right after the KEY must begin the relation piece
                    rel_ids = b.rels[b.slot_rel[m]]
                    got = ids[r, key_pos[m] + 1:key_pos[m] + 1 + len(rel_ids)].tolist()
                    assert got == rel_ids, \
                        f"row{r} slot{m}: relation ids {got} != '{rel}' {rel_ids} after KEY"
            print(f"[selftest] varied KEY/VALUE per-binding positions VERIFIED for all M={b.M} slots "
                  f"(subject@key_pos, object@val_pos, relation piece between them)")
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
    ap.add_argument("--phrasing", default="dict",
                    choices=["dict", "manifest", "natural", "varied", "counterfactual"],
                    help="doc format; dict (cargo: name) is the best base substrate per recall_basecheck; "
                         "natural = '<Subject> lives in <Object>.' single-relation NL facts (issue #1); "
                         "varied = per-fact relation drawn from a small template set (heterogeneous facts); "
                         "counterfactual = real (country->capital) facts the base KNOWS with DERANGED "
                         "capitals in memory — knowledge-editing override probe (recall_mag runs the "
                         "probe/filter; the deepmem selftest checks the fact table + derangement + doc)")
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
