# Acknowledgments

This project did not come from nowhere. It stands on a small number of specific ideas from other
people's work, and we want to name them precisely rather than gesture at "prior art."

## RecursiveMAS — the idea that started this path

The originating spark was a translator concept borrowed from **RecursiveMAS (Recursive Multi-Agent
Systems)**. Their **RecursiveLink** is a lightweight residual module that carries one model's
last-layer hidden states into another model's space so that latent "thoughts" can be exchanged across
models. We asked a naive question — *what if that link, instead of passing thoughts between agents,
carried a frozen external **memory** into an arbitrary frozen base model?* — and that question is the
whole reason this repository exists. Our base-agnostic translator (`TranslatedInjector`: a tiny affine
map `A: d_base2 → d_base1` and `B: d_base1 → d_base2` with a zero-init gate) is a direct descendant of
the RecursiveLink idea, repurposed for memory delivery rather than agent-to-agent communication.

> Xiyuan Yang, Jiaru Zou, Rui Pan, Ruizhong Qiu, Pan Lu, Shizhe Diao, Jindong Jiang, Hanghang Tong,
> Tong Zhang, Markus J. Buehler, Jingrui He, James Zou.
> **"Recursive Multi-Agent Systems."** arXiv:2604.25917 (2026). MIT license.
> Paper: https://arxiv.org/abs/2604.25917 · Code: https://github.com/RecursiveMAS/RecursiveMAS

```bibtex
@misc{recursivemas,
  title={Recursive Multi-Agent Systems},
  author={Xiyuan Yang and Jiaru Zou and Rui Pan and Ruizhong Qiu and Pan Lu and Shizhe Diao and
          Jindong Jiang and Hanghang Tong and Tong Zhang and Markus J. Buehler and Jingrui He and
          James Zou},
  year={2026},
  eprint={2604.25917},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2604.25917},
}
```

## Prior art this builds on

- **Titans** — the test-time / long-term memory architecture and the Memory-as-Gate (MAG) and
  Memory-as-Context (MAC) variants that this project's delivery mechanism is built around.
  *Behrouz et al., "Titans: Learning to Memorize at Test Time."*
- **Product-key memory** — the sparse, addressable large-memory layer our store of record is based on.
  *Lample et al., "Large Memory Layers with Product Keys," NeurIPS 2019.*
- **Relative representations** — the centered-cosine relative-representation probe used early on to test
  cross-model geometry. *Moschella et al., "Relative representations enable zero-shot latent space
  communication," ICLR 2023.*
- **lucidrains/titans-pytorch** — used as a reference implementation while building the memory module.
  (Not vendored into this repo.)

## Models

Experiments load third-party open-weight models (Qwen, Gemma, Llama) as frozen bases. Their weights are
**not redistributed** here; you download them yourself from their original sources under their own
licenses.
