<div align="center">

# PolyPlex

[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-black)](LICENSE)
[![Base model](https://img.shields.io/badge/base-PersonaPlex%207B-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Grafted](https://img.shields.io/badge/grafted-Qwen3--8B%20FFN-615CED)](https://huggingface.co/Qwen/Qwen3-8B)
[![Status](https://img.shields.io/badge/status-research%20in%20progress-orange)](RESEARCH.md#4-current-status)

[Overview](#-overview) · [Why](#-why-polyplex) · [How it works](#-how-it-works) · [Quickstart](#-quickstart) ·
[Repository layout](#-repository-layout) · [Documentation](#-documentation) · [한국어](docs/locales/README_ko.md)

</div>

---

## 🔭 Overview

Full-duplex spoken language models such as [Moshi](https://arxiv.org/abs/2410.00037) and
[PersonaPlex](https://arxiv.org/abs/2602.06053) listen and speak at the same time. They decide on their own when to
answer, when to back-channel and when to let the user interrupt. Today they speak English.

---

## 🚀 Quickstart

### 1. Install

```bash
git clone https://github.com/latentforge/PolyPlex
cd PolyPlex
uv sync --all-extras --inexact
```

A Blackwell GPU uses the `cu128` dependency group (the default). Weights, datasets and run outputs are written under
`.cache/` (`HF_HOME`).

Accept the [PersonaPlex model license](https://huggingface.co/nvidia/personaplex-7b-v1) and log in to the Hugging Face
Hub before the first run.

### 2. Run the pipeline

Every stage is driven by `./train.sh`, and every path and hyperparameter is an environment variable, so a run
is reproduced by exporting the same values.

```bash
./train.sh assemble  # assemble the DualFFN checkpoint from PersonaPlex and Qwen3-8B
./train.sh expand    # extend the vocabulary of the base checkpoint
./train.sh data1     # build the Phase 1 text mixture
./train.sh english   # cache the English states the preservation loss reads
./train.sh phase1    # Phase 1: text
./train.sh merge1    # fold the trained embeddings back into a checkpoint
```

`assemble` takes the attention, the original FFN, the normalizations, the audio embeddings and the depth decoder from
PersonaPlex, and the paired layers' FFNs, the text embedding, the LM head and the tokenizer from Qwen3-8B. The stages
from `expand` on still run the first design and are being moved to the DualFFN model.

### 3. Evaluate

`evaluate parallel` compares Korean with English on the same sentences, and `evaluate generate` checks generation
in English, while speaking and in silence. Caches and measurements have their own entry points:
`polyplex_research.datasets.duplex_data` (`text`, `merge`), `polyplex_research.datasets.ksponspeech` (`mimi`,
`duplex`, `import`) and `polyplex_research.utils.evaluate` (`english-states`, `capacity`, `basis`).

---

## 🗂️ Repository layout

```
polyplex_research/
├── models/        PersonaPlex relays and the Polyplex model (configuration, modeling, processing, tokenization)
├── adapters/      Trainable additions, one package per design: phase_1_<name>_adapter, phase_2_<name>_adapter
├── datasets/      Corpora, duplex caches and collators
└── utils/         Trainers, losses, optimizer (Muon + AdamW), NVFP4 quantization, assembly, evaluation
tests/             Unit tests on tiny models, mirroring the package tree
train.sh           Pipeline driver, one stage per command
docs/              Intent, spec, the v1 design record (metrics/VERSION_1.md) and the Korean README
```

---

## 🙏 Acknowledgements

PolyPlex builds on [PersonaPlex](https://github.com/NVIDIA/personaplex) by NVIDIA and on
[Moshi](https://github.com/kyutai-labs/moshi) by Kyutai. It grafts the FFNs of
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) by the Qwen team. Korean data comes from AI Hub (multi-session daily
conversation, colloquial Korean-English parallel corpus) and KsponSpeech.

## 📄 License

The code in this repository is released under the [Apache License 2.0](LICENSE). The PersonaPlex weights are released
by NVIDIA under the NVIDIA Open Model License, Qwen3 under Apache 2.0, and the datasets under their own terms.
