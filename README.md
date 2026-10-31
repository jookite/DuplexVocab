<div align="center">

# DuplexVocab

**Improving Korean pronunciation in RVQ-based full-duplex spoken language models with a conversational tokenizer**

[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-black)](LICENSE)
[![Base model](https://img.shields.io/badge/base-PersonaPlex%207B-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Text LLM](https://img.shields.io/badge/text%20LLM-Qwen3--8B-615CED)](https://huggingface.co/Qwen/Qwen3-8B)

[Overview](#-overview) · [Problem](#-problem) · [Method](#-method) · [Results](#-results) ·
[Quickstart](#-quickstart) · [Repository layout](#%EF%B8%8F-repository-layout) · [Team](#-team)

</div>

---

## 🔭 Overview

Full-duplex spoken language models such as [Moshi](https://arxiv.org/abs/2410.00037) and
[PersonaPlex](https://arxiv.org/abs/2602.06053) listen and speak at the same time, so they can back-channel and be
interrupted like a person in a real conversation. Each 80 ms frame (12.5 Hz) sums three streams, the user's audio,
the agent's text and the agent's audio, and the model emits **one text token per frame**.

That makes the tokenizer a question of *time* rather than *cost*. In a text LLM, a word that falls back to bytes only
makes generation a little slower. In a duplex model, every extra token takes up a frame of real speech. DuplexVocab
redesigns the text tokenizer and vocabulary of a duplex model around conversational speech, so that Korean (and the
other languages Qwen3 covers) stops spilling out of the frames it is spoken in.

This repository was built for the *Trustworthy AI* team project (Team 2, 2026).

## 🧩 Problem

### Byte fallback is worst on conversational text

When a character is missing from the vocabulary, a byte-level tokenizer writes it as UTF-8 bytes. A Hangul syllable
is three bytes, so `안녕` becomes six tokens instead of two: six frames, or 480 ms. LLM vocabularies are built from
web text, and colloquial endings, interjections and contractions are rare there, so the fallback rate jumps on
conversational data:

| Language | Wikipedia | Movie subtitles (OpenSubtitles) |
| -------- | --------: | ------------------------------: |
| Korean   |    13.2 % |                          21.3 % |
| Japanese |     8.0 % |                          20.5 % |
| Chinese  |     3.0 % |                          21.2 % |
| English  |     0.1 % |                          0.01 % |

<sub>Qwen3-8B tokenizer, 3,000 sentences per language. On the colloquial Korean validation set (AI Hub colloquial
Korean-English, 3,000 sentences) the rate is 15.7 %.</sub>

### Longer text becomes cut-off text

Audio arrives in real time and is never cut, but the text stream has to fit inside the frames a word is spoken in.
When the tokens run past that span, Moshi's layout overwrites them with `PAD`, or packs them together when there is
even less room. The text is truncated. This leads to two failures:

- **During training:** teacher forcing feeds the full ground-truth speech. The model therefore learns a wrong
  pairing, *truncated text ↔ complete speech*. Ordinary TTS and text LMs never see this, because they have no fixed
  one-token-per-frame budget.
- **During generation:**
  - **Case 1, truncation:** the text comes out cut off, and the speech tied to it is cut off too.
  - **Case 2, drawl:** the text backbone completes the word anyway, the speech cannot keep up, and the utterance
    slows down and drags.

## 🛠️ Method

### 1. A conversational Unigram tokenizer

| | Before: Qwen3 byte-level BPE | Ours: Unigram + byte fallback |
| --- | --- | --- |
| Segmentation | Merges learned on web text | Unigram scores learned by hard EM on 5.94 M colloquial words |
| Out-of-vocabulary syllable | Three UTF-8 byte pieces | One of 256 byte-fallback tokens, used only for characters outside the vocabulary |
| Adding tokens | Hard, because merges are fixed | Append to the piece list and use it right away |
| Word start | `Ġ` | `▁`, the same mark as Helium (SentencePiece), so it lines up with the word starts `text_stream` places |

The token list is the same Qwen3 + Helium list in both cases. Helium's 32,000 tokens keep their ids, so the existing
embedding and head rows are preserved and only new rows are added.

### 2. A multilingual vocabulary built in tiers

The goal is not average compression but **"is a frequent short word a single piece?"**. For every language we measure
the **word cut rate** on real conversational speech, using YODAS2 with word timings from Whisper large-v3
cross-attention DTW. The vocabulary grows only until that rate falls below the target: 1 % in general, and the lowest
reachable rate + 0.2 pp for English, Korean, Japanese and Chinese. Each language therefore gets the size it needs.

- **Tier 1, character floor:** every standard character (KS X 1001, JIS X 0208, GB 2312, CLDR) plus its word-start
  form.
- **Tier 2, frequent word forms:** spoken words seen at least twice, added 250 at a time in order of frames saved.
- **Tier 3, subwords:** subwords for words that still get cut, 1,000 at a time (up to 20,000), plus non-standard
  characters seen at least three times.

Both orders, tier 2 first and tier 3 first, are tried per language, and the one that reaches the target with fewer
tokens is kept. Languages Qwen3 already covers barely grow (English +10, Arabic +41), while Sinhala +15,145,
Japanese +8,820, Chinese +6,853 and Korean +6,230 grow the most. The final vocabulary holds **223,973** tokens.

The whole build is in [`build_vocab.ipynb`](build_vocab.ipynb).

### 3. Grafting the vocabulary into the model

- **New rows start from existing ones, not at random.** Each new embedding row is a sparse combination of 32
  existing rows plus a residual, and the output head rows add a prior correction. Every new token therefore starts
  near similar existing tokens.
- **Frozen:** the Temporal and Depth Transformers, Mimi and the existing embeddings.
- **Trained:** the appended rows and low-rank corrections in some layers. The input embedding and the output head
  share (tie) the new rows.
- **Controlled comparison:** the two runs use the same model and differ only in the text vocabulary.
  - **Before:** Helium 32,000 + Qwen3 minus duplicates.
  - **Ours:** Helium 32,000 + Qwen3 minus written-only words, plus the conversational duplex vocabulary.

## 📊 Results

| Condition | Byte fallback | WER ↓ | UTMOS ↑ |
| --- | ---: | ---: | ---: |
| Qwen3 tokenizer as is | 15.7 % | 18.4 | 3.21 |
| **Ours** (conversational Unigram) | **0.01 %** | **14.7** | **3.68** |

<sub>3,000 cases, LibriTTS-R speech generation evaluation. WER rises with the number of truncated cases. UTMOS predicts
listener-rated quality and also reflects drawl and slurring.</sub>

Tokenizers on the colloquial validation split (3,000 Korean and 3,000 English sentences, disjoint from training):

| Tokenizer | Korean tokens / word | Korean byte pieces | English tokens / word | English byte pieces |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 byte BPE | 2.76 | 15.70 % | 1.26 | 0.00 % |
| Unigram (same list) | 2.70 | 0.00 % | 1.26 | 0.00 % |
| Unigram + tier 1 | 2.51 | 0.00 % | 1.25 | 0.00 % |

The token sequences show where the frames go. `속마음을` spans 11 frames: Qwen3 spends 4 tokens on it, 2 of them
byte pieces, while ours spends 3. `빨리` spans only 3 frames: Qwen3's 4 tokens run past the span and get cut, while
ours fits in 2.

## ⚠️ Limitations

- **Duplex ability is not benchmarked.** The effect of the tokenizer on turn-taking and interruption was not measured.
- **The model grows.** With 223k tokens, the embedding and head have more rows and therefore more parameters.
- **Below 1 % fallback is not enough.** One word in a hundred still matters, and below 0.5 % has not been reached
  yet.

The tokenizer is a transitional fix. A structural fix would remove the tokenizer altogether (BLT/Bolmo, H-Nets), or
keep generated text and generated speech from sharing one self-attention so they do not compete for the same frames.
The takeaway is that **the tokenizer and embeddings of a duplex LM have to be designed by different criteria than
those of a text LLM.**

---

## 🚀 Quickstart

### 1. Install

```bash
uv sync --all-extras --inexact
```

A Blackwell GPU uses the `cu128` dependency group (the default). Weights, datasets and run outputs are written under
`.cache/` (`HF_HOME`). Accept the [PersonaPlex model license](https://huggingface.co/nvidia/personaplex-7b-v1) and
log in to the Hugging Face Hub before the first run.

### 2. Build the vocabulary

Run [`build_vocab.ipynb`](build_vocab.ipynb) from top to bottom. It matches the base vocabularies, selects tiers 1-3
per language, trains the Unigram scores and saves the Polyplex tokenizer.

### 3. Run the pipeline

Every stage is driven by `./train.sh`, and every path and hyperparameter is an environment variable, so a run is
reproduced by exporting the same values.

```bash
./train.sh assemble  # assemble the checkpoint from PersonaPlex and Qwen3-8B
./train.sh expand    # extend the text vocabulary of the base checkpoint
./train.sh data1     # build the Phase 1 text mixture
./train.sh phase1    # Phase 1: train the new text rows
./train.sh merge1    # fold the trained rows back into a checkpoint
./train.sh data2     # write the Phase 2 speech caches from KsponSpeech
./train.sh phase2    # Phase 2: speaking, scored through the depth decoder
./train.sh score     # score a run on the Korean validation set and held-out English speech
```

`./train.sh` without arguments lists every stage. The modeling code of the backbone
(`polyplex_research/models/polyplex/modeling_polyplex.py`) is part of an ongoing research project and is withheld from
this repository.

---

## 🗂️ Repository layout

```
polyplex_research/
├── models/        PersonaPlex relays and the Polyplex configuration, processor and tokenizer
├── adapters/      Trainable additions, one package per design: phase_1_<name>_adapter, phase_2_<name>_adapter
├── datasets/      Corpora, duplex caches and collators (training data and the KsponSpeech evaluation set)
└── utils/         Trainers, losses, optimizer (Muon + AdamW), NVFP4 quantization, assembly, evaluation (WER, CER)
tests/             Unit tests on tiny models, mirroring the package tree
build_vocab.ipynb  Vocabulary construction and tokenizer training
train.sh           Pipeline driver, one stage per command
```

---

## 👥 Team

| Member | Role |
| --- | --- |
| 서주연 · [jookite](https://github.com/jookite) | Method design, vocabulary construction and tokenizer weight training |
| 문채운 · [b-re-w](https://github.com/b-re-w) | Problem observation and topic proposal, representation correction training, trainer |
| 유채민 · [Gtend](https://github.com/Gtend) | Comparison experiment design, tokenizer class design (BPE → Unigram), training datasets |
| 정현민 · [jung-hyeonmin](https://github.com/jung-hyeonmin) | Evaluation benchmark and evaluation datasets, WER/UTMOS evaluation, case and qualitative analysis |

## 🙏 Acknowledgements

This work builds on [PersonaPlex](https://github.com/NVIDIA/personaplex) by NVIDIA, on
[Moshi](https://github.com/kyutai-labs/moshi) by Kyutai, and on [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) by
the Qwen team. The data comes from AI Hub (multi-session daily conversation, colloquial Korean-English parallel
corpus), KsponSpeech, YODAS2, FLEURS, OpenSubtitles, LibriTTS-R, FineWeb-Edu, FineWeb-2 and HAERAE KOREAN-WEBTEXT.

## 📄 License

The code in this repository is released under the [Apache License 2.0](LICENSE). The PersonaPlex weights are released
by NVIDIA under the NVIDIA Open Model License, Qwen3 under Apache 2.0, and the datasets under their own terms.
