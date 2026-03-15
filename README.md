# SDAIA GPT Project 2026

A complete decoder-only transformer (GPT) built from scratch in PyTorch. Supports Arabic and English text end-to-end, covering data preprocessing, byte-level BPE tokenization, transformer architecture, pretraining, supervised fine-tuning (SFT), text generation, evaluation, and a Streamlit demo interface.

## Overview

This project implements a GPT-2 style language model without relying on any pre-built transformer libraries. Every component — from attention mechanisms to the tokenizer — is written from scratch to demonstrate deep understanding of the architecture.

### Key Design Decisions

- **Pre-norm transformer**: LayerNorm before attention and MLP (GPT-2 style) for training stability
- **Byte-level BPE**: Tokenizer operates on UTF-8 bytes, handling Arabic multi-byte characters natively
- **Arabic normalization**: Alef variant normalization (آ أ إ → ا) built into preprocessing
- **Property-based testing**: 27 correctness properties validated with Hypothesis (100+ iterations each)
- **Streamlit demo**: RTL-aware web interface for interactive text generation

### Architecture

```
Input Text → BPE Tokenizer → Token IDs
    → Token Embedding + Positional Embedding
    → N × TransformerBlock (LayerNorm → MultiHeadAttention → Residual → LayerNorm → MLP → Residual)
    → Final LayerNorm → Linear Head → Logits (vocab_size)
```

Default configuration: 256-dim embeddings, 8 attention heads, 6 layers, 512 max sequence length, 8000 vocabulary.

## Project Structure

```
SDAIA-gpt-project2026/
├── data/
│   ├── pretrain/                    # Raw text corpus (.txt)
│   └── finetune/
│       ├── arabic_sft.jsonl         # General Arabic SFT data (15 records)
│       ├── story_completion/        # Arabic story completion (12 records)
│       └── poetry/                  # Arabic poetry (12 records)
├── src/
│   ├── config.py                    # GPTConfig, TrainConfig, SFTConfig
│   ├── data/preprocessor.py         # DataPreprocessor (cleaning, normalization)
│   ├── tokenizer/
│   │   ├── vocab.py                 # Vocabulary (bidirectional mapping)
│   │   └── bpe_tokenizer.py         # BPETokenizer (byte-level BPE)
│   ├── model/
│   │   ├── attention.py             # ScaledDotProductAttention, MultiHeadAttention
│   │   └── transformer.py           # TransformerBlock, GPTModel
│   ├── training/
│   │   ├── pretrain.py              # PretrainPipeline (AdamW, warmup+cosine LR)
│   │   └── finetune.py              # SFTPipeline (instruction-masked loss)
│   ├── generation/generator.py      # TextGenerator (greedy, top-k, top-p, temperature)
│   ├── evaluation/
│   │   ├── metrics.py               # Evaluator (perplexity, samples, loss plots)
│   │   ├── human_eval.py            # LLMJudge (external LLM scoring)
│   │   └── error_analysis.py        # ErrorAnalyzer (repetition, coherence)
│   └── demo/app.py                  # Streamlit web interface
├── checkpoints/
│   ├── pretrained/                  # Pretrained model checkpoints
│   └── finetuned/                   # Fine-tuned model checkpoints
├── results/
│   ├── sample_generations/          # Generated text samples
│   └── plots/                       # Loss curves
├── demo/
│   ├── recording/                   # Demo video script
│   └── screenshots/                 # Demo screenshots
├── notebooks/                       # 11 Jupyter notebooks (mirrors src/)
├── tests/                           # pytest + Hypothesis property tests
├── run_pretrain.py                  # Main pretraining script
├── run_finetune.py                  # Main fine-tuning script
├── run_evaluate.py                  # Main evaluation script
└── requirements.txt
```

## Setup

```bash
# Clone and install dependencies
cd SDAIA-gpt-project2026
pip install -r requirements.txt

# Run tests (98 tests, 27 property-based)
python -m pytest tests/ -v
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- hypothesis, pytest, streamlit, matplotlib, numpy, requests

## Pipeline Phases

### 1. Data Preprocessing

The `DataPreprocessor` handles:
- UTF-8 validation with byte-offset error reporting
- HTML tag and control character removal
- Arabic alef normalization (آ أ إ → ا)
- JSONL/JSON parsing for SFT instruction datasets
- Corpus and dataset size validation

### 2. Tokenizer Training

Byte-level BPE tokenizer that:
- Operates on raw UTF-8 bytes (no code-point splitting for Arabic)
- Learns merge rules from corpus text
- Includes special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`
- Supports save/load for persistence

### 3. Pretraining

Next-token prediction with:
- AdamW optimizer with configurable weight decay and betas
- Linear warmup + cosine decay learning rate schedule
- Train/validation split with periodic evaluation
- Overfitting detection (patience-based)
- Checkpoint saving with full config for reproducibility
- Deterministic seeding (Python, NumPy, PyTorch)

### 4. Supervised Fine-Tuning (SFT)

Instruction-following fine-tuning:
- Formats instruction + optional input + separator + output
- Loss computed only on output tokens (instruction tokens masked)
- Lower learning rate than pretraining
- Separate checkpoint directory

### 5. Text Generation

Multiple sampling strategies:
- Greedy decoding (temperature=0)
- Temperature scaling
- Top-k sampling
- Top-p (nucleus) sampling
- Stops at `<eos>` or max token limit

### 6. Evaluation

- Perplexity: exp(average cross-entropy loss)
- Sample generation with prompt lists
- Loss curve plotting
- LLM-as-judge scoring (fluency, coherence, instruction-following)
- Error analysis (repetition, incoherence, off-topic, grammatical)

## Running the Pipeline

```bash
# Pretrain (requires corpus in data/pretrain/data.txt)
python run_pretrain.py --corpus data/pretrain/data.txt --max-steps 10000

# Fine-tune on Arabic SFT data
python run_finetune.py --sft-data data/finetune/arabic_sft.jsonl

# Evaluate
python run_evaluate.py --checkpoint checkpoints/finetuned/checkpoint_latest.pt

# Launch Streamlit demo
streamlit run src/demo/app.py
```

## Arabic Language Support

The project supports Arabic text natively:
- Byte-level BPE handles Arabic UTF-8 without code-point splitting
- Alef variant normalization in preprocessing
- Arabic SFT datasets: story completion, poetry, dialect translation, general tasks
- RTL text rendering in Streamlit demo and notebooks

### Arabic Datasets

| Dataset | Records | Description |
|---------|---------|-------------|
| `arabic_sft.jsonl` | 15 | Mixed: stories, poetry, dialect, summarization |
| `story_completion/arabic_stories.jsonl` | 12 | Arabic story continuation tasks |
| `poetry/arabic_poetry.jsonl` | 12 | Classical, Nabati, and free verse poetry |

## Testing

98 tests total: unit tests + 27 property-based tests using Hypothesis.

```bash
python -m pytest tests/ -v --tb=short
```

Property tests validate correctness properties including:
- UTF-8 output validity and Arabic normalization idempotency
- Tokenizer encode/decode round-trip for Arabic and English
- Attention shape invariants and causal mask correctness
- Learning rate schedule monotonicity
- SFT loss masking and example formatting
- Generation determinism and length bounds
- Reproducibility via seed

## Technical Limitations

- **Context length**: 512 tokens max (configurable but limited by memory)
- **Vocabulary**: 8000 tokens default — sufficient for demo, not production
- **Compute**: Designed for single-GPU or CPU training on small corpora
- **No pre-trained weights**: Model must be trained from scratch each time

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | Config | Configuration dataclasses |
| 02 | Data Preprocessing | Cleaning, normalization, SFT parsing |
| 03 | Tokenizer | BPE training, encode/decode |
| 04 | Attention | Scaled dot-product and multi-head attention |
| 05 | Model | TransformerBlock and GPTModel |
| 06 | Training | Pretraining and fine-tuning pipelines |
| 07 | Generation | Text generation with sampling strategies |
| 08 | Evaluation | Perplexity, error analysis |
| 09 | Demo | Interactive generation (non-Streamlit) |
| 10 | Arabic Demo | Arabic preprocessing, tokenization, RTL rendering |
| 11 | End-to-End | Full pipeline in one notebook |
