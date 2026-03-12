# SDAIA GPT Project 2026

Build Your Own GPT from Scratch — Decoder-Only Transformer implementation using PyTorch.

## Project Structure

```
SDAIA-gpt-project2026/
├── data/
│   ├── pretrain/          # Raw text corpus (.txt)
│   └── finetune/          # SFT instruction data (.jsonl)
├── src/
│   ├── model/             # Transformer architecture (attention, blocks, GPT)
│   ├── tokenizer/         # BPE tokenizer and vocabulary
│   ├── training/          # Pretraining and fine-tuning loops
│   ├── evaluation/        # Metrics, LLM-as-judge, error analysis
│   └── demo/              # Streamlit/Gradio deployment app
├── checkpoints/
│   ├── pretrained/        # Pretrained model checkpoints
│   └── finetuned/         # Fine-tuned model checkpoints
├── results/
│   ├── sample_generations/ # Generated text samples
│   └── plots/             # Loss curves and evaluation plots
├── demo/
│   ├── recording/         # Recorded demo video
│   └── screenshots/       # Demo screenshots
├── notebooks/             # Exploration and evaluation notebooks
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Spec Files

Requirements, design, and tasks are managed in `.kiro/specs/gpt-from-scratch/`.
