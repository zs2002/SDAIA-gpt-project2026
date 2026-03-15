# GPT from Scratch — Demo Recording Script

## Section 1: Introduction (2-3 minutes)

### Project Overview
- Introduce the project: "Building a GPT model from scratch using PyTorch"
- Explain the goal: understand transformer architecture by implementing every component
- Mention Arabic + English bilingual support

### Data Summary
- Pretraining corpus: raw text data for next-token prediction
- SFT datasets: 39 Arabic instruction-response pairs across 3 categories
  - General Arabic tasks (15 records): dialect translation, summarization, explanation
  - Story completion (12 records): Arabic narrative continuation
  - Poetry (12 records): classical, Nabati, and free verse Arabic poetry

### Architecture Overview
- Show the architecture diagram (pre-norm transformer)
- Key specs: 256-dim, 8 heads, 6 layers, 512 max seq len, 8000 vocab

---

## Section 2: Pre-trained Model Demo (3-4 minutes)

### Prompt 1: Arabic text
- Input: "في يوم من الأيام"
- Show generated output, discuss quality of untrained vs trained model

### Prompt 2: English text
- Input: "The quick brown fox"
- Show generated output

### Prompt 3: Mixed Arabic-English
- Input: "مرحبا Hello World"
- Demonstrate bilingual handling

### Discussion Points
- How the byte-level BPE handles Arabic without code-point splitting
- Generation quality at this stage (random but structurally valid)
- Show temperature and top-k/top-p parameter effects

---

## Section 3: Fine-tuned Model Demo (3-4 minutes)

### Task 1: Story Completion
- Input: Arabic story prompt from the SFT dataset
- Show side-by-side: pretrained vs fine-tuned output
- Discuss improvement in coherence and relevance

### Task 2: Poetry Generation
- Input: "اكتب أبياتا من الشعر عن الوطن"
- Show side-by-side comparison
- Discuss how SFT improves instruction following

### Discussion Points
- How instruction masking in SFT loss helps the model learn to follow instructions
- Difference in output quality between pretrained and fine-tuned models
- Limitations with small dataset size

---

## Section 4: Evaluation Results (2-3 minutes)

### Perplexity
- Show perplexity values for pretrained and fine-tuned models
- Explain what perplexity means (lower = better prediction)

### Sample Generations
- Show saved samples from `results/sample_generations/`
- Highlight good and bad examples

### Error Analysis
- Show error analysis report categories:
  - Repetition detection
  - Coherence assessment
  - Off-topic detection

### Loss Curves
- Show training and validation loss plots from `results/plots/`
- Discuss convergence behavior

---

## Section 5: Conclusion (1-2 minutes)

### Key Learnings
- Understanding attention mechanisms and causal masking
- Importance of byte-level tokenization for multilingual support
- How SFT with instruction masking improves task performance
- Value of property-based testing for correctness guarantees

### Future Work
- Scale to larger corpus and vocabulary
- Implement KV-cache for faster inference
- Add RLHF or DPO alignment
- Expand Arabic dialect coverage
- Implement flash attention for memory efficiency

### Technical Stats
- 98 tests passing (27 property-based with Hypothesis)
- 27 formal correctness properties validated
- 11 Jupyter notebooks documenting every component
- Full pipeline: preprocessing → tokenization → training → generation → evaluation
