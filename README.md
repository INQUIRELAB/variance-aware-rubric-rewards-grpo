# Variance-Aware Rubric Rewards for Heart-Focused Medical QA with GRPO

<p align="center">
  <img src="evaluation_animation.gif" alt="Cumulative accuracy of all evaluated models across 500 held-out heart-related HealthBench samples" width="800">
</p>

---

**Improving Heart-Focused Medical Question Answering in LLMs via Variance-Aware Rubric Rewards with Group Relative Policy Optimization**

Arash Ahmadi\*, Parisa Masnadi Khiabani\*, Sarah Sharif, Charles Nicholson, David Ebert, Mike Banad

<sub>School of Electrical and Computer Engineering, University of Oklahoma · Data Science and Analytics Institute, University of Oklahoma · School of Industrial and Systems Engineering, University of Oklahoma · Office of Responsible Artificial Intelligence (ORAI), University of Arizona · INQUIRE Laboratory, University of Oklahoma · Data Institute for Societal Challenges (DISC), University of Oklahoma</sub>

<sub>\*Co-first authors with equal contribution. Correspondence: bana@ou.edu</sub>

---

## Abstract

Large language models have shown strong promise in healthcare applications, yet deploying them in clinical settings remains difficult due to data privacy constraints, inference costs, and limited suitability for edge or on-device use. This repository contains the training and evaluation pipeline for a GRPO-based post-training framework that optimizes Qwen3-14B against physician-written rubric criteria for heart-related medical question answering. We propose a **Variance-Aware Reward Framework** that replaces weighted binary criterion aggregation and single overall Likert-style scoring with continuous analytical reward functions derived from criterion-level rubric outcomes. This formulation provides richer optimization signals for feedback that is sparse, multi-criteria, and difficult to verify automatically.

On a held-out heart-related subset of HealthBench ($n=500$, seed 42), our best GRPO variant improves accuracy from **0.362 to 0.502** (+38.7%) and F1 from **0.532 to 0.668** (+25.7%) relative to the Qwen3-14B base model, while remaining competitive with GPT-OSS-120B (0.508 accuracy, 0.674 F1). The final model runs on a single workstation GPU with 4-bit quantization and LoRA adapters, supporting privacy-preserving local deployment.

## Results

### Full Model Comparison

All models were evaluated on the held-out heart-related HealthBench subset ($n=500$, seed 42). Models are sorted by accuracy.

| Model | Accuracy | 95% CI | F1 |
|-------|----------|--------|-----|
| Kimi-K2 (~1T params) | **0.570** | [0.526, 0.612] | **0.726** |
| GPT-OSS-120B | 0.508 | [0.466, 0.552] | 0.674 |
| **Qwen3-14B GRPO (Complexity)** | **0.502** | [0.460, 0.546] | **0.668** |
| **Qwen3-14B GRPO (Hybrid)** | **0.498** | [0.454, 0.542] | **0.665** |
| MedGemma-27B | 0.448 | [0.406, 0.492] | 0.619 |
| Gemma3-12B | 0.442 | [0.398, 0.486] | 0.613 |
| Phi4-14B | 0.442 | [0.398, 0.486] | 0.613 |
| Llama-4-Scout-17B | 0.432 | [0.388, 0.476] | 0.603 |
| Qwen3-14B GRPO (RaR-Implicit) | 0.412 | [0.370, 0.456] | 0.584 |
| Llama-3.3-70B | 0.398 | [0.356, 0.442] | 0.569 |
| Qwen3-14B GRPO (RaR-Explicit) | 0.396 | [0.354, 0.438] | 0.567 |
| MedGemma-4B | 0.396 | [0.354, 0.438] | 0.567 |
| Llama-4-Maverick-17B | 0.390 | [0.348, 0.432] | 0.561 |
| Qwen3-14B Base | 0.362 | [0.320, 0.402] | 0.532 |
| MedGemma-1.5-4B | 0.322 | [0.282, 0.362] | 0.487 |

The locally trained GRPO variants (Complexity and Hybrid) achieve performance on par with GPT-OSS-120B, a model roughly an order of magnitude larger. Kimi-K2 achieves the highest overall performance, but its ~1 trillion parameters far exceed the memory capacity of academic hardware, making local training or serving infeasible.

### Improvement Over the Base Model

| GRPO Variant | Acc Δ | Acc Δ% | F1 Δ | F1 Δ% |
|-------------|-------|--------|------|--------|
| Complexity | +0.140 | +38.7% | +0.137 | +25.7% |
| Hybrid | +0.136 | +37.6% | +0.133 | +25.0% |
| RaR-Implicit | +0.050 | +13.8% | +0.052 | +9.8% |
| RaR-Explicit | +0.034 | +9.4% | +0.036 | +6.7% |

Pairwise McNemar tests confirm that both variance-aware rewards (Complexity and Hybrid) outperform the RaR-Explicit ($p < 10^{-5}$) and RaR-Implicit ($p < 10^{-3}$) baselines.

### Per-Criterion Evaluation Visualization

The animation below shows how each model satisfies or misses individual rubric criteria on the evaluation prompts. If the video does not play inline, [click here to view or download it](criteria_animation.mp4).

<p align="center">
  <video src="criteria_animation.mp4" controls width="800">
  </video>
</p>

## Approach

The pipeline consists of four stages: data curation, supervised fine-tuning, reinforcement learning via GRPO, and comparative evaluation.

### Data Curation

Training data comes from [RaR-Medicine](https://huggingface.co/datasets/google/rar-medicine), a dataset of medical questions with reference answers and rubric annotations. Each rubric defines criteria that a good response should satisfy (positive criteria with positive point values) and behaviors it should avoid (negative criteria with negative point values). We filter the full corpus to heart-related queries using a dedicated classifier built on MedGemma 3:27B. The classifier assigns a binary heart-related label along with a theme category (e.g., heart attack, hypertension, arrhythmia) and keyword evidence grounded in the original query text. A keyword verification step ensures that the classifier's decisions are traceable to actual content in the prompt rather than hallucinated associations.

The filtered heart-related subset is augmented with synthetic reasoning traces generated by MedGemma 3:27B. These traces provide step-by-step clinical reasoning that connects the question to the reference answer, written in continuous prose. The augmented training split is then divided 50/50 into disjoint subsets for supervised fine-tuning and GRPO training.

Evaluation uses a separate held-out dataset, [HealthBench](https://github.com/openai/healthbench), which provides 5,000 multi-turn medical conversations graded by 262 physicians from 26 medical specialties and 60 countries, with over 48,000 unique rubric criteria spanning seven clinical themes and five behavioral axes. We evaluate on a non-synthetic, heart-related subset of 500 examples.

### Supervised Fine-Tuning (SFT)

The first training stage teaches the model a structured output format that separates reasoning from recommendations. Responses are formatted with explicit `<start_working_out>` / `<end_working_out>` tags for the reasoning trace and `<SOLUTION>` / `</SOLUTION>` tags for the final answer. Instruction tokens (everything before the reasoning section) are masked during training so the model only learns to generate responses, not to memorize prompts. The model uses LoRA adapters (rank 16, alpha 32) targeting all attention and MLP projections, with gradient checkpointing and 4-bit quantization to keep training feasible on academic hardware.

### Group Relative Policy Optimization (GRPO)

The second stage applies GRPO to optimize response quality against rubric-based rewards. GRPO avoids a separate critic/value model and instead computes advantages from relative scores among sampled completions for each prompt. This reduces memory requirements compared to PPO-style methods and fits reward settings where relative ranking is more reliable than absolute calibration.

For each training prompt, the model generates $G=6$ completions. A judge model (GPT-OSS-120B, served via the Groq inference platform) evaluates each completion against every rubric criterion independently, returning a binary present/absent decision per criterion. The criterion-level verdicts are aggregated into positive and negative scores, which are then transformed into a scalar reward through one of the reward shaping functions described below.

Training focuses on prompts with more than three rubric criteria, filtering out trivially simple examples that would produce uninformative reward signals. Each reward variant trains for 1,000 GRPO steps (~26 hours on a single NVIDIA RTX 6000 PRO).

### Reward Functions

A central challenge in applying GRPO to rubric-based tasks is that naive binary rewards (perfect score or zero) produce near-zero variance in each sampled group, collapsing the advantage estimates and preventing learning. Our reward functions address this by preserving partial-credit information and accounting for rubric complexity. Both satisfy four prerequisites for effective GRPO training: non-zero variance, monotonicity with respect to quality, preservation of partial credit, and awareness of task complexity.

**Hybrid Reward.** This function combines a continuous base reward proportional to the normalized rubric score with a discrete perfection bonus:

$$r_{\text{hybrid}} = \max(0,\; 15 \cdot s_{\text{norm}} - 4.5 \cdot \rho) + 5 \cdot \mathbb{1}[\text{all positive met} \wedge \text{no negative triggered}]$$

The base portion (up to 15 points) scales linearly with the fraction of positive criteria met, with a penalty proportional to the fraction of negative criteria triggered. A 5-point bonus is added only when all positive criteria are satisfied and no negative criteria are triggered. This design provides gradient signal for incremental improvements while maintaining a strong incentive for fully correct and safe responses.

**Complexity-Aware Reward.** This function applies a power-law transformation to the normalized score and scales the result by a logarithmic complexity factor:

$$r_{\text{complexity}} = 20 \cdot \hat{s}^{1.2} \cdot \left(1 + 0.2 \cdot \frac{\log(1 + n_c)}{\log(26)}\right)$$

where $\hat{s} = \max(0,\; s_{\text{norm}} - 0.5\rho)$ is the penalty-adjusted score and $n_c$ is the number of rubric criteria. The convex power curve ($\alpha = 1.2$) emphasizes high-quality responses while the complexity scaling assigns higher rewards to prompts with more criteria, which represent harder evaluation targets. This amplifies learning from the prompts where the base model struggles most.

### Why Variance-Aware Rewards Outperform RaR Aggregation

We also trained GRPO variants using the reward strategies from the original Rubrics as Rewards (RaR) framework: RaR-Explicit (weighted binary aggregation) and RaR-Implicit (holistic Likert scoring). Both produced only modest gains (+9.4% and +13.8% relative accuracy), far below the +38.7% and +37.6% from Complexity and Hybrid rewards. Three factors explain the gap:

1. **Fixed categorical weights.** RaR-Explicit uses rigid, hand-tuned weights that impose the same importance hierarchy on every prompt, regardless of clinical context.
2. **Coarse holistic scoring.** RaR-Implicit collapses multi-dimensional rubric information into a single Likert score, discarding granular criterion-level signal.
3. **No complexity awareness.** Neither RaR strategy distinguishes between a prompt with 5 criteria and one with 18. A base model can often satisfy all criteria on simple rubrics out of the box, so those easy prompts contribute little training signal. Satisfying 17 out of 18 criteria on a complex prompt represents a substantially greater achievement than a perfect score on a 5-criterion prompt, yet both RaR strategies assign comparable normalized rewards.

Our variance-aware rewards address this asymmetry. The Complexity-aware variant applies a logarithmic bonus that scales with rubric size, converting partial-credit differences on complex rubrics into high-value training signal.

## Dataset

The repository includes the heart-related RaR-Medicine dataset used for training:

**[`(dataset) rar_medicine_with_synthetic_reasoning_2026-02-09-06-26-40.jsonl`](<(dataset) rar_medicine_with_synthetic_reasoning_2026-02-09-06-26-40.jsonl>)**

| Property | Value |
|----------|-------|
| Records | 2,206 |
| Format | JSONL |
| Size | ~16 MB |
| Splits | train (1,808) · val (204) · test (194) |
| Heart themes | 14 cardiac categories |
| Source | Filtered from [RaR-Medicine](https://huggingface.co/datasets/google/rar-medicine) |
| Classifier | MedGemma 3:27B |
| Reasoning traces | MedGemma 3:27B |

Each record contains a medical question, a reference answer, a set of signed rubric criteria (positive for desired behaviors, negative for pitfalls), verified heart-related classification metadata, and a synthetic reasoning trace. The train split is divided 50/50 into disjoint subsets for SFT and GRPO.

See [DATASET.md](DATASET.md) for the full schema, field descriptions, theme list, and example records.

**[HealthBench](https://github.com/openai/healthbench)** (evaluation): A held-out medical benchmark with 5,000 multi-turn conversations and rubric annotations from 262 physicians in 26 medical specialties and 60 countries. We evaluate on a non-synthetic, heart-related subset of 500 examples. No HealthBench data appears in the training pipeline.

## Usage

### Prerequisites

Training was conducted on a single NVIDIA RTX 6000 PRO (Blackwell Workstation Edition, 600W TDP). The 4-bit quantized model with LoRA adapters can also be served on other workstation GPUs for inference. A Groq API key is required for the LLM judge during GRPO training and for frontier model evaluation.

```bash
pip install unsloth torch pandas numpy datasets trl pydantic seaborn matplotlib scikit-learn groq pyarrow
export GROQ_API_KEY="your-key-here"
```

### Running the Full Pipeline

The main script `Hybrid-Complexity-Qwen3-14B.py` supports modular execution through command-line flags:

```bash
# Full pipeline: SFT + GRPO + evaluation
python Hybrid-Complexity-Qwen3-14B.py --both-sft-grpo --eval-local --eval-groq

# Quick verification (5 SFT steps + 5 GRPO steps + limited eval)
python Hybrid-Complexity-Qwen3-14B.py --smoke-train

# SFT only
python Hybrid-Complexity-Qwen3-14B.py --only-sft

# GRPO only (requires existing SFT checkpoint)
python Hybrid-Complexity-Qwen3-14B.py --only-grpo --reward-type hybrid

# Evaluation only (requires existing checkpoints)
python Hybrid-Complexity-Qwen3-14B.py --only-eval --eval-local --eval-groq

# Run both reward types sequentially
python Hybrid-Complexity-Qwen3-14B.py --both-sft-grpo --reward-type all
```

### Key Options

| Flag | Description |
|------|-------------|
| `--reward-type {hybrid,complexity,all}` | Reward function for GRPO training (default: hybrid) |
| `--sft-steps N` | Override SFT training steps (default: 500) |
| `--grpo-steps N` | Override GRPO training steps (default: 1000) |
| `--eval-samples N` | Number of evaluation samples (default: 500) |
| `--gpu N` | Explicitly select a GPU index |
| `--gpu-failover` | Automatically retry on another GPU after OOM |
| `--skip-smoke-test` | Skip pre-training model verification |
| `--smoke-train` | Minimal pipeline verification (5 steps each) |

### Training Configuration

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Batch size | 4 (×4 gradient accumulation = 16 effective) | 1 prompt, 6 completions per prompt |
| Learning rate | 2×10⁻⁴ | 5×10⁻⁶ |
| Optimizer | AdamW 8-bit | AdamW 8-bit |
| Max sequence length | 4,096 tokens | 1,024 token completions |
| Steps | 500 (2 epochs) | 1,000 |
| LoRA rank | 16 (alpha 32) | 16 (alpha 32) |
| Quantization | 4-bit | 4-bit |
| Wall-clock time | ~2 hours | ~26 hours |

### GPU Auto-Selection

The script automatically queries `nvidia-smi` for GPU utilization and selects the GPU with the most free memory and less than 30% utilization. If `--gpu-failover` is enabled and training crashes with an out-of-memory error, the script automatically relaunches itself on the next available GPU with sufficient memory.

## Citation

If you use this code, dataset, or methodology in your research, please cite:

```bibtex
@article{ahmadi2026variance,
  title={Improving Heart-Focused Medical Question Answering in LLMs via Variance-Aware Rubric Rewards with GRPO},
  author={Ahmadi, Arash and Masnadi Khiabani, Parisa and Sharif, Sarah and Nicholson, Charles and Ebert, David and Banad, Mike},
  year={2026}
}
```

## License

Please refer to the individual dataset licenses for [RaR-Medicine](https://huggingface.co/datasets/google/rar-medicine) and [HealthBench](https://github.com/openai/healthbench). The training code in this repository is provided for research purposes.

---

