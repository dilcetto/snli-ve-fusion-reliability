# Why Two Multimodal Models Disagree on the Same Image

**A case study in evaluating AI reliability under cross-modal contradiction**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1k55WtdPBZwYwONQTafLjeqF03vbOxbs9)

## The question

When a vision-language model is shown an image and a caption that either matches or contradicts it, does *how* the model fuses vision and language change whether it gets fooled?

I ran a controlled comparison of two open-source multimodal models, LLaVA-1.5-7B and BLIP-2-OPT-2.7B, to find out. The result wasn't just "one model is better." One architecture failed in a single, predictable direction.

## Setup

- **Models compared:** LLaVA-1.5-7B (late fusion: visual features projected directly into the language model) vs. BLIP-2-OPT-2.7B (intermediate fusion: visual features compressed through a 32-token Q-Former bottleneck before reaching the language model)
- **Task:** Binary visual entailment on 1,000 stratified examples from SNLI-VE (500 entailment, 500 contradiction), scoring whether each model correctly judges if a caption is supported or contradicted by its paired image
- **Environment:** 4-bit quantised inference (bitsandbytes) on a single T4 GPU in Colab. Compute constraints shaped several methodology decisions, documented below rather than hidden.
- **Metrics:** per-condition accuracy, 95% confidence intervals, two-sample proportion (z) tests

## Results

| Model | Entailment accuracy | Contradiction accuracy | Gap |
|---|---|---|---|
| LLaVA-1.5-7B | 93.8% [91.7, 95.9] | 91.0% [88.5, 93.5] | +2.8 pp (CI crosses zero; not significant) |
| BLIP-2-OPT-2.7B | 26.0% [22.2, 29.8] | 83.2% [79.9, 86.5] | **−57.2 pp** (CI entirely below zero) |

LLaVA stayed balanced across both conditions. BLIP-2 didn't just perform worse overall: it developed a strong directional bias. When the image *actually supported* the caption, BLIP-2 called it a contradiction most of the time. Both between-model differences are statistically significant (entailment: z = 30.29, p < .001; contradiction: z = 3.70, p < .001).

![Per-condition accuracy with 95% CI](results/fig1_per_condition_accuracy.png)

## Why this happened (and why I'm not overclaiming it)

The most plausible explanation: BLIP-2's Q-Former compresses 196 visual patches down to 32 query tokens before the language model ever sees them. Under uncertainty, a language model with less visual signal to lean on tends to fall back on its strongest prior, and in this task "contradiction" turned out to be that prior. That's a real, testable hypothesis, not a confirmed mechanism. The experiment didn't probe internal representations, and the two models also differ in backbone size (7B vs. 2.7B) and required different prompt formats. Both are documented confounds, not results dressed up as clean causation.

That distinction, what the data shows vs. what it merely suggests, is the actual point of this exercise. The most common evaluation error is quietly generalising a striking result past what the design can support.

## Why this matters beyond one dataset

The real risk isn't "BLIP-2 is worse." It's that **aggregate accuracy hides this completely.** BLIP-2's overall accuracy across both conditions was 54.6%. Reported alone, that reads as unremarkable, middling performance. Split by condition, it reveals a model that is systematically unreliable in one specific, predictable direction. Any evaluation pipeline that only reports a single top-line number would have missed this failure mode entirely. That's the case for condition-aware, stress-tested evaluation over single-number benchmarking, the same principle production AI evaluation and red-teaming work is built on.

## What this demonstrates

- Designing a controlled comparison that isolates a specific failure condition, rather than just benchmarking overall performance
- Statistical rigour: confidence intervals and significance testing, not eyeballed differences
- Prompt engineering across two architecturally different models that required different elicitation strategies to produce comparable output
- Running quantised model inference under real compute constraints (single T4 GPU, 4-bit)
- Documenting confounds and stating precisely what a result does and doesn't support

## Reproducing

**Notebook:** [`notebook/thesis.ipynb`](notebook/thesis.ipynb)

Everything runs end to end on a free-tier Colab T4 (16 GB VRAM). No API keys, no local GPU.

### Pipeline

1. **Setup:** install dependencies, mount Google Drive (used only to checkpoint results), and stream 1,000 stratified examples (500 entailment, 500 contradiction) from the SNLI-VE validation split (`sedrickkeh/snli-ve` on Hugging Face; the first 500 of each label in stream order, so the sample is deterministic).
2. **LLaVA-1.5-7B:** 4-bit NF4 quantisation via bitsandbytes; yes/no question prompt. Inference over all 1,000 examples, then the model is unloaded to free memory.
3. **Prompt ablation for BLIP-2:** question formats produced unparseable output, so a completion format (`"{hypothesis} The answer is"`) was selected; the ablation experiments are documented in the notebook.
4. **BLIP-2-OPT-2.7B:** rebuilt on the identical 1,000-example sample and run fp16 (it fits the T4 unquantised; a documented asymmetry with LLaVA). Scored on the same pairs.
5. **Analysis:** per-condition accuracy, accuracy gaps, 95% CIs, and the figure above (`results/fig1_per_condition_accuracy.png`).

### Quick view without re-running inference

The results are saved as CSVs in [`results/`](results/). In the notebook, running only the setup cells and the analysis section (`### 4`) loads these CSVs and regenerates the accuracy table and Figure 5.1 in seconds, with no model downloads. Full re-inference from scratch is supported but takes several hours of T4 time.

### Notes and caveats

- **Google Drive is required** for checkpointing: inference results are written to Drive every 100 examples, so an interrupted run resumes rather than restarts.
- **Prompt formats differ by design.** The two models required different elicitation to produce parseable output; this is documented as a confound in the thesis, not a bug in the notebook.
- **Quantisation asymmetry.** LLaVA runs 4-bit; BLIP-2 runs fp16. This is a documented limitation (thesis §4.7.5), not a controlled variable.

## Repository contents

```
├── README.md
├── notebook/snli_ve_fusion_eval.ipynb   # full pipeline (Colab, T4)
├── results/                             # raw predictions (CSV) + Figure 5.1
├── docs/thesis.pdf                      # full methodology, literature review, limitations
├── requirements.txt
└── LICENSE
```

## Full write-up

The complete methodology, literature review, confound discussion, and the analytical framework behind this experiment are in [`docs/thesis.pdf`](docs/thesis.pdf).
