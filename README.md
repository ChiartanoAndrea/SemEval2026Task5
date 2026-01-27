# SemEval Task 5 - Plausibility Detection (Flan-T5 Encoder Strategy)

## 1. Introduction & Objective

This report documents the technical implementation for **SemEval Task 5**. The objective is to predict the plausibility of short stories on a continuous scale (1-5). Our solution leverages a **Large Language Model (LLM) in Encoder-Only mode**, utilizing **Parameter-Efficient Fine-Tuning (PEFT/LoRA)** and a specialized **Hybrid Loss Function** designed to handle annotator uncertainty.

**Key constraints addressed:**
- **Computational Efficiency:** Training a 3B parameter model (Flan-T5 XL) on consumer/cloud hardware (e.g., T4/A100) using LoRA and Adafactor.
- **Label Uncertainty:** Explicit handling of the standard deviation ($\sigma$) provided in the dataset to weigh sample reliability.
- **Metric Optimization:** Geometric inference strategies to maximize "Accuracy within Standard Deviation".

## 2. Data Preparation and Prompt Engineering

### 2.1 Prompt Template
Since we utilize an instruction-tuned model (Flan-T5), we formulated the regression problem as a text comprehension task. The input is formatted with a specific prompt that includes the context, the target sentence, and lexical definitions to ground the semantics:

```text
Rate how plausible the meaning is in the context.
Answer ONLY with a number between 1 and 5.
You may use decimals (e.g., 2.543, 4.032).

Story:
{precontext}
{sentence}
{ending}

Target word: {homonym}
Sense: {judged_meaning}
Example: {example_sentence}

Answer:
```


## 2.2 Standard Deviation (StDev) Extraction

During preprocessing, we do not only load the target labels but also extract the **standard deviation (`stdev`)** for each sample. This is critical for our *Uncertainty Weighting* strategy (see Section 4.1), allowing the model to distinguish between "consensus" samples (low variance) and "ambiguous" samples (high variance).

## 3. Model Architecture

### 3.1 Base Model: T5 Encoder-Only
We selected **`google/flan-t5-xl`** (approx. 3B parameters) but deployed it in a **T5EncoderModel** configuration.

- **Rationale:** Although T5 is a Seq2Seq (Encoder-Decoder) model, text generation is unnecessary for a scalar regression task. By discarding the decoder and using only the encoder to extract hidden states, we reduce memory footprint and computational cost by approximately 50% while retaining the model's deep semantic understanding capabilities.

### 3.2 Custom Regression Head
We implemented a custom, deep regression head on top of the encoder output, replacing the standard classification head.

- **Pooling Strategy:** We perform **Masked Mean Pooling** on the encoder's last hidden state. This ensures that padding tokens do not dilute the semantic representation of the sentence.
- **Head Structure:** A deep Multi-Layer Perceptron (MLP):
  `Linear` -> `LayerNorm` -> `GELU` -> `Dropout` -> `Linear (64)` -> `GELU` -> `Linear (1)`.

### 3.3 PEFT: Low-Rank Adaptation (LoRA)
To enable training on limited VRAM, we apply LoRA to the encoder:

- **Configuration:** Rank $r=8$, Alpha $\\alpha=16$, Dropout $0.1$.
- **Target Modules:** `["q", "v", "k", "o"]` (All attention projection layers).
- **Impact:** This makes less than 1% of the parameters trainable (approx. 5.3M params), drastically reducing memory requirements while maintaining performance.

## 4. Training Strategy & Loss Engineering

We engineered a **Composite Loss Function** to balance absolute precision (MSE) with relative ranking (Contrastive).

### 4.1 Uncertainty-Weighted Smooth L1 Loss
Instead of standard MSE, we utilize **SmoothL1Loss weighted by sample uncertainty**.

**Formula:**
$$ w_i = e^{-\\lambda \\cdot \\sigma_i} $$
$$ \\mathcal{L}_{reg} = \\frac{1}{N} \\sum w_i \\cdot \\text{SmoothL1}(y_{pred}, y_{true}) $$
*(Where $\\lambda = 2.0$ is the uncertainty scaling factor)*.

- **Reasoning:** Samples with high standard deviation ($\\sigma$) represent annotator disagreement. By penalizing these samples (lower $w_i$), we prevent the model from overfitting to "noisy" labels, forcing it to learn robust patterns from high-consensus data.

### 4.2 Contrastive Regression Loss
We added a geometric contrastive loss term:

- **Objective:** To preserve relative distances in the embedding space. If two stories have similar labels (e.g., 4.0 and 4.1), their embeddings should be close. If they are divergent (e.g., 1.0 and 5.0), embeddings should be distant.
- **Weighting:** This component contributes 20% (`cont_weight=0.2`) to the total loss.

### 4.3 Optimization
- **Optimizer:** **Adafactor**. Chosen over AdamW for its memory efficiency (it does not store second-moment estimates), which is crucial for loading the XL model.
- **Scheduler:** `ReduceLROnPlateau`, allowing dynamic learning rate adjustment when validation loss stalls.

## 5. Inference Optimization (Metric Exploitation)

We mathematically analyzed the evaluation metric: **Accuracy within Standard Deviation** (prediction is correct if $|pred - label| \\le 1.0$).

### 5.1 Strategic Clipping
During inference (not training), we apply aggressive clipping bounds of **`[1.99, 4.01]`** instead of the natural `[1.0, 5.0]`.

**Mathematical Rationale:**
- Predicting **1.0** covers the real interval `[0.0, 2.0]`. Since labels $< 1.0$ do not exist, half of the coverage is wasted.
- Predicting **1.99** covers the interval `[0.99, 2.99]`.
    - If Truth is **1.0**: $|1.99 - 1.0| = 0.99 < 1.0$ -> **CORRECT**.
    - If Truth is **2.5**: $|1.99 - 2.5| = 0.51 < 1.0$ -> **CORRECT**.
- This technique "compresses" predictions toward the center, minimizing risk on edge cases while maximizing the metric's valid geometric coverage area.

## 6. Experimental Results

The combination of Flan-T5 Encoder, Uncertainty-Weighted Loss, and Strategic Clipping yielded:

| Metric | Performance | Analysis |
| :--- | :--- | :--- |
| **Spearman Correlation** | **~0.72** | High rank correlation confirms that the Contrastive Loss effectively taught the relative ordering of story plausibility. |
| **Accuracy within Std** | **~0.79** | Competitive accuracy achieved via Weighted SmoothL1 combined with inference-time boundary optimization. |

## 7. Resources

- **Base Model:** `google/flan-t5-xl`
- **Libraries:** `transformers`, `peft`, `datasets`, `torch`, `scikit-learn`.
- **Hardware:** Compatible with NVIDIA T4 (low batch size) and A100.


