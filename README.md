Here is the revised **Technical Report** in English, formatted in Markdown and updated to reflect the specific prompt implementation you provided.

---

# Technical Report: SemEval Task 5 - Plausibility Detection with Flan-T5

## 1. Introduction & Objective

This report documents the technical implementation for **SemEval Task 5**. The objective is to predict the semantic plausibility of short stories on a continuous scale (1-5). The proposed solution leverages a **Large Language Model (LLM) in Encoder-Only mode**, optimized via **Parameter-Efficient Fine-Tuning (PEFT/LoRA)** and a specialized **Hybrid Loss Function** designed to handle annotator uncertainty.

**Key Challenges Addressed:**

* **Computational Efficiency:** Training a 3-billion parameter model (Flan-T5 XL) on limited hardware using LoRA and memory-efficient optimizers.
* **Label Uncertainty:** Explicit handling of the standard deviation () provided in the dataset to weigh sample reliability during training.
* **Metric Optimization:** Implementation of *Inference Clipping* strategies to maximize "Accuracy within Standard Deviation".

---

## 2. Methodology and Data Preprocessing

### 2.1 Prompt Engineering Strategy

To effectively leverage the semantic understanding of Flan-T5, we formulated the regression task as a structured query. The prompt is designed to explicitly present all necessary context components—the full story, the target ambiguous word, the specific sense to evaluate, and a usage example.

The prompt instructs the model to act as a rater, providing a continuous score between 1 and 5. The implementation is defined as follows:



This structured format ensures the encoder's pooling mechanism captures the interaction between the *context story* and the *proposed meaning*.

### 2.2 Standard Deviation (StDev) Extraction

During preprocessing, in addition to the target labels, we extract the **standard deviation (`stdev`)** for each sample. This value is critical for our *Uncertainty Weighting* strategy (Section 4.1), allowing the model to distinguish between "high-consensus" samples (low variance) and "ambiguous" samples (high variance).

---

## 3. Model Architecture

### 3.1 Base Model: T5 Encoder-Only

We selected **`google/flan-t5-xl`** (approx. 3B parameters) configured as a **T5EncoderModel**.

* **Rationale:** Although T5 is a Seq2Seq model, text generation is unnecessary for a scalar regression task. By utilizing only the encoder, we reduce memory consumption by ~50% while retaining deep semantic understanding capabilities.

### 3.2 Custom Regression Head

A deep regression head (Deep MLP) was implemented on top of the encoder output:

1. **Masked Mean Pooling:** Calculates the average of the encoder's hidden states, excluding padding tokens to prevent signal dilution.
2. **MLP Structure:**
`Linear`  `LayerNorm`  `GELU`  `Dropout`  `Linear (64)`  `GELU`  `Linear (1)`.

### 3.3 PEFT: Low-Rank Adaptation (LoRA)

To enable training on GPUs with limited VRAM, we applied LoRA to the encoder:

* **Configuration:** Rank , Alpha , Dropout .
* **Target Modules:** `["q", "v", "k", "o"]` (All attention projection layers).
* **Impact:** Less than 1.2% of total parameters are trainable, drastically reducing memory requirements.

---

## 4. Training Strategy & Loss Engineering

We engineered a **Composite Loss Function** to balance absolute precision (MSE) with relative ranking (Contrastive).

### 4.1 Uncertainty-Weighted Smooth L1 Loss

Instead of standard MSE, we utilize a **SmoothL1Loss weighted by uncertainty**.



**Formula for Weights:**
$$w_i = e^{-\lambda \cdot \sigma_i}$$

**Formula for Weighted Loss:**
$$\mathcal{L}_{reg} = \frac{1}{N} \sum w_i \cdot \text{SmoothL1}(y_{pred}, y_{true})$$

*(Where $\lambda$ is the uncertainty scaling factor)*.

* **Rationale:** Samples with high standard deviation () indicate disagreement among human annotators. By penalizing these samples (low weight ), we prevent the model from overfitting to "noisy" data, forcing it to learn robust patterns from high-consensus samples.

### 4.2 Contrastive Regression Loss

A geometric contrastive loss term was added, contributing 20% to the total loss:

* **Objective:** Preserve relative distances in the embedding space. Stories with similar labels should have close embeddings; stories with divergent labels should have distant embeddings.

### 4.3 Optimization

* **Optimizer:** **Adafactor** (chosen for memory efficiency over AdamW).
* **Scheduler:** `ReduceLROnPlateau`, to dynamically adapt the learning rate when validation loss stalls.

---

## 5. Inference Optimization: Strategic Clipping

We mathematically analyzed the evaluation metric: **Accuracy within Standard Deviation** (prediction is correct if ).

To exploit the geometry of this metric, we apply **Strategic Clipping** to predictions during *inference* (post-training):

* **Natural Range:** `[1.0, 5.0]`
* **Clipped Range:** `[1.99, 4.01]`

**Mathematical Rationale:**

* Predicting **1.0** covers the real interval `[0.0, 2.0]`. Since labels  do not exist, half of the coverage area is wasted.
* Predicting **1.99** covers the interval `[0.99, 2.99]`.
* If Truth is **1.0**:   **CORRECT**.
* If Truth is **2.5**:   **CORRECT**.


* This technique "compresses" predictions toward the center, minimizing risk on edge cases and maximizing the valid coverage area of the metric.

---

## 6. Experimental Results

The combination of the Flan-T5 Encoder, Structured Prompting, Weighted Loss, and Strategic Clipping yielded the following results on the Dev Set:

| Metric | Performance | Analysis |
| --- | --- | --- |
| **Spearman Correlation** | **0.72412** | High correlation confirms the Contrastive Loss effectively taught the relative ordering of plausibility. |
| **Accuracy within Std** | **0.84516** | Competitive accuracy achieved via Weighted SmoothL1 combined with inference-time boundary optimization. |

---

## 7. Resources and Requirements

* **Base Model:** `google/flan-t5-xl`
* **Libraries:** `transformers`, `peft`, `datasets`, `torch`, `scikit-learn`.
* **Hardware:** Compatible with NVIDIA A100 (recommended) or T4 (with reduced batch size).