# CT Report Anomaly Detection: High-Performance Text Classifier (Aliyun Tianchi Top 5 Solution)

This repository contains the advanced deep learning solution developed for the **Aliyun Tianchi Competition: NLP Series Learning Race - CT Hunter: Medical Image Report Anomaly Detection**.

Our final optimized model, built on an enhanced Transformer architecture, secured a Mean AUC of $\mathbf{0.90}$ on the private leaderboard, achieving a **Top 5** rank (5th out of 399 teams) in the first season of the competition.

---

## 🏆 Competition Performance Summary

| Metric | Score Achieved (Optimized Model) | Competition Rank (Season 1) |
| :--- | :--- | :--- |
| **Mean AUC** | **0.90** | **5 / 399** |

---

## 💡 Key Architectural & Training Enhancements

The substantial jump in performance from a baseline solution was driven by targeted optimizations, primarily focused on handling the **multi-label classification** and **severe class imbalance** inherent in anomaly detection.

### 1. Advanced Model Architectures

The codebase provides two state-of-the-art sequence models, with the Transformer being the core of the high-ranking solution:

| Model | Class | Technical Improvements |
| :--- | :--- | :--- |
| **Improved Transformer** | `ImprovedTransformerClassifier` | Implemented modern best practices: **Pre-Normalization** (`norm_first=True`) for stable training, **GELU** activation, learnable **Positional Encoding**, and a multi-layer classification head. |
| **Attention-BiLSTM** | `ImprovedCTClassifier` | A highly robust recurrent network featuring an optional **Multi-Head Attention** mechanism to focus on relevant tokens, **BatchNorm1d** after sequence pooling, and a multi-layer classification head. |

### 2. Robust Training Strategy

* **Loss Function:** Utilized **Focal Loss** (`FocalLoss`). This is critical for multi-label tasks with high imbalance, as it down-weights easily classified samples, forcing the model to concentrate on the **hard-to-classify anomaly (positive) samples**.
* **Optimization & Scheduling:** Employed the **AdamW** optimizer for effective weight decay. Optionally supports **ReduceLROnPlateau** scheduler to dynamically adjust the learning rate based on validation AUC.
* **Stability & Regularization:**
    * Implemented **Gradient Clipping** (`max_norm=1.0`) to prevent exploding gradients.
    * Incorporated **Early Stopping** based on validation AUC (`patience`), a key defense against overfitting.
* **Comprehensive Metrics:** Validation tracks Mean AUC, **Mean F1 Score**, and **Average Precision (AP)** for holistic performance monitoring during training.

---

## 🚀 Getting Started

### Prerequisites

You'll need a standard deep learning environment setup:

```bash
pip install torch numpy pandas scikit-learn tqdm matplotlib
```

## 🖥️ Usage

The script is executed using command-line arguments to configure the entire pipeline.

### Recommended Command (Transformer with Focal Loss)

This configuration reflects the key settings that led to the $\mathbf{0.90}$ AUC.

```bash
python CT_Hunter_promoted.py \
    --train combined_train_data.csv \
    --test track1_round1_testB.csv \
    --model transformer \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.00139 \
    --embedding_dim 256 \
    --num_layers 3 \
    --nhead 8 \
    --dim_feedforward 512 \
    --loss_type focal \
    --patience 6 \
    --max_length 150 \
    --use_scheduler False \
    --output predictions.csv
```

### Key Hyperparameters

| Argument | Default Value | Description |
| :--- | :--- | :--- |
| `--model` | `lstm` | Select model type: `lstm` or `transformer`. |
| `--loss_type` | `focal` | Choose loss function: `bce` or `focal`. **Focal Loss is strongly recommended.** |
| `--lr` | `0.00139` | Initial learning rate. |
| `--embedding_dim` | `256` | Dimensionality of word embeddings (and Transformer's $d_{model}$). |
| `--num_layers` | `3` | Number of LSTM or Transformer Encoder layers. |
| `--patience` | `6` | Early stopping patience (epochs without validation AUC improvement). |
| `--max_length` | `150` | Maximum sequence length for padding/truncation. |
| `--use_attention` | `True` | Only for `lstm` model: whether to use Multi-Head Attention. |
| `--use_scheduler` | `False` | Whether to use the `ReduceLROnPlateau` scheduler. |

---

## 📂 Code Modules Overview

| Module/Function | Description |
| :--- | :--- |
| `ImprovedTransformerClassifier` | Defines the optimized, **Pre-Norm based Transformer** model. |
| `ImprovedCTClassifier` | Defines the enhanced **Bi-LSTM** model with optional Attention and BatchNorm. |
| `FocalLoss` | Custom implementation of **Focal Loss** for imbalanced classification. |
| `load_data()` | Handles robust loading of the custom-formatted text data, token parsing, and label vectorization. |
| `collate_fn()` | Dynamic batching function responsible for token padding and generating the crucial **attention masks**. |
| `train_model()` | The core training loop, which incorporates logging, **Early Stopping**, **gradient clipping**, model saving, and visualization of training curves. |
| `compute_metrics()` | Calculates and reports multi-label metrics: Mean AUC, F1 Score, and Average Precision. |