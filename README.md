# Pneumonia Detection Using Diffusion Models (DDPM & DDIM)
### Unsupervised Anomaly Detection on Chest X-Rays

> Numerical Methods Course Project — March 2026

---

## Overview

This project applies **Denoising Diffusion Probabilistic Models (DDPM)** and **Denoising Diffusion Implicit Models (DDIM)** to detect pneumonia in chest X-rays — without ever showing the model a single sick lung during training.

The core idea: train a generative model exclusively on **healthy lungs** so it learns what a healthy lung should look like. When a sick lung is passed through it, the model forces it back toward healthy tissue. The **pixel-wise difference** between the original and the reconstructed image becomes the anomaly map highlighting the infection.

This is **unsupervised anomaly detection** — no pneumonia labels are needed during training.

---

## How It Works

```
Healthy lungs only
       ↓
  Train U-Net DDPM
       ↓
  Given a test image (healthy or sick):
       ↓
  Add noise up to t_start  →  Heal back to t=0
       ↓
  |original − healed| = Anomaly Map
```

---

## Dataset

**PneumoniaMNIST** from [MedMNIST v2](https://medmnist.com/)

| Split | Normal | Pneumonia | Total |
|---|---|---|---|
| Train | 1,214 | — | 1,214 (healthy only) |
| Test | 234 | 390 | 624 |

- Resolution: `128 × 128` (upsampled from default 28×28)
- Normalization: pixel values scaled to `[-1, 1]`
- Dataset is auto-downloaded on first run

---

## Methods

### Forward Diffusion (Adding Noise)
Gradually corrupts a clean image $x_0$ over $T = 1000$ steps using a linear noise schedule. The closed-form formula allows jumping directly to any timestep:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### U-Net (Noise Prediction Network)
Takes a noisy image $x_t$ and timestep $t$, and predicts the added noise:

$$\hat{\epsilon} = \epsilon_\theta(x_t, t)$$

Trained by minimizing MSE loss:

$$\mathcal{L} = \mathbb{E}\left[\left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2\right]$$

**Architecture:** ~1M parameters | Sinusoidal time embeddings | GroupNorm + SiLU | Skip connections

### DDPM Reverse (Stochastic)
Iteratively denoises over all 1000 steps:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta\right) + \sigma_t z$$

### DDIM Reverse (Deterministic)
Uses a small subset of timesteps ($S \ll T$) for fast, deterministic reconstruction:

$$x_{t_{\text{prev}}} = \sqrt{\bar{\alpha}_{t_{\text{prev}}}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t_{\text{prev}}}}\,\epsilon_\theta$$

Setting `eta=0` makes sampling fully deterministic — ~10× faster than DDPM.

### Anomaly Scoring
Two strategies are compared on the reconstruction residual:

| Strategy | Description |
|---|---|
| **Mean** | Average pixel-wise error across the whole image |
| **Top-K** | Mean error of the top `k%` highest-error pixels (0.5%, 1%, 2%, 5%) |

---

## Results

### DDPM

| Method | AUROC | AUPRC |
|---|---|---|
| MEAN | 0.7800 | 0.8370 |
| TOP-0.50% | 0.8200 | 0.8990 |
| TOP-1.00% | 0.8300 | 0.9010 |
| TOP-2.00% | 0.8400 | **0.9010** |
| TOP-5.00% | 0.8300 | 0.8990 |

### DDIM

| Method | AUROC | AUPRC |
|---|---|---|
| **MEAN** | **0.8481** | **0.9000** |
| TOPK-0.5% | 0.8312 | 0.8670 |
| TOPK-1% | 0.8389 | 0.8770 |
| TOPK-2% | 0.8421 | 0.8850 |
| TOPK-5% | 0.8455 | 0.8940 |

**@ 95% Specificity (DDIM MEAN — best overall):**

| Metric | Value |
|---|---|
| Sensitivity | 0.531 |
| Specificity | 0.953 |
| Precision | 0.950 |
| F1 Score | 0.681 |
| Accuracy | 0.689 |
| Random baseline (AUPRC) | 0.625 |

> DDIM outperforms DDPM across all metrics while being significantly faster. Both methods sit well above the random baseline of 0.62.

---

## Project Structure

```
your-repo/
├── pneumonia_diffusion_anomaly_detection_ddpm_ddim_V2_1.ipynb  ← notebook here
├── weights/
│   └── ddpm_pneumonia_model_128x128.pth
├── cache/
│   ├── cached_scores_ddpm_t400_multi_topk_updated.npz
│   └── cached_scores_ddim_t400_s100_eta0_multi_topk_updated.npz
└── README.md
---

## Notebook Structure

| Cell | Section | Description |
|---|---|---|
| 1 | Project Overview | Full methodology, equations, and design decisions |
| 2–3 | Imports | All required libraries |
| 4–5 | Data Loading | Download, filter, normalize, visualize |
| 6–7 | U-Net Architecture | Sinusoidal embeddings, blocks, full model |
| 8–9 | DDPM Training | Noise schedule, training loop, loss curve |
| 10–11 | Load Pre-trained | Skip training by loading saved weights |
| 12–13 | DDPM Sampling | Generate X-rays from pure noise |
| 14–15 | DDPM Anomaly Detection | Single image visual demo |
| 16–17 | DDPM Score Caching | Full test set scoring, saved to `.npz` |
| 18–19 | DDPM Evaluation | ROC, PR curves, confusion matrix, metrics |
| 20–21 | DDIM Sampling | Fast deterministic generation |
| 22–23 | DDIM Reconstruction | Heal-and-compare pipeline |
| 24–25 | DDIM Evaluation | Full evaluation mirroring DDPM |

---

## Requirements

```
torch
torchvision
medmnist
numpy
matplotlib
seaborn
scikit-learn
tqdm
```

Install with:
```bash
pip install torch torchvision medmnist numpy matplotlib seaborn scikit-learn tqdm
```

---

## Running the Notebook

**To train from scratch:**
1. Run all cells top to bottom
2. Training takes ~30–60 minutes on CPU at `128×128`

**Pre-trained weights and cached scores are included in the repo.**
To reproduce the exact reported results without training:
1. Run cells 2–7 (imports, data, model definition)
2. Run cell 10–11 (load pre-trained weights)
3. Skip directly to cell 18–19 for DDPM evaluation
4. Skip directly to cell 24–25 for DDIM evaluation

**Hardware:** Auto-detects Apple Silicon (MPS), CUDA GPU, or falls back to CPU.

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Train on healthy images only | Model has no knowledge of pneumonia — forced to reconstruct healthy tissue |
| Closed-form noising | Jump directly to any $x_t$ without 1000 sequential steps |
| `t_start = 400` for evaluation | Balances information destruction vs. hallucination risk |
| Score caching to `.npz` | Avoids re-running slow reverse diffusion for every evaluation |
| Top-K scoring | Pneumonia appears as localized spots, not uniform image-wide noise |
| `eta=0` for DDIM | Fully deterministic reconstruction gives consistent anomaly maps |

---

## Limitations

- **Sensitivity is moderate (0.53 at best)** — the model misses ~47% of pneumonia cases at 95% specificity. This is the inherent trade-off of unsupervised detection.
- Supervised models on the same dataset reach sensitivity of 0.85–0.95, but require labelled anomaly data during training.
- Results are sensitive to `t_start` — tuning this further may improve performance.
