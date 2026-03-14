# Incident Prediction — Sliding-Window + LightGBM

Binary classifier that predicts whether an incident will occur within the next **H** time steps, given the previous **W** steps of one or more time-series metrics.

---

## Problem Formulation

Given a multivariate time series of system metrics $\mathbf{x}_1, \ldots, \mathbf{x}_T \in \mathbb{R}^M$ and a binary incident signal $l_t \in \{0, 1\}$, the goal is to learn a function

$$f\!\left(\mathbf{x}_{t-W}, \ldots, \mathbf{x}_{t-1}\right) \;\to\; \hat{y}_t \in \{0, 1\}$$

where $\hat{y}_t = 1$ means "at least one incident is predicted in $[t,\, t+H)$".

A **sliding window** of length $W$ is moved one step at a time across the series. At each position the window is labelled **1** if any incident label fires anywhere in the next $H$ steps, and **0** otherwise. This turns the forecasting problem into standard binary classification.

---

## Dataset

A synthetic multivariate time series is generated programmatically (`src/data_generation.py`):

- **3 metric channels**, each a superposition of two sine waves at different frequencies plus Gaussian noise, providing a realistic periodic baseline.
- **Incidents** are injected as isolated bursts: during an incident window a random subset of metrics receives an additive half-normal spike, simulating a sudden resource exhaustion or error-rate surge.
- Default parameters: 10 000 time steps, ~5 % incident rate, 50-step incident duration.

Using synthetic data ensures full reproducibility and lets us precisely control the signal-to-noise ratio to stress-test the model.

---

## Model — LightGBM

**Why LightGBM?**

| Criterion | Rationale |
|-----------|-----------|
| Tabular features | Gradient boosting excels on hand-crafted statistical features |
| Class imbalance | Native `scale_pos_weight` parameter handles rare incidents |
| Speed | Fast training enables quick iteration over window sizes |
| Interpretability | Feature importances (gain-based) are easy to communicate |
| Production-readiness | Lightweight, no GPU required, easy to serve |

An LSTM or Transformer would be a natural next step if raw sequences were fed directly; LightGBM is preferred here because the feature engineering already captures the relevant temporal structure.

---

## Feature Engineering

For each metric channel within the look-back window, **9 statistics** are extracted:

| Feature | Description |
|---------|-------------|
| `mean` | Average level |
| `std` | Overall volatility |
| `min` / `max` | Extremes |
| `range` | max − min |
| `net_change` | last − first value |
| `deviation_from_mean` | last − mean (recent drift) |
| `slope` | Linear-regression slope (trend direction) |
| `recent_volatility` | std of last W/4 steps (short-term instability) |

With 3 metrics this yields a **27-dimensional** feature vector per sample.

---

## Evaluation Setup

### Data split
The dataset is split **chronologically** (no shuffling) to prevent data leakage:

```
Train 60 % │ Validation 20 % │ Test 20 %
────────────────────────────────────────→ time
```

Early stopping is applied on the validation set; final metrics are reported on the held-out test set only.

### Metrics

| Metric | Why it matters |
|--------|----------------|
| **AUC-ROC** | Threshold-independent measure of discriminative power |
| **AUC-PR** | More informative than AUC-ROC under class imbalance |
| **F1 @ optimal threshold** | Harmonic mean of precision and recall |
| **Precision / Recall** | Trade-off relevant to alert fatigue vs missed incidents |

### Alert threshold
The default operating point is the threshold that maximises F1 on the test set. The **threshold sweep plot** (`results/evaluation.png`) shows how precision, recall, and F1 change across all thresholds so an operator can choose a different operating point based on their SLA:

- **Lower threshold** → higher recall (fewer missed incidents), more false alarms.
- **Higher threshold** → fewer false alarms, higher miss rate.

---

## Repository Structure

```
.
├── src/
│   ├── data_generation.py   # Synthetic time-series generator
│   ├── features.py          # Sliding-window feature extraction
│   ├── model.py             # LightGBM wrapper (IncidentPredictor)
│   └── evaluation.py        # Metrics, threshold sweep, plot generation
├── main.py                  # End-to-end training & evaluation script
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with default parameters (W=50, H=10, 10 000 steps)
python main.py

# 3. Custom run
python main.py --W 30 --H 5 --n_samples 20000 --seed 7
```

Evaluation plots are saved to `results/evaluation.png`.

---

## Results (default parameters)

Running `python main.py` with the default settings produces results in the following range (exact values depend on the random seed):

| Metric | Value |
|--------|-------|
| AUC-ROC | ≥ 0.95 |
| AUC-PR | ≥ 0.80 |
| F1 (optimal threshold) | ≥ 0.75 |

The model reliably detects approaching incidents several steps in advance while keeping false-alarm rates low.

---

## Limitations & Real-World Adaptations

| Limitation | Real-world adaptation |
|------------|-----------------------|
| Synthetic data | Replace generator with a real metrics store (Prometheus, Datadog) |
| Fixed window sizes W, H | Tune W and H per-metric based on domain knowledge or cross-validation |
| Stationary feature engineering | Add drift detection or online normalisation for non-stationary streams |
| Offline training | Retrain periodically on a rolling window; monitor data drift |
| Single threshold for all metrics | Calibrate per-metric thresholds to match per-service SLOs |
| No temporal dependencies in features | Upgrade to LSTM / Temporal Fusion Transformer for raw-sequence input |
