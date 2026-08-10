# Privacy-Preserving Personalization in Education
## A Federated Recommender System for Student Performance Prediction

Source code for the paper submitted to **Expert Systems with Applications (ESA)**.

**Authors:** Rodrigo Tertulino, Ricardo Almeida  
**Affiliation:** Federal Institute of Education, Science, and Technology of Rio Grande do Norte (IFRN) — LaPEA Lab  

---

## Repository Structure

```
.
├── data/
│   ├── interactions_real_rich_scaled_processed.csv  # pre-processed cohort (see Dataset below)
│   └── .gitkeep
├── outputs/                       # Generated metrics, models, and figures
│   └── .gitkeep
├── recommender_net.py             # RecommenderNet architecture (Table 3)
├── 01_data_preparation.py         # ASSISTments preprocessing pipeline (Section 4.2)
├── 02_centralized_baseline.py     # XGBoost centralized benchmark (Section 4.3.1)
├── 03_federated_training.py       # FL simulation: FedAvg + FedProx (Section 4.3.2)
├── 04_visualization.py            # Publication figures (Figures 4, 5, 6)
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the **ASSISTments Skill Builder Dataset**, publicly available at:

> https://doi.org/10.34740/kaggle/dsv/13081046

There are two ways to get data into `data/`, depending on whether you want to reproduce the preprocessing itself:

1. **From raw ASSISTments logs (full pipeline).** Download `skill_builder_data_corrected_collapsed.csv` from the link above, place it in `data/`, and run `01_data_preparation.py` — this reproduces the filtering, feature engineering, and scaling described in Section 4.2, and writes `data/processed_assistments.csv`.
2. **From the already-processed cohort (shortcut).** `data/interactions_real_rich_scaled_processed.csv`, included in this repo, is the exact post-processing output (1,365 students × 107 skills after filtering, features scaled to [0, 1]) — the same file backing the [`student-performance-for-recommender-systems`](https://www.kaggle.com/datasets/rodrigotertulino/student-performance-for-recommender-systems) Kaggle dataset and the [reproducibility notebook](https://www.kaggle.com/code/rodrigotertulino/federated-recommender-reproducibility). It has `target_correct_rate` but not yet the binarized `target` column `02_centralized_baseline.py`/`03_federated_training.py` expect — derive it with:
   ```python
   df['target'] = (df['target_correct_rate'] >= 0.70).astype(int)
   ```

---

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10/3.11. GPU (CUDA) support is optional for training.

> **Note:** `requirements.txt` pins `flwr==1.7.0` and `ray==2.6.3` for federated simulation. `ray==2.6.3` has no published wheel for Python 3.12 — if you're on 3.12 (e.g. Kaggle's default image), install unpinned `flwr`/`ray` instead (`pip install flwr ray`); `flwr.simulation.start_simulation` remains available in newer releases. See the [reproducibility notebook](https://www.kaggle.com/code/rodrigotertulino/federated-recommender-reproducibility) for a working Python 3.12 setup.

---

## Reproducing the Experiments

Run the scripts in order:

```bash
# 1. Preprocess the ASSISTments dataset
python 01_data_preparation.py

# 2. Train and evaluate the centralized XGBoost baseline (Table 4)
python 02_centralized_baseline.py

# 3. Run all federated experiments (Table 6)
#    -- This trains FedAvg + FedProx (mu=0.1, 0.5, 1.0) for 100 rounds each
python 03_federated_training.py --all

#    -- Or run a single configuration
python 03_federated_training.py --mu 0.5

# 4. Generate all publication figures
python 04_visualization.py
```

---

## Architecture Summary (Table 3)

| Layer | Type | Input Shape | Output Shape | Activation |
|---|---|---|---|---|
| 1 | Embedding (User ID) | (batch, 1) | (batch, 10) | — |
| 2 | Embedding (Skill ID) | (batch, 1) | (batch, 10) | — |
| 3 | Input (Engineered) | (batch, 3) | (batch, 3) | — |
| 4 | Concatenation | (batch, 10+10+3) | (batch, 23) | — |
| 5 | Dense (Hidden 1) | (batch, 23) | (batch, 32) | ReLU |
| 6 | Dense (Hidden 2) | (batch, 32) | (batch, 16) | ReLU |
| 7 | Dense (Output) | (batch, 16) | (batch, 1) | Sigmoid |

---

## Key Hyperparameters (Table 2)

| Parameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999, ε=1e-8) |
| Learning rate | 1×10⁻³ |
| Loss function | Binary Cross-Entropy |
| Local epochs | 5 |
| Local batch size | 32 |
| Communication rounds | 100 |
| Clients per round (train) | 10% (≈136) |
| Clients per round (eval) | 20% (≈273) |
| min_fit_clients | 50 |
| FedProx μ (grid search) | {0.1, 0.5, 1.0} |
| Random seed | 42 |

---

## Expected Results (Table 6)

| Strategy | Best F1 | Best Round | Mean F1 | Std Dev |
|---|---|---|---|---|
| FedAvg | 0.7584 | 70 | 0.7249 | 0.0249 |
| FedProx μ=0.1 | 0.7526 | 89 | 0.7226 | 0.0242 |
| **FedProx μ=0.5** | **0.7628** | **88** | 0.7238 | 0.0205 |
| FedProx μ=1.0 | 0.7555 | 80 | 0.7280 | 0.0152 |

Centralized XGBoost baseline: **F1 = 0.8285** (round 24).

---

## Independent Reproducibility Check

This pipeline was independently re-run end-to-end (unmodified code, `01`→`04`) against the [`student-performance-for-recommender-systems`](https://www.kaggle.com/datasets/rodrigotertulino/student-performance-for-recommender-systems) dataset — 100 communication rounds × 4 strategies, ~10h15min on a 10-core machine. A self-contained notebook with the live pipeline and the full results below is published on Kaggle:

**→ [rodrigotertulino/federated-recommender-reproducibility](https://www.kaggle.com/code/rodrigotertulino/federated-recommender-reproducibility)** (Kaggle notebook)
**→ [reports/reproducibility_report.html](reports/reproducibility_report.html)** ([rendered view](https://htmlpreview.github.io/?https://github.com/rodrigoronner/federated-recommender-education/blob/main/reports/reproducibility_report.html)) — visual summary of the comparison below

**Centralized baseline** — reproduced within 0.1–0.3pp of Table 4 on every metric (F1 0.8274 vs. 0.8285, round 20 vs. 24).

**Federated results** — Table 6 comparison:

| Strategy | Best F1 (paper) | Best F1 (repro) | Mean F1 (paper) | Mean F1 (repro) | Std Dev (paper) | Std Dev (repro) |
|---|---|---|---|---|---|---|
| FedAvg | 0.7584 | 0.7710 | 0.7249 | 0.7279 | 0.0249 | 0.0150 |
| FedProx μ=0.1 | 0.7526 | 0.7638 | 0.7226 | 0.7275 | 0.0242 | 0.0161 |
| **FedProx μ=0.5** | **0.7628** | **0.7737** | 0.7238 | 0.7291 | 0.0205 | 0.0162 |
| FedProx μ=1.0 | 0.7555 | 0.7581 | 0.7280 | 0.7280 | **0.0152** | **0.0129** |

The paper's central claims replicate — FedProx μ=0.5 gives the best peak F1, and μ=1.0 gives the most stable training — while absolute F1 values ran ~1–1.5pp higher and the ranking of *least*-stable strategy did not reproduce (FedAvg was among the most stable here, not the least). This is consistent with run-to-run variance inherent to Flower/Ray's client-sampling and actor-scheduling order, which a fixed `torch.manual_seed` does not fully pin down across processes.

Two environment gaps surfaced during reproduction and are reflected in this repo: `ray` was missing from `requirements.txt` (added), and the pinned `flwr`/`ray` versions don't install on Python 3.12 (documented above).

---

## License

This code is released under the **Apache 2.0 License**.

---

## Citation

If you use this code, please cite the arXiv version of the paper:

```
Tertulino, R., & Almeida, R. (2025). Privacy-preserving personalization in education: A federated recommender
system for student performance prediction. arXiv:2509.10516v3 [cs.LG]. https://doi.org/10.48550/arXiv.2509.10516
```
