# Privacy-Preserving Personalization in Education
## A Federated Recommender System for Student Performance Prediction

Source code for the paper submitted to the **Journal of Educational Data Mining (JEDM)**.

**Authors:** Rodrigo Tertulino, Ricardo Almeida  
**Affiliation:** Federal Institute of Education, Science, and Technology of Rio Grande do Norte (IFRN) — LaPEA Lab  

---

## Repository Structure

```
.
├── data/                          # Raw and processed datasets (not included)
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

Download and place `skill_builder_data_corrected_collapsed.csv` inside the `data/` folder before running any script.

---

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10. GPU (CUDA) support is optional for training.

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

## License

This code is released under the **Apache 2.0 License**.

---

## Citation

If you use this code, please cite:

```
Tertulino, R., & Almeida, R. (2025). Privacy-preserving personalization in education: A federated recommender system for student performance prediction. arXiv preprint arXiv:2509.10516.
```
