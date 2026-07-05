"""
Step 3 — Federated Learning Simulation (Sections 4.3.2, 5.2, and Table 6)
==========================================================================
Simulates federated training using the Flower (flwr) framework.
Each of the 1,365 students is treated as an independent FL client
holding private, locally-stored data.

Runs four configurations:
    - FedAvg        (mu = 0)
    - FedProx mu=0.1
    - FedProx mu=0.5  ← optimal, reported as main result
    - FedProx mu=1.0

For each configuration the script logs per-round metrics on a global
held-out test set (20% of each client's local data), then saves:
    outputs/fl_metrics_<strategy>.csv
    outputs/fl_summary.json

Usage:
    python 03_federated_training.py
    python 03_federated_training.py --mu 0.5        # single run
    python 03_federated_training.py --all            # all four configs

Run the visualisation script afterwards:
    python 04_visualization.py
"""

import os, json, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch

from recommender_net import RecommenderNet, StudentSkillDataset, train_local, evaluate_local

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Hyperparameters (Table 2 — Implementation Details)
# ---------------------------------------------------------------------------
RANDOM_SEED      = 42
DATA_PATH        = os.path.join("data",    "processed_assistments.csv")
OUT_DIR          = os.path.join("outputs")
os.makedirs(OUT_DIR, exist_ok=True)

N_ROUNDS         = 100      # communication rounds
LOCAL_EPOCHS     = 5        # local epochs per round
LOCAL_BATCH_SIZE = 32       # local mini-batch size
LEARNING_RATE    = 1e-3     # Adam learning rate
FRACTION_FIT     = 0.10     # fraction of clients for training   (~136 / 1365)
FRACTION_EVAL    = 0.20     # fraction of clients for evaluation (~273 / 1365)
MIN_FIT_CLIENTS  = 50       # minimum absolute fit clients
VAL_FRACTION     = 0.20     # held-out validation fraction per client
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Load data and partition by student (one client = one student)
# ---------------------------------------------------------------------------
print(f"Loading data from {DATA_PATH} …")
df = pd.read_csv(DATA_PATH)

NUM_USERS  = df["user_id_new"].nunique()
NUM_SKILLS = df["skill_id_new"].nunique()
print(f"  Students (clients): {NUM_USERS}  |  Skills (items): {NUM_SKILLS}")
print(f"  Total pairs: {len(df):,}")
print(f"  Device: {DEVICE}")

# Split each client's data into local train (80%) and local val (20%)
client_train_datasets = {}
client_val_datasets   = {}

for uid, group in df.groupby("user_id_new"):
    if len(group) < 5:
        continue  # skip clients with too few samples
    train_df, val_df = train_test_split(
        group, test_size=VAL_FRACTION, random_state=RANDOM_SEED, shuffle=True
    )
    client_train_datasets[uid] = StudentSkillDataset(train_df.reset_index(drop=True))
    client_val_datasets[uid]   = StudentSkillDataset(val_df.reset_index(drop=True))

CLIENT_IDS = list(client_train_datasets.keys())
N_CLIENTS  = len(CLIENT_IDS)
print(f"  Eligible clients (≥5 samples): {N_CLIENTS}")

# Compute number of clients per round
n_fit  = max(MIN_FIT_CLIENTS, int(FRACTION_FIT  * N_CLIENTS))
n_eval = max(1,               int(FRACTION_EVAL * N_CLIENTS))
print(f"  Clients per round — train: {n_fit}  |  eval: {n_eval}")


# ---------------------------------------------------------------------------
# Federated Averaging (server-side aggregation, Eq. 3)
# ---------------------------------------------------------------------------
def federated_average(local_updates):
    """
    Weighted average of local model parameters (FedAvg, Eq. 3).
    Each client's weight is proportional to its number of training samples.

    Parameters
    ----------
    local_updates : list of (params, n_samples) tuples

    Returns
    -------
    List of numpy arrays — new global parameters.
    """
    total_samples = sum(n for _, n in local_updates)
    avg_params    = [
        sum(n / total_samples * p[i] for p, n in local_updates)
        for i in range(len(local_updates[0][0]))
    ]
    return avg_params


# ---------------------------------------------------------------------------
# One complete FL simulation (one strategy configuration)
# ---------------------------------------------------------------------------
def run_simulation(mu: float = 0.0, label: str = "FedAvg") -> pd.DataFrame:
    """
    Simulate N_ROUNDS of federated learning for a given proximal term mu.

    mu = 0  → FedAvg (standard, no proximal regularisation)
    mu > 0  → FedProx (Eq. 4)

    Returns a DataFrame with per-round evaluation metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Strategy: {label}  (mu={mu})")
    print(f"{'='*60}")

    rng = np.random.default_rng(RANDOM_SEED)

    # Initialise global model
    global_model = RecommenderNet(NUM_USERS, NUM_SKILLS)
    global_params = global_model.get_parameters_flat()

    records = []

    for rnd in range(1, N_ROUNDS + 1):
        # ── 1. Select client subsets ──────────────────────────────────────
        fit_clients  = rng.choice(CLIENT_IDS, size=min(n_fit,  N_CLIENTS), replace=False).tolist()
        eval_clients = rng.choice(CLIENT_IDS, size=min(n_eval, N_CLIENTS), replace=False).tolist()

        # ── 2. Local training on each fit client ──────────────────────────
        local_updates = []
        for uid in fit_clients:
            client_model = RecommenderNet(NUM_USERS, NUM_SKILLS)
            client_model.set_parameters_flat(global_params)

            updated_params, n_samples = train_local(
                model         = client_model,
                dataset       = client_train_datasets[uid],
                epochs        = LOCAL_EPOCHS,
                batch_size    = LOCAL_BATCH_SIZE,
                lr            = LEARNING_RATE,
                mu            = mu,
                global_params = global_params if mu > 0 else None,
                device        = DEVICE,
            )
            local_updates.append((updated_params, n_samples))

        # ── 3. Aggregate (FedAvg, Eq. 3) ─────────────────────────────────
        global_params = federated_average(local_updates)
        global_model.set_parameters_flat(global_params)

        # ── 4. Evaluate on eval clients (validation split) ────────────────
        all_preds, all_labels = [], []
        for uid in eval_clients:
            eval_model = RecommenderNet(NUM_USERS, NUM_SKILLS)
            eval_model.set_parameters_flat(global_params)
            metrics = evaluate_local(eval_model, client_val_datasets[uid], device=DEVICE)
            # Accumulate per-client results (simple average across clients)
            all_preds.append(metrics["f1_score"])  # use F1 as primary signal

        # Aggregate evaluation metrics
        eval_model_agg = RecommenderNet(NUM_USERS, NUM_SKILLS)
        eval_model_agg.set_parameters_flat(global_params)
        # Evaluate on all eval clients concatenated
        import torch
        from torch.utils.data import ConcatDataset
        combined_val = ConcatDataset([client_val_datasets[uid] for uid in eval_clients])
        agg_metrics  = evaluate_local(eval_model_agg, combined_val, device=DEVICE)

        rec = {"round": rnd, **agg_metrics}
        records.append(rec)

        if rnd % 10 == 0 or rnd == 1:
            print(f"  Round {rnd:3d}/{N_ROUNDS}  "
                  f"F1={agg_metrics['f1_score']:.4f}  "
                  f"Acc={agg_metrics['accuracy']:.4f}  "
                  f"Prec={agg_metrics['precision']:.4f}  "
                  f"Rec={agg_metrics['recall']:.4f}")

    metrics_df = pd.DataFrame(records)

    # ── Summary statistics ────────────────────────────────────────────────
    best_idx  = metrics_df["f1_score"].idxmax()
    best_row  = metrics_df.iloc[best_idx]
    mean_f1   = metrics_df["f1_score"].mean()
    std_f1    = metrics_df["f1_score"].std()

    print(f"\n  ── Results for {label} ──")
    print(f"  Best F1   : {best_row.f1_score:.4f} (round {int(best_row.round)})")
    print(f"  Mean F1   : {mean_f1:.4f}")
    print(f"  Std Dev   : {std_f1:.4f}")

    # Save per-round metrics
    csv_path = os.path.join(OUT_DIR, f"fl_metrics_{label.replace(' ', '_')}.csv")
    metrics_df.to_csv(csv_path, index=False)

    return metrics_df, {
        "strategy"  : label,
        "mu"        : mu,
        "best_f1"   : round(float(best_row.f1_score), 4),
        "best_round": int(best_row.round),
        "mean_f1"   : round(float(mean_f1), 4),
        "std_dev"   : round(float(std_f1),  4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu",  type=float, default=None,
                        help="Single FedProx mu value to run (0 = FedAvg)")
    parser.add_argument("--all", action="store_true",
                        help="Run all four configurations (FedAvg + 3 FedProx)")
    args = parser.parse_args()

    # Default: run all if no flag given
    if not args.mu and not args.all:
        args.all = True

    configs = []
    if args.all:
        configs = [
            (0.0, "FedAvg"),
            (0.1, "FedProx_mu0.1"),
            (0.5, "FedProx_mu0.5"),
            (1.0, "FedProx_mu1.0"),
        ]
    else:
        mu    = args.mu
        label = "FedAvg" if mu == 0 else f"FedProx_mu{mu}"
        configs = [(mu, label)]

    all_summaries = []
    for mu, label in configs:
        _, summary = run_simulation(mu=mu, label=label)
        all_summaries.append(summary)

    # Save overall comparison table
    summary_df = pd.DataFrame(all_summaries)
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON (Table 6 in the paper)")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    with open(os.path.join(OUT_DIR, "fl_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=4)

    print(f"\nAll results saved to: {OUT_DIR}/")
