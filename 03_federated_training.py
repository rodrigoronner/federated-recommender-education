"""
Step 3 — Federated Learning Simulation (Sections 4.3.2, 5.2, and Table 6)
==========================================================================
Simulates federated training using the Flower (flwr) framework (Table 2:
Framework = Flower), which orchestrates client selection, local training,
and aggregation via its virtual-client simulation engine.

Each of the up to 1,365 students is registered as one independent FL client
(flwr.client.NumPyClient), holding private, locally-stored data — a strict
one-client-per-student mapping, with no student silently dropped.

The FedProx server-side strategy (flwr.server.strategy.FedProx) is used for
all four configurations; proximal_mu=0.0 recovers standard FedAvg exactly,
a correspondence the paper verifies explicitly in its ablation (Table 6).

Runs four configurations:
    - FedAvg        (mu = 0)
    - FedProx mu=0.1
    - FedProx mu=0.5  ← optimal, reported as main result
    - FedProx mu=1.0

For each configuration, the script logs per-round metrics aggregated by the
Flower strategy across the evaluation clients selected each round, then saves:
    outputs/fl_metrics_<strategy>.csv
    outputs/fl_summary.json

Usage:
    python 03_federated_training.py
    python 03_federated_training.py --mu 0.5        # single run
    python 03_federated_training.py --all            # all four configs

Run the visualization script afterwards:
    python 04_visualization.py
"""

import os, json, argparse, warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import flwr as fl
from flwr.client import NumPyClient, Client
from flwr.common import Scalar, NDArrays, ndarrays_to_parameters
from flwr.server.strategy import FedProx

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
# Load data and partition by student (one client = one student — Table 1)
# ---------------------------------------------------------------------------
print(f"Loading data from {DATA_PATH} …")
df = pd.read_csv(DATA_PATH)

NUM_USERS  = df["user_id_new"].nunique()
NUM_SKILLS = df["skill_id_new"].nunique()
print(f"  Students (clients): {NUM_USERS}  |  Skills (items): {NUM_SKILLS}")
print(f"  Total pairs: {len(df):,}")
print(f"  Device: {DEVICE}")

# Split each client's data into local train (80%) and local val (20%).
# Every student in the processed cohort becomes exactly one FL client — no
# student is dropped, matching the paper's "1,365 clients, one per student".
client_train_datasets = {}
client_val_datasets   = {}

for uid, group in df.groupby("user_id_new"):
    if len(group) >= 2:
        train_df, val_df = train_test_split(
            group, test_size=VAL_FRACTION, random_state=RANDOM_SEED, shuffle=True
        )
    else:
        # Too few rows to carve out a val split; train on all, empty val.
        train_df, val_df = group, group.iloc[0:0]
    client_train_datasets[uid] = StudentSkillDataset(train_df.reset_index(drop=True))
    client_val_datasets[uid]   = StudentSkillDataset(val_df.reset_index(drop=True))

CLIENT_IDS = sorted(client_train_datasets.keys())
N_CLIENTS  = len(CLIENT_IDS)
assert N_CLIENTS == NUM_USERS, (
    f"Expected one FL client per student ({NUM_USERS}), got {N_CLIENTS}."
)
print(f"  FL clients (one per student): {N_CLIENTS}")

n_fit = max(MIN_FIT_CLIENTS, int(FRACTION_FIT * N_CLIENTS))
print(f"  Clients per round — train: ~{n_fit}  |  eval: ~{int(FRACTION_EVAL * N_CLIENTS)}")


# ---------------------------------------------------------------------------
# Flower client (Section 4.3.2 — local training / evaluation per client)
# ---------------------------------------------------------------------------
class StudentClient(NumPyClient):
    """One Flower client per student, wrapping RecommenderNet local train/eval."""

    def __init__(self, uid: int):
        self.uid   = uid
        self.model = RecommenderNet(NUM_USERS, NUM_SKILLS)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.model.get_parameters_flat()

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.model.set_parameters_flat(parameters)

        # proximal_mu is injected automatically into `config` by the
        # FedProx strategy (0.0 for the FedAvg run — see Eq. 4 / Eq. 3).
        mu = float(config.get("proximal_mu", 0.0))

        updated_params, n_samples = train_local(
            model         = self.model,
            dataset       = client_train_datasets[self.uid],
            epochs        = int(config.get("local_epochs", LOCAL_EPOCHS)),
            batch_size    = int(config.get("batch_size", LOCAL_BATCH_SIZE)),
            lr            = float(config.get("lr", LEARNING_RATE)),
            mu            = mu,
            global_params = parameters if mu > 0 else None,
            device        = DEVICE,
        )
        return updated_params, n_samples, {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.model.set_parameters_flat(parameters)
        val_ds = client_val_datasets[self.uid]

        if len(val_ds) == 0:
            return 0.0, 0, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        metrics = evaluate_local(self.model, val_ds, device=DEVICE)

        # BCE loss on the local validation split (required scalar return by Flower).
        criterion = nn.BCELoss()
        self.model.eval()
        with torch.no_grad():
            u = val_ds.user_ids.to(DEVICE)
            s = val_ds.skill_ids.to(DEVICE)
            f = val_ds.features.to(DEVICE)
            y = val_ds.labels.to(DEVICE)
            preds = self.model(u, s, f)
            loss  = criterion(preds, y).item()

        return loss, len(val_ds), metrics


def client_fn(cid: str) -> Client:
    """Flower calls this once per selected client per round (Section 4.3.2)."""
    return StudentClient(uid=int(cid)).to_client()


# ---------------------------------------------------------------------------
# Server-side metric aggregation (weighted by client sample count, Eq. 3)
# ---------------------------------------------------------------------------
def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    total = sum(n for n, _ in metrics if n > 0)
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    return {
        key: sum(n * m[key] for n, m in metrics if n > 0) / total
        for key in ["accuracy", "precision", "recall", "f1_score"]
    }


def fit_config(server_round: int) -> Dict[str, Scalar]:
    return {
        "local_epochs": LOCAL_EPOCHS,
        "batch_size"  : LOCAL_BATCH_SIZE,
        "lr"          : LEARNING_RATE,
    }


# ---------------------------------------------------------------------------
# One complete FL simulation (one strategy configuration)
# ---------------------------------------------------------------------------
def run_simulation(mu: float = 0.0, label: str = "FedAvg") -> pd.DataFrame:
    """
    Run N_ROUNDS of Flower-orchestrated federated learning for a given
    FedProx proximal term mu (mu = 0 recovers FedAvg — Table 6 ablation).

    Returns a DataFrame with per-round evaluation metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Strategy: {label}  (mu={mu})  —  Flower simulation")
    print(f"{'='*60}")

    torch.manual_seed(RANDOM_SEED)
    initial_parameters = ndarrays_to_parameters(
        RecommenderNet(NUM_USERS, NUM_SKILLS).get_parameters_flat()
    )

    strategy = FedProx(
        fraction_fit                  = FRACTION_FIT,
        fraction_evaluate             = FRACTION_EVAL,
        min_fit_clients               = MIN_FIT_CLIENTS,
        min_evaluate_clients          = 2,
        min_available_clients         = N_CLIENTS,
        on_fit_config_fn              = fit_config,
        evaluate_metrics_aggregation_fn = weighted_average,
        initial_parameters            = initial_parameters,
        proximal_mu                   = mu,
    )

    history = fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = N_CLIENTS,
        config            = fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy          = strategy,
        client_resources  = {"num_cpus": 1},
        ray_init_args     = {"include_dashboard": False, "ignore_reinit_error": True},
    )

    # ── Extract per-round metrics from Flower's History object ─────────────
    dist = history.metrics_distributed
    rounds = [r for r, _ in dist.get("f1_score", [])]
    records = [
        {
            "round"    : rnd,
            "accuracy" : dict(dist.get("accuracy",  []))[rnd],
            "precision": dict(dist.get("precision", []))[rnd],
            "recall"   : dict(dist.get("recall",    []))[rnd],
            "f1_score" : dict(dist.get("f1_score",  []))[rnd],
        }
        for rnd in rounds
    ]
    metrics_df = pd.DataFrame(records)

    for rnd in [1] + [r for r in rounds if r % 10 == 0]:
        row = metrics_df.loc[metrics_df["round"] == rnd].iloc[0]
        print(f"  Round {rnd:3d}/{N_ROUNDS}  "
              f"F1={row.f1_score:.4f}  Acc={row.accuracy:.4f}  "
              f"Prec={row.precision:.4f}  Rec={row.recall:.4f}")

    # ── Summary statistics ────────────────────────────────────────────────
    best_idx  = metrics_df["f1_score"].idxmax()
    best_row  = metrics_df.iloc[best_idx]
    mean_f1   = metrics_df["f1_score"].mean()
    std_f1    = metrics_df["f1_score"].std()

    print(f"\n  ── Results for {label} ──")
    print(f"  Best F1   : {best_row.f1_score:.4f} (round {int(best_row['round'])})")
    print(f"  Mean F1   : {mean_f1:.4f}")
    print(f"  Std Dev   : {std_f1:.4f}")

    csv_path = os.path.join(OUT_DIR, f"fl_metrics_{label.replace(' ', '_')}.csv")
    metrics_df.to_csv(csv_path, index=False)

    return metrics_df, {
        "strategy"  : label,
        "mu"        : mu,
        "best_f1"   : round(float(best_row.f1_score), 4),
        "best_round": int(best_row["round"]),
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
