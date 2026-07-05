"""
Step 2 — Centralised XGBoost Baseline (Section 4.3.1 and Section 5.1)
======================================================================
Trains an XGBoost classifier on the fully aggregated processed dataset to
establish the upper-bound performance benchmark reported in Table 4 of the paper.

Input:   data/processed_assistments.csv
Outputs: outputs/centralized_metrics.json
         outputs/centralized_metrics_per_round.csv  (metrics across 100 boosting rounds)
         outputs/figure_centralized_metrics_evolution.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = os.path.join("data",    "processed_assistments.csv")
OUT_DIR   = os.path.join("outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES  = ["user_id_new", "skill_id_new",
             "user_mean_correct", "user_interaction_count", "skill_mean_correct"]
TARGET    = "target"
N_ROUNDS  = 100

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading processed dataset …")
df = pd.read_csv(DATA_PATH)

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
print(f"  Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")

# ---------------------------------------------------------------------------
# Train XGBoost incrementally over 100 rounds (Section 4.3.1)
# ---------------------------------------------------------------------------
print(f"\nTraining XGBoost over {N_ROUNDS} boosting rounds …")

records = []
model   = None

for n in range(1, N_ROUNDS + 1):
    model = xgb.XGBClassifier(
        n_estimators      = n,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = RANDOM_SEED,
        verbosity         = 0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    records.append({
        "round"    : n,
        "accuracy" : accuracy_score (y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score   (y_test, y_pred, zero_division=0),
        "f1_score" : f1_score       (y_test, y_pred, zero_division=0),
    })

metrics_df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Peak performance (paper reports peak at round 24)
# ---------------------------------------------------------------------------
best_idx = metrics_df["f1_score"].idxmax()
best     = metrics_df.iloc[best_idx]
print(f"\n{'='*50}")
print(f"PEAK CENTRALIZED PERFORMANCE (round {int(best.round)})")
print(f"{'='*50}")
print(f"  F1-Score  : {best.f1_score:.4f}  ({best.f1_score*100:.2f}%)")
print(f"  Accuracy  : {best.accuracy:.4f}  ({best.accuracy*100:.2f}%)")
print(f"  Precision : {best.precision:.4f}  ({best.precision*100:.2f}%)")
print(f"  Recall    : {best.recall:.4f}  ({best.recall*100:.2f}%)")
print(f"{'='*50}")

# Save peak metrics
peak_metrics = {
    "best_round": int(best.round),
    "f1_score"  : round(float(best.f1_score),  4),
    "accuracy"  : round(float(best.accuracy),  4),
    "precision" : round(float(best.precision), 4),
    "recall"    : round(float(best.recall),    4),
}
with open(os.path.join(OUT_DIR, "centralized_metrics.json"), "w") as f:
    json.dump(peak_metrics, f, indent=4)

metrics_df.to_csv(os.path.join(OUT_DIR, "centralized_metrics_per_round.csv"), index=False)

# ---------------------------------------------------------------------------
# Feature importance (Table 5 in the paper)
# ---------------------------------------------------------------------------
importances = dict(zip(FEATURES, model.feature_importances_))
print("\nFeature importances (XGBoost):")
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {feat:<30} {imp:.4f}")

# ---------------------------------------------------------------------------
# Plot — metrics evolution across 100 rounds (Figure 4 in the paper)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Centralised XGBoost — Metric Evolution (100 Boosting Rounds)", fontsize=13)

for ax, col, color in zip(axes,
                           ["accuracy", "precision", "recall", "f1_score"],
                           ["steelblue", "darkorange", "forestgreen", "crimson"]):
    ax.plot(metrics_df["round"], metrics_df[col], color=color, linewidth=1.5)
    ax.set_title(col.replace("_", " ").title())
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

plt.tight_layout()
out_fig = os.path.join(OUT_DIR, "figure_centralized_metrics_evolution.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_fig}")
