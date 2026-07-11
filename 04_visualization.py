"""
Step 4 — Visualization (Figures 4, 5, and 6 of the paper)
==========================================================
Reads the per-round metric CSV files produced by steps 2 and 3 and
generates publication-ready figures:

    Figure 4  — XGBoost metrics across 100 boosting rounds
    Figure 5  — FedProx (mu=0.5) metrics across 100 FL rounds
    Figure 6  — F1-Score comparison: FedProx mu=0.5 vs FedAvg

Input:
    outputs/centralized_metrics_per_round.csv
    outputs/fl_metrics_FedAvg.csv
    outputs/fl_metrics_FedProx_mu0.5.csv

Output:
    outputs/figure_centralized_metrics_evolution.png  (Figure 4)
    outputs/convergencia_metricas_federado.png         (Figure 5)
    outputs/fedavg_vs_fedprox_comparison.png           (Figure 6)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUT_DIR = os.path.join("outputs")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family" : "serif",
    "font.size"   : 11,
    "axes.grid"   : True,
    "grid.alpha"  : 0.3,
})

METRIC_LABELS = {
    "accuracy" : "Accuracy",
    "precision": "Precision",
    "recall"   : "Recall",
    "f1_score" : "F1-Score",
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ---------------------------------------------------------------------------
# Figure 4 — Centralized XGBoost metric evolution
# ---------------------------------------------------------------------------
def plot_centralized():
    path = os.path.join(OUT_DIR, "centralized_metrics_per_round.csv")
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found — run 02_centralized_baseline.py first.")
        return

    df = pd.read_csv(path)
    metrics = ["accuracy", "precision", "recall", "f1_score"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Centralised XGBoost — Performance Across 100 Boosting Rounds",
                 fontsize=13, y=1.02)

    for ax, metric, color in zip(axes, metrics, COLORS):
        ax.plot(df["round"], df[metric], color=color, linewidth=1.5, label=metric)
        best_idx = df[metric].idxmax()
        ax.axvline(df.loc[best_idx, "round"], color=color, linestyle="--",
                   alpha=0.5, linewidth=1)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Boosting Round")
        ax.set_ylabel("Score")
        ax.set_ylim(0.5, 1.0)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "figure_centralized_metrics_evolution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5 — Federated model (FedProx mu=0.5) metric evolution
# ---------------------------------------------------------------------------
def plot_federated(strategy_label: str = "FedProx_mu0.5"):
    path = os.path.join(OUT_DIR, f"fl_metrics_{strategy_label}.csv")
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found — run 03_federated_training.py first.")
        return

    df = pd.read_csv(path)
    metrics = ["accuracy", "precision", "recall", "f1_score"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Federated DNN ({strategy_label.replace('_', ' ')}) — "
        "Performance Across 100 Communication Rounds",
        fontsize=12, y=1.02,
    )

    for ax, metric, color in zip(axes, metrics, COLORS):
        ax.plot(df["round"], df[metric], color=color, linewidth=1.2, alpha=0.85)
        # Smoothed trend
        smoothed = df[metric].rolling(window=5, min_periods=1).mean()
        ax.plot(df["round"], smoothed, color=color, linewidth=2.2,
                linestyle="-", label="5-round moving avg")
        best_idx = df[metric].idxmax()
        ax.scatter(df.loc[best_idx, "round"], df.loc[best_idx, metric],
                   color=color, zorder=5, s=60,
                   label=f"Peak {df.loc[best_idx, metric]:.4f}")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "convergencia_metricas_federado.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6 — FedAvg vs FedProx (mu=0.5) F1-Score comparison
# ---------------------------------------------------------------------------
def plot_comparison():
    path_fedavg = os.path.join(OUT_DIR, "fl_metrics_FedAvg.csv")
    path_fedprox = os.path.join(OUT_DIR, "fl_metrics_FedProx_mu0.5.csv")

    if not os.path.exists(path_fedavg) or not os.path.exists(path_fedprox):
        print("[SKIP] FedAvg or FedProx_mu0.5 CSV not found.")
        return

    fedavg  = pd.read_csv(path_fedavg)
    fedprox = pd.read_csv(path_fedprox)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(fedprox["round"], fedprox["f1_score"],
            color="steelblue", linewidth=2.0, label="FedProx (μ=0.5)")
    ax.plot(fedavg["round"],  fedavg["f1_score"],
            color="darkorange", linewidth=1.8, linestyle="--", label="FedAvg")

    # Annotate peaks
    best_prox = fedprox.loc[fedprox["f1_score"].idxmax()]
    best_avg  = fedavg.loc[fedavg["f1_score"].idxmax()]

    ax.annotate(f"FedProx peak: {best_prox.f1_score:.4f}\n(round {int(best_prox['round'])})",
                xy=(best_prox['round'], best_prox.f1_score),
                xytext=(best_prox['round'] + 3, best_prox.f1_score - 0.04),
                arrowprops=dict(arrowstyle="->", color="steelblue"),
                color="steelblue", fontsize=9)

    ax.annotate(f"FedAvg peak: {best_avg.f1_score:.4f}\n(round {int(best_avg['round'])})",
                xy=(best_avg['round'], best_avg.f1_score),
                xytext=(best_avg['round'] + 3, best_avg.f1_score + 0.02),
                arrowprops=dict(arrowstyle="->", color="darkorange"),
                color="darkorange", fontsize=9)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score Comparison: FedProx (μ=0.5) vs FedAvg — 100 Rounds")
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 0.95)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fedavg_vs_fedprox_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    plot_centralized()
    plot_federated()
    plot_comparison()
    print("\nAll figures generated.")
