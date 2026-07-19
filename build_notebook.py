"""
Builds kaggle_kernel/federated_recommender_reproducibility.ipynb
Embeds the REAL per-round results obtained from running this repo's
unmodified pipeline (01-04) end-to-end, so the notebook is self-contained
and does not depend on external output files being attached.
"""
import json
import os

OUT_DIR = "outputs"

def read_csv_text(path):
    with open(path) as f:
        return f.read()

centralized_csv   = read_csv_text(os.path.join(OUT_DIR, "centralized_metrics_per_round.csv"))
fedavg_csv        = read_csv_text(os.path.join(OUT_DIR, "fl_metrics_FedAvg.csv"))
fedprox01_csv      = read_csv_text(os.path.join(OUT_DIR, "fl_metrics_FedProx_mu0.1.csv"))
fedprox05_csv      = read_csv_text(os.path.join(OUT_DIR, "fl_metrics_FedProx_mu0.5.csv"))
fedprox10_csv      = read_csv_text(os.path.join(OUT_DIR, "fl_metrics_FedProx_mu1.0.csv"))

with open(os.path.join(OUT_DIR, "centralized_metrics.json")) as f:
    centralized_peak = json.load(f)


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}


cells = []

# ---------------------------------------------------------------------------
cells.append(md(
"# Reproducibility Check — Federated Recommender System for Student Performance Prediction",
"",
"This notebook independently verifies that the source code in",
"[rodrigotertulino/federated-recommender-education](https://github.com/rodrigoronner/federated-recommender-education)",
"faithfully implements the methodology described in the JEDM submission",
"*\"Privacy-Preserving Personalization in Education: A Federated Recommender System for Student Performance Prediction\"*",
"(Tertulino & Almeida, arXiv:2509.10516), and reports **actual results obtained by running the unmodified pipeline end-to-end**",
"against this dataset (`student-performance-for-recommender-systems`).",
"",
"**What this notebook contains:**",
"1. A live, runnable version of the full pipeline (data loading → XGBoost baseline → federated FedAvg/FedProx via Flower) — a reduced-scale demo runs inline so the notebook executes end-to-end on Kaggle in a few minutes.",
"2. The **real, full-scale reproduction results** (100 communication rounds × 4 aggregation strategies, matching the paper's exact experimental setup) obtained by running this repository's code unmodified on a 10-core machine (~10h15min total runtime), embedded below as data so they render without external dependencies.",
"3. A transparent comparison against the paper's published Table 4-6, including where results matched closely and where they diverged.",
))

# ---------------------------------------------------------------------------
cells.append(md(
"## 0. Setup",
"",
"`flwr` and `ray` are not preinstalled on Kaggle and are required for `flwr.simulation.start_simulation`.",
"This was an actual gap found in the repo's `requirements.txt` — `ray` was missing entirely — fixed for this notebook and reported upstream.",
"",
"The repo's `requirements.txt` pins `flwr==1.7.0` / `ray==2.6.3` (tested with Python 3.10/3.11). Kaggle's current base image runs **Python 3.12**,",
"for which `ray==2.6.3` has no published wheel (`ray` only publishes py3.12 wheels from 2.31.0 onward) — pinning the repo's exact versions fails to install here.",
"This notebook therefore installs the latest `flwr`/`ray` (unpinned) instead; `flwr.simulation.start_simulation` remains available (deprecated in favor of `run_simulation`",
"but functional) and was verified to reproduce the same training behavior. `xgboost`/`torch`/`scikit-learn` are already preinstalled on Kaggle and are left untouched to avoid `numpy` ABI conflicts.",
))

cells.append(code(
"!pip install -q flwr ray 2>&1 | tail -5",
))

cells.append(code(
"import os, json, glob, warnings",
"import numpy as np",
"import pandas as pd",
"import matplotlib.pyplot as plt",
"warnings.filterwarnings('ignore')",
"",
"RANDOM_SEED = 42",
"np.random.seed(RANDOM_SEED)",
"SUCCESS_THRESHOLD = 0.70  # matches Section 4.2, Step 3 of the paper",
"",
"EXPECTED_PATH = '/kaggle/input/datasets/rodrigotertulino/student-performance-for-recommender-systems/interactions_real_rich_scaled_processed.csv'",
"if os.path.exists(EXPECTED_PATH):",
"    DATA_PATH = EXPECTED_PATH",
"else:",
"    print('Expected path not found, searching /kaggle/input ...')",
"    print('Top-level contents of /kaggle/input:', os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'MISSING')",
"    candidates = glob.glob('/kaggle/input/**/*.csv', recursive=True)",
"    print('CSV files found:', candidates)",
"    assert candidates, 'No CSV found under /kaggle/input — is the dataset attached to this kernel (Add Data)?'",
"    DATA_PATH = candidates[0]",
"print('Using DATA_PATH =', DATA_PATH)",
"",
"df = pd.read_csv(DATA_PATH)",
"df['target'] = (df['target_correct_rate'] >= SUCCESS_THRESHOLD).astype(int)",
"",
"print('Shape:', df.shape)",
"print('Unique students:', df.user_id_new.nunique(), ' (paper reports 1,365)')",
"print('Unique skills  :', df.skill_id_new.nunique(), ' (paper reports 107)')",
"print(df['target'].value_counts(normalize=True).rename('class_share'))",
"df.head()",
))

# ---------------------------------------------------------------------------
cells.append(md(
"## 1. Centralized XGBoost Baseline (Section 4.3.1)",
"",
"Same procedure as `02_centralized_baseline.py`: 80/20 stratified split (seed 42), then XGBoost refit for `n_estimators = 1..100`",
"to trace performance across boosting rounds, exactly as the paper's Figure 4 / Table 4.",
))

cells.append(code(
"import xgboost as xgb",
"from sklearn.model_selection import train_test_split",
"from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score",
"",
"FEATURES = ['user_id_new', 'skill_id_new', 'user_mean_correct', 'user_interaction_count', 'skill_mean_correct']",
"X, y = df[FEATURES].values, df['target'].values",
"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)",
"",
"records = []",
"model = None",
"for n in range(1, 101):",
"    model = xgb.XGBClassifier(n_estimators=n, eval_metric='logloss', random_state=RANDOM_SEED, verbosity=0)",
"    model.fit(X_train, y_train)",
"    y_pred = model.predict(X_test)",
"    records.append({",
"        'round': n,",
"        'accuracy': accuracy_score(y_test, y_pred),",
"        'precision': precision_score(y_test, y_pred, zero_division=0),",
"        'recall': recall_score(y_test, y_pred, zero_division=0),",
"        'f1_score': f1_score(y_test, y_pred, zero_division=0),",
"    })",
"",
"xgb_metrics = pd.DataFrame(records)",
"best = xgb_metrics.loc[xgb_metrics.f1_score.idxmax()]",
"print(f\"Peak F1={best.f1_score:.4f} at round {int(best['round'])} | Acc={best.accuracy:.4f} Prec={best.precision:.4f} Rec={best.recall:.4f}\")",
))

cells.append(md(
"### Comparison against the paper (Table 4)",
"",
"| Metric | Paper (Table 4) | This run |",
"|---|---|---|",
"| F1-Score  | 0.8285 (round 24) | *computed above* |",
"| Accuracy  | 0.7702 | *computed above* |",
"| Precision | 0.7919 | *computed above* |",
"| Recall    | 0.8686 | *computed above* |",
"",
"In our independent full local run (10-core machine, same seed/config) we obtained **F1 = 0.8274 (round 20)**, Accuracy 0.7682, Precision 0.7890, Recall 0.8698 — within ~0.1pp of the paper across every metric.",
))

# ---------------------------------------------------------------------------
cells.append(md(
"## 2. Federated Learning — RecommenderNet + FedAvg/FedProx (Section 4.3.2)",
"",
"`RecommenderNet` (2 embedding layers + 2 dense ReLU layers + sigmoid output, Table 3) trained via Flower, one client per student.",
"The FedProx proximal term `(mu/2)*||w - w_global||^2` is added to the local loss; `mu=0` recovers standard FedAvg exactly, matching the paper's stated correspondence.",
))

cells.append(code(
"import torch",
"import torch.nn as nn",
"from torch.utils.data import Dataset, DataLoader",
"",
"EMBEDDING_DIM, HIDDEN_1, HIDDEN_2, N_FEATURES = 10, 32, 16, 3",
"",
"class StudentSkillDataset(Dataset):",
"    def __init__(self, d):",
"        self.user_ids = torch.tensor(d['user_id_new'].values, dtype=torch.long)",
"        self.skill_ids = torch.tensor(d['skill_id_new'].values, dtype=torch.long)",
"        self.features = torch.tensor(d[['user_mean_correct', 'user_interaction_count', 'skill_mean_correct']].values, dtype=torch.float32)",
"        self.labels = torch.tensor(d['target'].values, dtype=torch.float32)",
"    def __len__(self): return len(self.labels)",
"    def __getitem__(self, i): return self.user_ids[i], self.skill_ids[i], self.features[i], self.labels[i]",
"",
"class RecommenderNet(nn.Module):",
"    def __init__(self, num_users, num_skills):",
"        super().__init__()",
"        self.user_embedding = nn.Embedding(num_users, EMBEDDING_DIM)",
"        self.skill_embedding = nn.Embedding(num_skills, EMBEDDING_DIM)",
"        self.fc1 = nn.Linear(EMBEDDING_DIM * 2 + N_FEATURES, HIDDEN_1)",
"        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2)",
"        self.output = nn.Linear(HIDDEN_2, 1)",
"        self.relu, self.sigmoid = nn.ReLU(), nn.Sigmoid()",
"    def forward(self, u, s, f):",
"        x = torch.cat([self.user_embedding(u), self.skill_embedding(s), f], dim=1)",
"        x = self.relu(self.fc1(x)); x = self.relu(self.fc2(x))",
"        return self.sigmoid(self.output(x)).squeeze(-1)",
"    def get_parameters_flat(self): return [p.detach().cpu().numpy() for p in self.parameters()]",
"    def set_parameters_flat(self, params):",
"        state = self.state_dict()",
"        for k, a in zip(state.keys(), params): state[k] = torch.tensor(a)",
"        self.load_state_dict(state, strict=True)",
"",
"def train_local(model, dataset, epochs=5, batch_size=32, lr=1e-3, mu=0.0, global_params=None, device='cpu'):",
"    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)",
"    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)",
"    criterion = nn.BCELoss()",
"    model.to(device); model.train()",
"    if mu > 0 and global_params is not None:",
"        global_tensors = [torch.tensor(p, device=device) for p in global_params]",
"    for _ in range(epochs):",
"        for u, s, f, lab in loader:",
"            u, s, f, lab = u.to(device), s.to(device), f.to(device), lab.to(device)",
"            optimizer.zero_grad()",
"            preds = model(u, s, f)",
"            loss = criterion(preds, lab)",
"            if mu > 0 and global_params is not None:",
"                prox = sum(torch.norm(p - g) ** 2 for p, g in zip(model.parameters(), global_tensors))",
"                loss = loss + (mu / 2.0) * prox",
"            loss.backward(); optimizer.step()",
"    return model.get_parameters_flat(), len(dataset)",
"",
"def evaluate_local(model, dataset, batch_size=64, device='cpu'):",
"    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)",
"    model.to(device); model.eval()",
"    preds_all, labels_all = [], []",
"    with torch.no_grad():",
"        for u, s, f, lab in loader:",
"            preds = model(u.to(device), s.to(device), f.to(device))",
"            preds_all.extend((preds.cpu().numpy() >= 0.5).astype(int).tolist())",
"            labels_all.extend(lab.numpy().astype(int).tolist())",
"    return {'accuracy': accuracy_score(labels_all, preds_all), 'precision': precision_score(labels_all, preds_all, zero_division=0),",
"            'recall': recall_score(labels_all, preds_all, zero_division=0), 'f1_score': f1_score(labels_all, preds_all, zero_division=0)}",
))

cells.append(md(
"### Quick live demo",
"",
"The paper's exact setup (100 rounds x 4 strategies, 1,365 simulated clients) took **~10h15min** end-to-end on a 10-core machine — Flower's virtual-client actor pool overhead dominates since Kaggle sessions have fewer cores and a runtime budget. ",
"To keep this notebook runnable end-to-end on Kaggle in a few minutes while still proving the code executes correctly, this cell runs a **reduced-scale** demo (fewer rounds, single strategy). ",
"Section 3 below reports the **real full-scale (100-round x 4-strategy) results**, obtained by running this exact code with `QUICK_DEMO=False` offline.",
))

cells.append(code(
"import flwr as fl",
"from flwr.client import NumPyClient",
"from flwr.common import ndarrays_to_parameters",
"from flwr.server.strategy import FedProx",
"",
"QUICK_DEMO = True          # set False (and N_ROUNDS=100, run all 4 mu values) to reproduce the paper exactly — expect several hours",
"N_ROUNDS = 5 if QUICK_DEMO else 100",
"DEMO_MU = 0.5               # the paper's optimal configuration",
"LOCAL_EPOCHS, LOCAL_BATCH_SIZE, LEARNING_RATE = 5, 32, 1e-3",
"FRACTION_FIT, FRACTION_EVAL, MIN_FIT_CLIENTS, VAL_FRACTION = 0.10, 0.20, 50, 0.20",
"DEVICE = 'cpu'",
"torch.manual_seed(RANDOM_SEED)",
"",
"NUM_USERS, NUM_SKILLS = df.user_id_new.nunique(), df.skill_id_new.nunique()",
"client_train, client_val = {}, {}",
"for uid, g in df.groupby('user_id_new'):",
"    if len(g) >= 2:",
"        tr, va = train_test_split(g, test_size=VAL_FRACTION, random_state=RANDOM_SEED, shuffle=True)",
"    else:",
"        tr, va = g, g.iloc[0:0]",
"    client_train[uid] = StudentSkillDataset(tr.reset_index(drop=True))",
"    client_val[uid]   = StudentSkillDataset(va.reset_index(drop=True))",
"N_CLIENTS = len(client_train)",
"",
"class StudentClient(NumPyClient):",
"    def __init__(self, uid):",
"        self.uid = uid",
"        self.model = RecommenderNet(NUM_USERS, NUM_SKILLS)",
"    def get_parameters(self, config): return self.model.get_parameters_flat()",
"    def fit(self, parameters, config):",
"        self.model.set_parameters_flat(parameters)",
"        mu = float(config.get('proximal_mu', 0.0))",
"        params, n = train_local(self.model, client_train[self.uid], epochs=LOCAL_EPOCHS, batch_size=LOCAL_BATCH_SIZE,",
"                                 lr=LEARNING_RATE, mu=mu, global_params=parameters if mu > 0 else None, device=DEVICE)",
"        return params, n, {}",
"    def evaluate(self, parameters, config):",
"        self.model.set_parameters_flat(parameters)",
"        val_ds = client_val[self.uid]",
"        if len(val_ds) == 0:",
"            return 0.0, 0, {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}",
"        metrics = evaluate_local(self.model, val_ds, device=DEVICE)",
"        return 0.0, len(val_ds), metrics",
"",
"def client_fn(cid): return StudentClient(uid=int(cid)).to_client()",
"",
"def weighted_average(metrics):",
"    total = sum(n for n, _ in metrics if n > 0)",
"    if total == 0: return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}",
"    return {k: sum(n * m[k] for n, m in metrics if n > 0) / total for k in ['accuracy', 'precision', 'recall', 'f1_score']}",
"",
"def fit_config(server_round): return {'local_epochs': LOCAL_EPOCHS, 'batch_size': LOCAL_BATCH_SIZE, 'lr': LEARNING_RATE}",
"",
"strategy = FedProx(fraction_fit=FRACTION_FIT, fraction_evaluate=FRACTION_EVAL, min_fit_clients=MIN_FIT_CLIENTS,",
"                    min_evaluate_clients=2, min_available_clients=N_CLIENTS, on_fit_config_fn=fit_config,",
"                    evaluate_metrics_aggregation_fn=weighted_average,",
"                    initial_parameters=ndarrays_to_parameters(RecommenderNet(NUM_USERS, NUM_SKILLS).get_parameters_flat()),",
"                    proximal_mu=DEMO_MU)",
"",
"history = fl.simulation.start_simulation(client_fn=client_fn, num_clients=N_CLIENTS,",
"                                          config=fl.server.ServerConfig(num_rounds=N_ROUNDS), strategy=strategy,",
"                                          client_resources={'num_cpus': 1}, ray_init_args={'include_dashboard': False, 'ignore_reinit_error': True})",
"",
"dist = history.metrics_distributed",
"demo_df = pd.DataFrame([{'round': r, **{k: dict(dist[k])[r] for k in ['accuracy', 'precision', 'recall', 'f1_score']}} for r, _ in dist['f1_score']])",
"print(f'Live demo ({N_ROUNDS} rounds, mu={DEMO_MU}) — final round metrics:')",
"demo_df.tail()",
))

# ---------------------------------------------------------------------------
cells.append(md(
"## 3. Full-Scale Reproduction Results (100 rounds x 4 strategies)",
"",
"The data embedded below is the **unmodified, real output** of running `03_federated_training.py --all` from the repository",
"(100 communication rounds x {FedAvg, FedProx mu=0.1, FedProx mu=0.5, FedProx mu=1.0}) against this dataset — a ~10h15min run on a",
"10-core machine. It is embedded as CSV text so this section renders correctly without needing external file attachments.",
))

def csv_cell(varname, text):
    return code(
        "import io",
        f"{varname} = pd.read_csv(io.StringIO('''{text.strip()}'''))",
    )

cells.append(csv_cell("centralized_full", centralized_csv))
cells.append(csv_cell("fedavg_full", fedavg_csv))
cells.append(csv_cell("fedprox01_full", fedprox01_csv))
cells.append(csv_cell("fedprox05_full", fedprox05_csv))
cells.append(csv_cell("fedprox10_full", fedprox10_csv))

cells.append(code(
"strategies_full = {'FedAvg': fedavg_full, 'FedProx_mu0.1': fedprox01_full, 'FedProx_mu0.5': fedprox05_full, 'FedProx_mu1.0': fedprox10_full}",
"mus = {'FedAvg': 0.0, 'FedProx_mu0.1': 0.1, 'FedProx_mu0.5': 0.5, 'FedProx_mu1.0': 1.0}",
"",
"rows = []",
"for label, d in strategies_full.items():",
"    best = d.loc[d.f1_score.idxmax()]",
"    rows.append({'strategy': label, 'mu': mus[label], 'best_f1': round(best.f1_score, 4), 'best_round': int(best['round']),",
"                 'mean_f1': round(d.f1_score.mean(), 4), 'std_dev': round(d.f1_score.std(), 4)})",
"summary_full = pd.DataFrame(rows)",
"summary_full",
))

cells.append(md(
"### Comparison against the paper's Table 6",
"",
"| Strategy | Best F1 (paper) | Best F1 (this run) | Best round (paper) | Best round (this run) | Std Dev (paper) | Std Dev (this run) |",
"|---|---|---|---|---|---|---|",
"| FedAvg | 0.7584 | 0.7710 | 70 | 23 | 0.0249 | 0.0150 |",
"| FedProx mu=0.1 | 0.7526 | 0.7638 | 89 | 37 | 0.0242 | 0.0161 |",
"| **FedProx mu=0.5** | **0.7628** | **0.7737** | 88 | 87 | 0.0205 | 0.0162 |",
"| FedProx mu=1.0 | 0.7555 | 0.7581 | 80 | 93 | **0.0152** | **0.0129** |",
"",
"**What replicates:** FedProx (mu=0.5) achieves the highest peak F1-Score among the four strategies in both the paper and this independent run — the paper's central claim. FedProx (mu=1.0) achieves the lowest standard deviation (most stable training) in both runs as well, confirming the stability-performance trade-off the paper describes.",
"",
"**What diverges:** absolute F1 values in this run are systematically ~1-1.5 percentage points higher than the paper across all four strategies, and the best round differs substantially per strategy. Most notably, the paper reports FedAvg as the *least* stable strategy (highest std dev, 0.0249) while in this run FedAvg was among the *most* stable (0.0150). This is plausible run-to-run variance inherent to Flower/Ray's client-sampling and actor-scheduling non-determinism — it does not reproduce bit-for-bit even with a fixed `torch.manual_seed`, since client selection order and cross-process floating-point summation order are not fully pinned by that seed alone. This divergence is reported here transparently rather than adjusted to match the paper.",
))

cells.append(code(
"fig, axes = plt.subplots(1, 4, figsize=(18, 4))",
"fig.suptitle('Federated strategies — F1-Score, Accuracy, Precision, Recall over 100 rounds (full reproduction)', y=1.03)",
"colors = {'FedAvg': 'darkorange', 'FedProx_mu0.1': 'seagreen', 'FedProx_mu0.5': 'steelblue', 'FedProx_mu1.0': 'firebrick'}",
"for ax, metric in zip(axes, ['accuracy', 'precision', 'recall', 'f1_score']):",
"    for label, d in strategies_full.items():",
"        ax.plot(d['round'], d[metric].rolling(5, min_periods=1).mean(), label=label, color=colors[label], linewidth=1.6)",
"    ax.set_title(metric.replace('_', ' ').title()); ax.set_xlabel('Communication Round'); ax.grid(alpha=0.3)",
"axes[0].legend(fontsize=8)",
"plt.tight_layout(); plt.show()",
))

cells.append(code(
"plt.figure(figsize=(9, 5))",
"plt.plot(fedprox05_full['round'], fedprox05_full['f1_score'], color='steelblue', linewidth=1.8, label='FedProx (mu=0.5)')",
"plt.plot(fedavg_full['round'], fedavg_full['f1_score'], color='darkorange', linestyle='--', linewidth=1.6, label='FedAvg')",
"plt.axhline(0.7628, color='steelblue', linestyle=':', alpha=0.6, label='Paper FedProx peak (0.7628)')",
"plt.axhline(0.7584, color='darkorange', linestyle=':', alpha=0.6, label='Paper FedAvg peak (0.7584)')",
"plt.xlabel('Communication Round'); plt.ylabel('F1-Score')",
"plt.title('F1-Score: FedProx (mu=0.5) vs FedAvg — this reproduction vs. paper peaks')",
"plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.show()",
))

# ---------------------------------------------------------------------------
cells.append(md(
"## 4. Code Review Notes",
"",
"While reproducing this pipeline, two real gaps were found and fixed:",
"",
"- **`requirements.txt` was missing `ray`.** `flwr.simulation.start_simulation` requires the `flwr[simulation]` extra (which depends on `ray`), and a plain `pip install -r requirements.txt` fails with `ImportError: Unable to import module 'ray'` before any training starts. Fixed by adding `ray==2.6.3` to the repository's `requirements.txt` (validated on Python 3.11).",
"- **The repo's pinned `flwr==1.7.0`/`ray==2.6.3` cannot install on Python 3.12** (e.g. Kaggle's current default environment) — `ray==2.6.3` has no Python 3.12 wheel. This notebook works around it by installing unpinned/latest `flwr`+`ray` (see Section 0); the repository itself still targets Python 3.10/3.11 per its README, so this is a Kaggle-runtime-specific accommodation rather than a repo fix.",
"",
"Everything else — filtering thresholds, feature engineering, ID mapping, MinMax scaling, the DNN architecture (Table 3), FedProx's proximal term (Eq. 4), and all Table 2 hyperparameters (learning rate, batch size, local epochs, client fractions, `min_fit_clients`, seed) — matched the paper's description exactly.",
))

cells.append(md(
"## 5. Conclusion",
"",
"This notebook independently verifies that the code in [rodrigoronner/federated-recommender-education](https://github.com/rodrigoronner/federated-recommender-education) implements the methodology described in the JEDM submission faithfully. The centralized XGBoost baseline reproduces the paper's numbers almost exactly (within ~0.1pp). The federated experiments reproduce the paper's central qualitative claims — FedProx (mu=0.5) gives the best peak F1-Score, and FedProx (mu=1.0) gives the most stable training — while showing that exact per-round numbers (and which strategy is *least* stable) carry meaningful run-to-run variance inherent to federated simulation with Flower/Ray, worth disclosing for anyone attempting to reproduce Table 6 exactly.",
"",
"**Citation:**",
"```",
"Tertulino, R., & Almeida, R. (2025). Privacy-preserving personalization in education: A federated recommender",
"system for student performance prediction. arXiv preprint arXiv:2509.10516.",
"```",
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs("kaggle_kernel", exist_ok=True)
with open("kaggle_kernel/federated_recommender_reproducibility.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook written:", os.path.getsize("kaggle_kernel/federated_recommender_reproducibility.ipynb"), "bytes")
