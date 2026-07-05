"""
RecommenderNet — DNN Architecture (Section 4.3.2 and Table 3 of the paper)
==========================================================================
A hybrid Deep Neural Network that combines user and skill embedding layers
with engineered feature inputs, as described in the paper's architecture table.

Architecture:
    Layer 1 — Embedding(user_id)      : (batch, 1)     → (batch, 10)
    Layer 2 — Embedding(skill_id)     : (batch, 1)     → (batch, 10)
    Layer 3 — Input (3 engineered)    : (batch, 3)     → (batch, 3)
    Layer 4 — Concatenation           : (batch, 23)
    Layer 5 — Dense (Hidden 1)        : (batch, 23)    → (batch, 32)  ReLU
    Layer 6 — Dense (Hidden 2)        : (batch, 32)    → (batch, 16)  ReLU
    Layer 7 — Dense (Output)          : (batch, 16)    → (batch, 1)   Sigmoid
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

EMBEDDING_DIM = 10   # embedding size for user and skill (Section 4.3.2)
HIDDEN_1      = 32   # hidden layer 1 neurons
HIDDEN_2      = 16   # hidden layer 2 neurons
N_FEATURES    = 3    # engineered features


class StudentSkillDataset(Dataset):
    """PyTorch Dataset for student-skill interaction pairs."""

    def __init__(self, df):
        self.user_ids  = torch.tensor(df["user_id_new"].values,  dtype=torch.long)
        self.skill_ids = torch.tensor(df["skill_id_new"].values, dtype=torch.long)
        self.features  = torch.tensor(
            df[["user_mean_correct", "user_interaction_count", "skill_mean_correct"]].values,
            dtype=torch.float32,
        )
        self.labels = torch.tensor(df["target"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.user_ids[idx], self.skill_ids[idx],
                self.features[idx], self.labels[idx])


class RecommenderNet(nn.Module):
    """
    Hybrid DNN recommender combining embedding-based collaborative filtering
    with engineered contextual features, trained via Federated Learning.
    """

    def __init__(self, num_users: int, num_skills: int):
        super().__init__()
        # Embedding layers (Layers 1-2)
        self.user_embedding  = nn.Embedding(num_users,  EMBEDDING_DIM)
        self.skill_embedding = nn.Embedding(num_skills, EMBEDDING_DIM)

        # Dense layers (Layers 5-7)
        in_dim = EMBEDDING_DIM + EMBEDDING_DIM + N_FEATURES  # = 23
        self.fc1    = nn.Linear(in_dim,   HIDDEN_1)
        self.fc2    = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.output = nn.Linear(HIDDEN_2, 1)

        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, skill_ids, features):
        # Layers 1-2: look up embeddings
        user_emb  = self.user_embedding(user_ids)    # (batch, 10)
        skill_emb = self.skill_embedding(skill_ids)  # (batch, 10)

        # Layer 4: concatenation (batch, 23)
        x = torch.cat([user_emb, skill_emb, features], dim=1)

        # Layers 5-6: dense + ReLU
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Layer 7: output sigmoid
        return self.sigmoid(self.output(x)).squeeze(-1)

    def get_parameters_flat(self):
        """Return all model parameters as a list of numpy arrays (for Flower)."""
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_parameters_flat(self, params):
        """Load parameters from a list of numpy arrays (from Flower server)."""
        state = self.state_dict()
        for key, arr in zip(state.keys(), params):
            state[key] = torch.tensor(arr)
        self.load_state_dict(state, strict=True)


def train_local(model, dataset, epochs: int = 5, batch_size: int = 32,
                lr: float = 1e-3, mu: float = 0.0, global_params=None,
                device: str = "cpu"):
    """
    Local training step for one FL client (Section 4.3.2).

    Parameters
    ----------
    model        : RecommenderNet instance (already loaded with global weights)
    dataset      : StudentSkillDataset for this client
    epochs       : local epochs per round (default 5, per Table 2)
    batch_size   : local batch size (default 32, per Table 2)
    lr           : learning rate (1e-3, per Table 2)
    mu           : FedProx proximal hyperparameter (0 = FedAvg, per Eq. 4)
    global_params: snapshot of global weights for proximal term (required when mu > 0)
    device       : torch device string

    Returns
    -------
    updated parameters as list of numpy arrays, number of samples trained on
    """
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.BCELoss()
    model.to(device)
    model.train()

    # Cache global parameters for proximal term (FedProx, Eq. 4)
    if mu > 0 and global_params is not None:
        global_tensors = [torch.tensor(p, device=device) for p in global_params]

    for _ in range(epochs):
        for user_ids, skill_ids, features, labels in loader:
            user_ids  = user_ids.to(device)
            skill_ids = skill_ids.to(device)
            features  = features.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            preds = model(user_ids, skill_ids, features)
            loss  = criterion(preds, labels)

            # FedProx proximal term: (mu/2) * ||w - w_global||^2 (Eq. 4)
            if mu > 0 and global_params is not None:
                prox = sum(
                    torch.norm(p - g) ** 2
                    for p, g in zip(model.parameters(), global_tensors)
                )
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()

    return model.get_parameters_flat(), len(dataset)


def evaluate_local(model, dataset, batch_size: int = 64, device: str = "cpu"):
    """
    Evaluate the model on a client's local validation split.

    Returns dict with accuracy, precision, recall, f1.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for user_ids, skill_ids, features, labels in loader:
            preds = model(user_ids.to(device), skill_ids.to(device), features.to(device))
            all_preds.extend((preds.cpu().numpy() >= 0.5).astype(int).tolist())
            all_labels.extend(labels.numpy().astype(int).tolist())

    return {
        "accuracy" : accuracy_score (all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall"   : recall_score   (all_labels, all_preds, zero_division=0),
        "f1_score" : f1_score       (all_labels, all_preds, zero_division=0),
    }
