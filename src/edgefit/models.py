"""Candidate models: three classical baselines and a small PyTorch MLP."""

import copy
import time

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from edgefit.config import (
    MLP_BATCH_SIZE,
    MLP_DROPOUT,
    MLP_EPOCHS,
    MLP_HIDDEN,
    MLP_LR,
    MLP_WEIGHT_DECAY,
    RANDOM_STATE,
)


def classical_models(n_estimators: int = 300) -> dict:
    """Fixed, untuned configurations so the first comparison stays transparent."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2500, random_state=RANDOM_STATE)
        ),
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


class MLP(nn.Module):
    """n_in -> 128 + ReLU -> Dropout -> 64 + ReLU -> 6 logits."""

    def __init__(self, n_in: int = 561, n_out: int = 6, hidden=MLP_HIDDEN, dropout: float = MLP_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], n_out),
        )

    def forward(self, x):
        return self.net(x)


def to_tensor(X) -> torch.Tensor:
    return torch.tensor(np.asarray(X), dtype=torch.float32)


def predict_mlp(model: nn.Module, X, batch_size: int = 512) -> np.ndarray:
    """Predicted activity ids (1-6)."""
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(to_tensor(X)), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        pred = torch.cat([model(xb.to(device)).argmax(dim=1).cpu() for (xb,) in loader]).numpy()
    return pred + 1


def train_mlp(X_tr, y_tr, X_val, y_val, epochs: int = MLP_EPOCHS, device: str | None = None):
    """Train with Adam and keep the epoch with the best validation Macro F1.

    Labels are activity ids 1-6 (shifted to 0-5 internally). Returns the model,
    a per-epoch history table and the total training time in seconds.
    """
    torch.manual_seed(RANDOM_STATE)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(
        TensorDataset(to_tensor(X_tr), torch.tensor(np.asarray(y_tr) - 1, dtype=torch.long)),
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    model = MLP(n_in=X_tr.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)

    best_f1, best_state, history = -1.0, None, []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(xb)
        val_f1 = f1_score(y_val, predict_mlp(model, X_val), average="macro")
        history.append({"epoch": epoch, "train_loss": loss_sum / len(loader.dataset), "val_macro_f1": val_f1})
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())
    train_time = time.perf_counter() - t0

    model.load_state_dict(best_state)
    return model.cpu().eval(), pd.DataFrame(history), train_time
