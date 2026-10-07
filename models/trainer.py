"""
models/trainer.py
-----------------
Training loop shared by all neural models.

- Loss: L1 on the original passenger scale, normalised by the mean training
  boardings. Minimising it is the same as minimising WMAPE, the headline
  metric. (An L1 loss on log-scaled targets fits the median in log space and
  systematically under-forecasts busy hours.)
- Early stopping and LR reduction on validation WMAPE.
- Deterministic seeding.
"""

import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn

from utils.metrics import compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    def __init__(self, model: nn.Module, scaler, edge_index=None, edge_weight=None,
                 lr: float = 1e-3, weight_decay: float = 1e-4, max_epochs: int = 60,
                 patience: int = 10, device: torch.device = None, verbose: bool = True):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.edge_index = edge_index.to(self.device) if edge_index is not None else None
        self.edge_weight = edge_weight.to(self.device) if edge_weight is not None else None
        self.scaler = scaler
        self.log_max = torch.tensor(scaler.log_max_, device=self.device).view(1, -1, 1)
        self.max_epochs, self.patience, self.verbose = max_epochs, patience, verbose
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=max(2, patience // 3))
        self.history = {"train_loss": [], "val_wmape": [], "lr": []}
        self.best_epoch = 0

    def _to_raw(self, y_scaled):
        return torch.expm1(y_scaled.clamp(max=2.0) * self.log_max)

    def _forward(self, x_hist, x_fut):
        return self.model(x_hist.to(self.device), x_fut.to(self.device),
                          self.edge_index, self.edge_weight)

    def fit(self, train_loader, val_loader):
        y_norm = float(self.scaler.inverse(train_loader.dataset.y.numpy()).mean())
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            self.model.train()
            total, count = 0.0, 0
            for x_hist, x_fut, y, _ in train_loader:
                y = y.to(self.device)
                y_hat = self._forward(x_hist, x_fut)
                loss = (self._to_raw(y_hat) - self._to_raw(y)).abs().mean() / y_norm
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                total += loss.item() * y.size(0)
                count += y.size(0)

            val_wmape = self.evaluate(val_loader)["WMAPE"]
            self.scheduler.step(val_wmape)
            self.history["train_loss"].append(total / count)
            self.history["val_wmape"].append(val_wmape)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            improved = val_wmape < best - 1e-3
            if improved:
                best, bad, self.best_epoch = val_wmape, 0, epoch
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
            if self.verbose:
                print(f"  epoch {epoch:03d} | train {total / count:.4f} | val WMAPE {val_wmape:6.2f}%"
                      f"{' *' if improved else ''} | {time.time() - t0:.1f}s")
            if bad >= self.patience:
                break

        self.model.load_state_dict(best_state)
        return self.history

    @torch.no_grad()
    def predict(self, loader) -> tuple[np.ndarray, np.ndarray]:
        """Returns (predictions in passengers [S, N, H], origins [S])."""
        self.model.eval()
        preds, origins = [], []
        for x_hist, x_fut, _, o in loader:
            preds.append(self._forward(x_hist, x_fut).cpu().numpy())
            origins.append(o.numpy())
        y = self.scaler.inverse(np.concatenate(preds))
        return np.maximum(y, 0), np.concatenate(origins)

    def evaluate(self, loader) -> dict:
        pred, _ = self.predict(loader)
        true = self.scaler.inverse(loader.dataset.y.numpy())
        return compute_metrics(true, pred)
