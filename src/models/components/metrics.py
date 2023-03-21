from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torchmetrics


class CIndex(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self._value: float = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        summ = 0.0
        pair = 0

        for i in range(1, len(target)):
            for j in range(0, i):
                if i is not j:
                    if target[i] > target[j]:
                        pair += 1
                        summ += 1 * (preds[i] > preds[j]) + 0.5 * (preds[i] == preds[j])

        if pair != 0:
            self._value = float(summ / pair)
        else:
            self._value = 0.0

    def compute(self) -> torch.Tensor:
        """Computes the final statistics."""
        return torch.tensor(self._value)

    def reset(self) -> None:
        """Resets current statistics."""
        self._value = 0.0


def r_squared_error(y_obs: npt.ArrayLike[float], y_pred: npt.ArrayLike[float]) -> float:
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y) for y in y_obs]
    y_pred_mean = [np.mean(y) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))
