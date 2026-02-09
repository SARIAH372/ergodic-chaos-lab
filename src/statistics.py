# src/statistics.py
import numpy as np

def observable(traj: np.ndarray, name: str) -> np.ndarray:
    if name == "x":
        return traj
    if name == "x2":
        return traj ** 2
    if name == "sin":
        return np.sin(2.0 * np.pi * traj)
    if name == "indicator":
        return (np.abs(traj - 0.5) < 0.05).astype(float)
    raise ValueError("Unknown observable")

def birkhoff_average(traj: np.ndarray, name: str) -> np.ndarray:
    f = observable(traj, name)
    return np.cumsum(f) / (np.arange(len(f)) + 1)

def autocorrelation(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.zeros(max_lag + 1, dtype=float)

    ac = np.zeros(max_lag + 1, dtype=float)
    ac[0] = 1.0
    for lag in range(1, max_lag + 1):
        ac[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)
    return ac
