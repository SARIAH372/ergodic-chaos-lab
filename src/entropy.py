# src/entropy.py
import numpy as np
from .maps import step_derivative

def ks_entropy_from_measure(
    bin_centers: np.ndarray,
    bin_probs: np.ndarray,
    system: str,
    param: float,
    eps: float = 1e-15,
) -> float:
    bin_probs = np.asarray(bin_probs, dtype=float)
    bin_probs = bin_probs / max(bin_probs.sum(), eps)

    logs = []
    for x in bin_centers:
        d = abs(step_derivative(float(x), system, param))
        logs.append(np.log(max(d, eps)))
    logs = np.asarray(logs, dtype=float)

    return float(np.sum(bin_probs * logs))

def ks_entropy_from_trajectory(traj: np.ndarray, system: str, param: float, eps: float = 1e-15) -> float:
    derivs = np.array([abs(step_derivative(float(x), system, param)) for x in traj], dtype=float)
    derivs = np.maximum(derivs, eps)
    return float(np.mean(np.log(derivs)))
