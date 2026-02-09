# src/invariant.py
import numpy as np

def empirical_invariant_density(traj: np.ndarray, bins: int = 200):
    hist, edges = np.histogram(traj, bins=bins, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist, edges

def analytic_logistic_r4_density(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    x = np.clip(x, eps, 1 - eps)
    return 1.0 / (np.pi * np.sqrt(x * (1 - x)))
