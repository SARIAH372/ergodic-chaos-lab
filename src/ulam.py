# src/ulam.py
import numpy as np

def ulam_markov_matrix(traj: np.ndarray, bins: int = 200):
    traj = np.asarray(traj, dtype=float)
    traj = np.clip(traj, 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(traj, edges) - 1, 0, bins - 1)

    P = np.zeros((bins, bins), dtype=float)
    a = idx[:-1]
    b = idx[1:]
    for i_from, i_to in zip(a, b):
        P[i_from, i_to] += 1.0

    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    P = P / row_sums
    return P, edges

def stationary_distribution(P: np.ndarray, tol: float = 1e-12, max_iter: int = 100_000) -> np.ndarray:
    n = P.shape[0]
    pi = np.full(n, 1.0 / n, dtype=float)

    for _ in range(max_iter):
        pi_next = pi @ P
        if np.linalg.norm(pi_next - pi, ord=1) < tol:
            pi = pi_next
            break
        pi = pi_next

    s = pi.sum()
    return pi / s if s > 0 else pi

def spectral_gap_proxy(P: np.ndarray) -> float:
    vals = np.linalg.eigvals(P.T)
    mags = np.sort(np.abs(vals))[::-1]
    if len(mags) < 2:
        return 0.0
    return float(max(0.0, 1.0 - mags[1]))
