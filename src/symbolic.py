# src/symbolic.py
import numpy as np

def itinerary_1d(traj: np.ndarray, partition: float = 0.5) -> np.ndarray:
    traj = np.asarray(traj, dtype=float)
    return (traj >= partition).astype(np.int8)

def block_counts(symbols: np.ndarray, k: int) -> dict:
    s = np.asarray(symbols, dtype=np.int8)
    n = len(s)
    if k <= 0 or n < k:
        return {}
    counts = {}
    for i in range(n - k + 1):
        block = 0
        for j in range(k):
            block = (block << 1) | int(s[i + j])
        counts[block] = counts.get(block, 0) + 1
    return counts

def block_entropy(symbols: np.ndarray, k: int) -> float:
    counts = block_counts(symbols, k)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()], dtype=float)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))

def entropy_rate_estimate(symbols: np.ndarray, k_max: int = 8) -> dict:
    H = []
    for k in range(1, k_max + 1):
        H.append(block_entropy(symbols, k))
    h_inc = [H[0]] + [H[k] - H[k - 1] for k in range(1, k_max)]
    return {"H": H, "h_inc": h_inc}

def kneading_sequence(symbols: np.ndarray, m: int = 64) -> str:
    m = min(m, len(symbols))
    return "".join("1" if b else "0" for b in symbols[:m])
