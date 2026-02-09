# src/markov.py
import numpy as np

def tent_itinerary(x: float, s: float = 2.0, k: int = 16) -> np.ndarray:
    bits = np.zeros(k, dtype=np.int8)
    xi = float(x)
    for i in range(k):
        bits[i] = 0 if xi < 0.5 else 1
        xi = (s * xi) if xi < 0.5 else (s * (1.0 - xi))
        xi = min(max(xi, 0.0), 1.0)
    return bits

def shift_property_check(xs: np.ndarray, s: float = 2.0, k: int = 16) -> float:
    xs = np.asarray(xs, dtype=float)
    ok = 0
    total = 0
    for x in xs:
        b = tent_itinerary(float(x), s=s, k=k)
        x1 = (s * x) if x < 0.5 else (s * (1.0 - x))
        b1 = tent_itinerary(float(x1), s=s, k=k)
        total += 1
        if np.array_equal(b1[:k-1], b[1:]):
            ok += 1
    return ok / total if total > 0 else 0.0

def word_to_bits(word: str) -> np.ndarray:
    word = word.strip()
    if not word or any(c not in "01" for c in word):
        raise ValueError("Word must be a non-empty binary string like 01011.")
    return np.array([1 if c == "1" else 0 for c in word], dtype=np.int8)

def cylinder_mask(traj: np.ndarray, word_bits: np.ndarray, partition: float = 0.5) -> np.ndarray:
    traj = np.asarray(traj, dtype=float)
    k = len(word_bits)
    if len(traj) < k:
        return np.zeros(len(traj), dtype=bool)
    sym = (traj >= partition).astype(np.int8)
    mask = np.zeros(len(traj), dtype=bool)
    for i in range(len(traj) - k + 1):
        if np.array_equal(sym[i:i+k], word_bits):
            mask[i] = True
    return mask
