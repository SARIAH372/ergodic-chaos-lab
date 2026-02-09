# src/lyapunov.py
import numpy as np
from .maps import step_derivative

def estimate_lyapunov(traj: np.ndarray, system: str, param: float, eps: float = 1e-15) -> float:
    derivs = np.array([abs(step_derivative(float(x), system, param)) for x in traj], dtype=float)
    derivs = np.maximum(derivs, eps)
    return float(np.mean(np.log(derivs)))

