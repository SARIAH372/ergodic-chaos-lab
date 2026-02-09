# src/maps.py
import numpy as np

def logistic_map(x: float, r: float) -> float:
    return r * x * (1.0 - x)

def logistic_derivative(x: float, r: float) -> float:
    return r * (1.0 - 2.0 * x)

def tent_map(x: float, s: float) -> float:
    return s * x if x < 0.5 else s * (1.0 - x)

def tent_derivative(x: float, s: float) -> float:
    return s if x < 0.5 else -s

def step_map(x: float, system: str, param: float) -> float:
    if system == "logistic":
        return logistic_map(x, param)
    if system == "tent":
        return tent_map(x, param)
    raise ValueError("Unknown system")

def step_derivative(x: float, system: str, param: float) -> float:
    if system == "logistic":
        return logistic_derivative(x, param)
    if system == "tent":
        return tent_derivative(x, param)
    raise ValueError("Unknown system")

def clip_unit(x: float, eps: float = 1e-12) -> float:
    return float(np.clip(x, eps, 1.0 - eps))
