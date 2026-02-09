# src/henon.py
import numpy as np

def henon_step(x: float, y: float, a: float, b: float):
    xn = 1.0 - a * x * x + y
    yn = b * x
    return float(xn), float(yn)

def henon_jacobian(x: float, y: float, a: float, b: float) -> np.ndarray:
    return np.array([[-2.0 * a * x, 1.0],
                     [b,            0.0]], dtype=float)

def simulate_henon(a: float, b: float, x0: float, y0: float, n_steps: int, burn_in: int):
    total = burn_in + n_steps
    xs = np.empty(total, dtype=float)
    ys = np.empty(total, dtype=float)
    xs[0] = x0
    ys[0] = y0
    x, y = x0, y0
    for t in range(1, total):
        x, y = henon_step(x, y, a, b)
        xs[t] = x
        ys[t] = y
    return xs[burn_in:], ys[burn_in:]
