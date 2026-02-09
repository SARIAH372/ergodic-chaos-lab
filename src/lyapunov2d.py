# src/lyapunov2d.py
import numpy as np
from .henon import henon_step, henon_jacobian

def lyapunov_spectrum_henon(
    a: float,
    b: float,
    x0: float,
    y0: float,
    n_steps: int,
    burn_in: int,
    eps: float = 1e-18,
):
    x, y = float(x0), float(y0)
    for _ in range(burn_in):
        x, y = henon_step(x, y, a, b)

    Q = np.eye(2, dtype=float)
    sum_logs = np.zeros(2, dtype=float)

    for _ in range(n_steps):
        J = henon_jacobian(x, y, a, b)
        Z = J @ Q
        Q, R = np.linalg.qr(Z)
        diag = np.maximum(np.abs(np.diag(R)), eps)
        sum_logs += np.log(diag)
        x, y = henon_step(x, y, a, b)

    lambdas = sum_logs / max(n_steps, 1)
    lam1, lam2 = float(lambdas[0]), float(lambdas[1])
    if lam2 > lam1:
        lam1, lam2 = lam2, lam1
    return lam1, lam2
