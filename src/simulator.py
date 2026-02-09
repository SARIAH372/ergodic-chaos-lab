# src/simulator.py
import numpy as np
from .maps import step_map, clip_unit

def simulate_trajectory(
    system: str,
    param: float,
    x0: float,
    n_steps: int,
    burn_in: int,
    seed: int = 0,
    jitter: float = 1e-12,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = clip_unit(x0 + jitter * rng.standard_normal())

    total = burn_in + n_steps
    traj = np.empty(total, dtype=float)
    traj[0] = x

    for t in range(1, total):
        traj[t] = clip_unit(step_map(traj[t - 1], system, param))

    return traj[burn_in:].copy()
