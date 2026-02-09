# Ergodic Chaos Lab

## Overview
This repository provides a computational study of ergodic theory and chaotic dynamical systems. It implements numerical experiments for one-dimensional chaotic maps (logistic and tent) and the two-dimensional Hénon map, with a focus on invariant measures, Lyapunov exponents, ergodic averages, symbolic dynamics, transfer operators, and entropy.

The project emphasizes mathematically grounded diagnostics paired with reproducible simulation and interactive visualization.

---

## Mathematical Scope

### One-Dimensional Maps
- Logistic map
- Tent map

For these systems, the code studies:
- Empirical invariant measures
- Transfer operator approximation (Ulam discretization)
- Lyapunov exponents
- Ergodic averages (Birkhoff theorem, numerical)
- Entropy estimates:
  - Time-average entropy via log-derivative
  - Measure-based entropy via invariant density
- Symbolic dynamics:
  - Binary itineraries
  - Block entropies
  - Entropy-rate proxies
- Markov partition diagnostics for the tent map:
  - Shift conjugacy checks
  - Cylinder set occurrences

### Two-Dimensional System
- Hénon map

For the Hénon map, the code computes:
- Long-run attractor structure
- Full Lyapunov spectrum (λ₁, λ₂) via QR / Benettin method
- Determinant consistency check (λ₁ + λ₂ ≈ log|b|)
- ## Repository Structure
- 
- 
- 
- ergodic-chaos-lab/
│
├── app.py
├── requirements.txt
├── LICENSE
├── README.md
│
└── src/
├── init.py
├── maps.py
├── simulator.py
├── lyapunov.py
├── invariant.py
├── ulam.py
├── statistics.py
├── entropy.py
├── symbolic.py
├── markov.py
├── henon.py
└── lyapunov2d.py

---

## Running the Application

The project is designed as an interactive Streamlit application.

### Local execution
```bash
pip install -r requirements.txt
streamlit run app.py
Deployment

The application is compatible with Streamlit Community Cloud using app.py as the entry point.

Numerical Methods and Notes

Transfer operators are approximated using Ulam’s method with finite partitions.

Entropy estimates are numerical and should be interpreted as finite-sample approximations.

Symbolic dynamics results depend on partition choice and trajectory length.

Lyapunov exponents are computed using standard time-averaging (1D) and QR-based methods (2D).

These implementations are intended for exploration and illustration of theoretical concepts rather than formal proof.
License

This project is released under the MIT License. See the LICENSE file for details.

---

## Repository Structure

