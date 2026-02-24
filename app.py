import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.simulator import simulate_trajectory
from src.lyapunov import estimate_lyapunov
from src.invariant import empirical_invariant_density, analytic_logistic_r4_density
from src.ulam import ulam_markov_matrix, stationary_distribution, spectral_gap_proxy
from src.statistics import birkhoff_average, observable, autocorrelation
from src.entropy import ks_entropy_from_trajectory, ks_entropy_from_measure

from src.symbolic import itinerary_1d, entropy_rate_estimate, kneading_sequence
from src.henon import simulate_henon
from src.lyapunov2d import lyapunov_spectrum_henon

from src.markov import word_to_bits, cylinder_mask, shift_property_check


st.set_page_config(page_title="Ergodic Chaos Lab", page_icon="🌀", layout="wide")
st.title("🌀 Ergodic Chaos Lab")
st.caption("Invariant measures, Lyapunov exponents, ergodic averages, symbolic dynamics, and 2D chaos diagnostics.")

tab1, tab2 = st.tabs(["1D Maps", "2D Hénon Map"])

# ============================================================
# TAB 1: 1D MAPS
# ============================================================
with tab1:
    st.sidebar.header("1D map")

    system = st.sidebar.selectbox("Map", ["logistic", "tent"], key="map_1d")
    if system == "logistic":
        param = st.sidebar.slider("r", 3.5, 4.0, 4.0, step=0.001, key="r_1d")
    else:
        param = st.sidebar.slider("s", 1.0, 2.0, 2.0, step=0.001, key="s_1d")

    st.sidebar.header("Simulation")
    n_steps = st.sidebar.slider("Steps (post burn-in)", 5_000, 200_000, 50_000, step=5_000, key="n_1d")
    burn_in = st.sidebar.slider("Burn-in", 0, 50_000, 5_000, step=1_000, key="burn_1d")
    x0 = st.sidebar.slider("Initial x₀", 0.0001, 0.9999, 0.1234, step=0.0001, key="x0_1d")
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=10_000_000, value=7, step=1, key="seed_1d")

    st.sidebar.header("Discretization")
    bins_hist = st.sidebar.slider("Histogram bins", 30, 400, 160, step=10, key="bins_hist")
    ulam_bins = st.sidebar.slider("Ulam bins", 50, 300, 160, step=10, key="ulam_bins")
    max_lag = st.sidebar.slider("Autocorr max lag", 10, 500, 120, step=10, key="max_lag")

    obs_name = st.sidebar.selectbox(
        "Observable f (Birkhoff)",
        ["x", "x2", "sin", "indicator"],
        format_func=lambda z: {
            "x": "f(x)=x",
            "x2": "f(x)=x^2",
            "sin": "f(x)=sin(2πx)",
            "indicator": "1{|x-0.5|<0.05}",
        }[z],
        key="obs",
    )

    with st.spinner("Simulating trajectory..."):
        traj = simulate_trajectory(system, float(param), float(x0), int(n_steps), int(burn_in), int(seed))

    lyap = estimate_lyapunov(traj, system, float(param))
    h_time = ks_entropy_from_trajectory(traj, system, float(param))

    m1, m2, m3 = st.columns(3)
    m1.metric("Lyapunov exponent λ", f"{lyap:.6f}")
    m2.metric("Entropy estimate h (time-average)", f"{h_time:.6f} nats")
    if system == "tent" and abs(param - 2.0) < 1e-6:
        m3.metric("Reference log(2)", f"{np.log(2):.6f} nats")
    elif system == "logistic" and abs(param - 4.0) < 1e-6:
        m3.metric("Reference log(2)", f"{np.log(2):.6f} nats")
    else:
        m3.metric("Reference", "—")

    st.markdown("---")

    cA, cB = st.columns(2, gap="large")
    with cA:
        st.subheader("Trajectory")
        show_n = min(len(traj), 3000)
        fig = plt.figure()
        plt.plot(np.arange(show_n), traj[:show_n], linewidth=1.0)
        plt.xlabel("n")
        plt.ylabel("x_n")
        plt.title("Trajectory (first segment)")
        st.pyplot(fig, use_container_width=True)

    with cB:
        st.subheader("Autocorrelation")
        g = observable(traj, "sin") if obs_name == "sin" else traj
        ac = autocorrelation(g, int(max_lag))
        fig = plt.figure()
        plt.plot(np.arange(len(ac)), ac, linewidth=1.2)
        plt.xlabel("lag")
        plt.ylabel("corr")
        plt.title("Autocorrelation vs lag")
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Invariant measure (empirical)")
    centers, dens, edges = empirical_invariant_density(traj, bins=int(bins_hist))
    fig = plt.figure()
    plt.plot(centers, dens, linewidth=1.5, label="Empirical density")
    if system == "logistic" and abs(param - 4.0) < 1e-6:
        xx = np.linspace(0.0, 1.0, 1200)
        plt.plot(xx, analytic_logistic_r4_density(xx), linewidth=2.0, label="Analytic density (r=4)")
    elif system == "tent" and abs(param - 2.0) < 1e-6:
        plt.axhline(1.0, linewidth=2.0, label="Uniform density")
    plt.title("Invariant density")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Transfer operator (Ulam)")
    P, u_edges = ulam_markov_matrix(traj, bins=int(ulam_bins))
    pi = stationary_distribution(P)  # bin mass
    gap = spectral_gap_proxy(P)

    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    bin_w = u_edges[1] - u_edges[0]
    pi_density = pi / bin_w

    h_ulam = ks_entropy_from_measure(u_centers, pi, system, float(param))

    u1, u2, u3 = st.columns(3)
    u1.metric("Spectral gap proxy", f"{gap:.5f}")
    u2.metric("Entropy estimate h (Ulam integral)", f"{h_ulam:.6f} nats")
    u3.metric("|h_time - h_ulam|", f"{abs(h_time - h_ulam):.6f}")

    fig = plt.figure()
    plt.plot(u_centers, pi_density, linewidth=2.0, label="Ulam stationary density")
    plt.plot(centers, dens, linewidth=1.2, alpha=0.8, label="Empirical density")
    if system == "logistic" and abs(param - 4.0) < 1e-6:
        xx = np.linspace(0.0, 1.0, 1200)
        plt.plot(xx, analytic_logistic_r4_density(xx), linewidth=2.0, label="Analytic density (r=4)")
    plt.title("Invariant density comparison")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Ergodic average (Birkhoff)")
    runmean = birkhoff_average(traj, obs_name)
    final_time_avg = float(runmean[-1])
    f_centers = observable(u_centers, obs_name)
    space_avg_proxy = float(np.sum(pi * f_centers))

    b1, b2, b3 = st.columns(3)
    b1.metric("Final time average", f"{final_time_avg:.6f}")
    b2.metric("Space average proxy", f"{space_avg_proxy:.6f}")
    b3.metric("Absolute gap", f"{abs(final_time_avg - space_avg_proxy):.6f}")

    fig = plt.figure()
    plt.plot(runmean, linewidth=1.2, label="Running time average")
    plt.axhline(space_avg_proxy, linewidth=2.0, label="Space average proxy")
    plt.title("Time average vs space average proxy")
    plt.xlabel("n")
    plt.ylabel("(1/n) Σ f(x_k)")
    plt.legend()
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Symbolic dynamics")
    partition = st.slider("Partition point c", 0.1, 0.9, 0.5, step=0.01)
    k_max = st.slider("Max block length k", 2, 14, 8, step=1)

    symbols = itinerary_1d(traj, float(partition))
    er = entropy_rate_estimate(symbols, int(k_max))
    H = er["H"]
    h_inc = er["h_inc"]

    st.code(kneading_sequence(symbols, 64))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = plt.figure()
        plt.plot(np.arange(1, len(H) + 1), H, marker="o")
        plt.title("Block entropy H_k (nats)")
        plt.xlabel("k")
        plt.ylabel("H_k")
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig = plt.figure()
        plt.plot(np.arange(1, len(h_inc) + 1), h_inc, marker="o")
        plt.title("Increment h_k = H_k - H_{k-1}")
        plt.xlabel("k")
        plt.ylabel("h_k")
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Markov partition diagnostics (tent map)")
    if system != "tent" or abs(param - 2.0) > 1e-3:
        st.write("Diagnostics shown for tent map near s=2.")
    else:
        sample_n = st.slider("Sample points", 50, 2000, 400, step=50)
        xs_sample = traj[:min(len(traj), int(sample_n))]
        frac_ok = shift_property_check(xs_sample, s=float(param), k=16)
        st.metric("Shift property rate", f"{frac_ok:.3f}")

        word = st.text_input("Binary word", value="01011")
        try:
            bits = word_to_bits(word)
            mask = cylinder_mask(traj, bits, partition=0.5)
            count = int(mask.sum())
            st.write(f"Block-start count: {count}")

            idxs = np.where(mask)[0]
            show = idxs[:10]
            st.write("First indices:", show.tolist() if len(show) else "None")

            fig = plt.figure()
            nplot = min(5000, len(traj))
            plt.plot(traj[:nplot], linewidth=1.0, label="x_n")
            if len(show) > 0:
                pts = show[show < nplot]
                plt.scatter(pts, traj[pts], s=30, label="block start")
            plt.title("Cylinder occurrences on trajectory snippet")
            plt.xlabel("n")
            plt.ylabel("x_n")
            plt.legend()
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(str(e))

    st.markdown("---")

    st.subheader("Export (1D)")
    df = pd.DataFrame({"x_n": traj, "symbol": symbols})
    st.dataframe(df.head(10), use_container_width=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="trajectory_1d.csv", mime="text/csv")


# ============================================================
# TAB 2: HÉNON MAP (NO SRB)
# ============================================================
with tab2:
    st.sidebar.header("2D Hénon map")

    a = st.sidebar.slider("a", 0.5, 2.0, 1.4, step=0.01, key="henon_a")
    b = st.sidebar.slider("b", 0.05, 0.6, 0.3, step=0.01, key="henon_b")

    st.sidebar.header("Simulation")
    n2 = st.sidebar.slider("Steps (post burn-in)", 5_000, 200_000, 50_000, step=5_000, key="henon_n")
    burn2 = st.sidebar.slider("Burn-in", 0, 100_000, 10_000, step=1_000, key="henon_burn")
    x0_2 = st.sidebar.number_input("x0", value=0.1, key="henon_x0")
    y0_2 = st.sidebar.number_input("y0", value=0.1, key="henon_y0")

    st.subheader("Hénon map")

    with st.spinner("Simulating..."):
        xs, ys = simulate_henon(float(a), float(b), float(x0_2), float(y0_2), int(n2), int(burn2))

    with st.spinner("Estimating Lyapunov spectrum..."):
        lam1, lam2 = lyapunov_spectrum_henon(
            float(a), float(b), float(x0_2), float(y0_2),
            n_steps=min(int(n2), 80000),
            burn_in=int(burn2)
        )

    det_check = np.log(abs(float(b)))

    m1, m2, m3 = st.columns(3)
m1.metric("λ1", f"{lam1:.6f}")
m2.metric("λ2", f"{lam2:.6f}")
m3.metric("λ1+λ2 ; log|b|", f"{(lam1+lam2):.6f} ; {det_check:.6f}")

    fig = plt.figure()
    plt.scatter(xs[:30000], ys[:30000], s=1)
    plt.title("Attractor (scatter)")
    plt.xlabel("x")
    plt.ylabel("y")
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Export (Hénon)")
    df2 = pd.DataFrame({"x": xs, "y": ys})
    st.dataframe(df2.head(10), use_container_width=True)
    csv2 = df2.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv2, file_name="henon_trajectory.csv", mime="text/csv")


