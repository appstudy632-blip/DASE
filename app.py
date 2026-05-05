# app.py
# Streamlit app for Fair Division with Storage (Infinite or Finite)

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import pulp
import math


# ============================
# Parsing & validation
# ============================

def parse_matrix(text: str) -> np.ndarray:
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    data = []
    for r in rows:
        r = r.replace(",", " ")
        data.append([float(x) for x in r.split()])
    return np.array(data, dtype=float)


def parse_vector(text: str) -> np.ndarray:
    arr = parse_matrix(text)
    if arr.shape[0] == 1:
        return arr.flatten()
    if arr.shape[1] == 1:
        return arr.flatten()
    raise ValueError("Supply must be a single row or column.")


def parse_capacity(text: str) -> float | None:
    """Parse capacity input. Returns None for infinite, float for finite."""
    text = text.strip().upper()
    if text in ["INF", "INFINITE", "∞", ""]:
        return None
    try:
        C = float(text)
        if C < 0:
            raise ValueError("Capacity must be non-negative.")
        return C
    except ValueError:
        raise ValueError(
            f"Invalid capacity: '{text}'. Use a non-negative number or 'INF'."
        )


def validate(d: np.ndarray, S: np.ndarray, C: float = None):
    if d.ndim != 2:
        raise ValueError("Demands must be a matrix.")
    if S.ndim != 1:
        raise ValueError("Supply must be a vector.")
    if d.shape[1] != len(S):
        raise ValueError("Supply length must equal number of time steps.")
    if np.any(d < 0) or np.any(S < 0):
        raise ValueError("All values must be non-negative.")

    row_sums = d.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("Each agent must have positive total demand.")
    if not np.allclose(row_sums, row_sums[0]):
        raise ValueError("All demand rows must sum to the same total.")
    if C is not None and C < 0:
        raise ValueError("Capacity must be non-negative.")


# ============================
# DA / DA-Finite
# ============================

def da_single_agent(d_i: np.ndarray, S: np.ndarray, n: int, C: float = None):
    """
    Agent i maximizes utility from S_i = S/n with optional capacity C_i = C/n.
    """
    T = len(S)
    Si = S / n

    prob = pulp.LpProblem("DA", pulp.LpMaximize)

    w = pulp.LpVariable.dicts("w", range(T), lowBound=0)
    y = pulp.LpVariable.dicts("y", range(T), lowBound=0)
    beta = pulp.LpVariable("beta", lowBound=0)

    prob += beta

    # Feasibility constraints
    prob += w[0] + y[0] <= Si[0]
    for t in range(1, T):
        prob += w[t] + y[t] <= Si[t] + y[t - 1]

    # Capacity constraints (finite storage only)
    if C is not None:
        C_i = C / n
        for t in range(T):
            prob += y[t] <= C_i

    # Tightness constraints
    for t in range(T):
        if d_i[t] > 0:
            prob += beta * d_i[t] <= w[t]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return np.zeros(T), 0.0

    w_val = np.array([pulp.value(w[t]) for t in range(T)])
    beta_val = float(pulp.value(beta))
    return w_val, beta_val


def DA(demands: np.ndarray, S: np.ndarray, C: float = None):
    """DA or DA-Finite depending on C."""
    n, T = demands.shape
    alloc = np.zeros((n, T))
    beta = np.zeros(n)

    for i in range(n):
        alloc[i], beta[i] = da_single_agent(demands[i], S, n, C)

    return alloc, beta


# ============================
# SE / SE-Finite
# ============================

def SE(demands: np.ndarray, S_step: np.ndarray) -> np.ndarray:
    """
    Sequential Equalizing (works for both infinite and finite storage).
    For finite storage, capacity is implicitly satisfied.
    """
    n, T = demands.shape
    w = np.zeros((n, T), dtype=float)

    cumS = np.cumsum(S_step).astype(float)
    if cumS.min() < -1e-9:
        raise ValueError("Prefix-infeasible supply.")

    active = set(range(n))
    p = 0  # frontier

    while active and p < T:
        cumAlloc = np.cumsum(w.sum(axis=0))
        slack = cumS - cumAlloc

        base = 0.0 if p == 0 else slack[p - 1]

        best_t = None
        best_ratio = math.inf

        for t in range(p, T):
            denom = sum(demands[i, p : t + 1].sum() for i in active)
            if denom <= 0:
                continue

            seg_slack = slack[t] - base
            if seg_slack < -1e-12:
                continue

            ratio = seg_slack / denom
            if ratio < best_ratio:
                best_ratio = ratio
                best_t = t

        if best_t is None:
            break

        Delta = best_ratio

        for i in active:
            w[i, p:] += Delta * demands[i, p:]

        p = best_t + 1
        active = {i for i in active if not np.any(demands[i, :p] > 0)}

    return w


# ============================
# DASE / DASE-Finite
# ============================

def residual_prefix_budget(S: np.ndarray, w_DA: np.ndarray) -> np.ndarray:
    """Cumulative residual supply after DA."""
    S = S.astype(float)
    A = w_DA.sum(axis=0).astype(float)
    Shat_rem = np.cumsum(S) - np.cumsum(A)

    if Shat_rem.min() < -1e-9:
        t_bad = int(np.argmin(Shat_rem))
        raise ValueError(f"DA allocation is prefix-infeasible at t={t_bad}.")
    return Shat_rem


def prefix_budget_to_step_supply(Shat: np.ndarray) -> np.ndarray:
    """Convert cumulative to per-step supply."""
    Shat = Shat.astype(float)
    S_step = np.empty_like(Shat)
    S_step[0] = Shat[0]
    S_step[1:] = Shat[1:] - Shat[:-1]
    return S_step


def DASE(demands: np.ndarray, S: np.ndarray, w_DA: np.ndarray) -> np.ndarray:
    """DASE / DASE-Finite = DA + SE on residual."""
    Shat_rem = residual_prefix_budget(S, w_DA)
    S_rem = prefix_budget_to_step_supply(Shat_rem)
    w_SE = SE(demands, S_rem)
    return w_DA + w_SE


def alpha_from_alloc(w: np.ndarray, d: np.ndarray) -> float:
    """Leontief utility."""
    mask = d > 0
    if not np.any(mask):
        return float("inf")
    return float(np.min(w[mask] / d[mask]))


# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="Fair Division with Storage", layout="wide")
st.title("Fair Division Over Time with Storage")

st.markdown("""
### About

This tool implements fair division mechanisms for allocating a divisible resource over time with storage:

- **DA / DA-Finite:** Each agent independently optimizes from an equal 1/n share
- **SE / SE-Finite:** Sequential equalizing distributes supply proportionally
- **DASE / DASE-Finite:** Combines DA + SE for Pareto efficiency

**Storage Capacity:**
- **Infinite (C = INF):** No storage limit
- **Finite (C = number):** Total capacity C, each agent gets C/n

---
""")

col1, col2 = st.columns([2, 1])

with col1:
    dem_text = st.text_area(
        "**Demands** (rows = agents, columns = time steps)",
        placeholder="Example:\n1 2 1 0 0\n3 1 0 0 0\n0 2 1 0 1",
        height=140,
    )

with col2:
    sup_text = st.text_area(
        "**Supply** (per time step)",
        placeholder="Example:\n36 36 36 42 50",
        height=140,
    )

capacity_text = st.text_input(
    "**Storage Capacity C**",
    value="INF",
    help="Enter 'INF' for infinite storage, or a number for finite capacity.",
)

st.info("ℹ️ All demand rows must sum to the same value.")

if st.button("⚡ Compute Allocations", type="primary", use_container_width=True):
    try:
        # Parse
        demands = parse_matrix(dem_text)
        supply = parse_vector(sup_text)
        C = parse_capacity(capacity_text)
        validate(demands, supply, C)

        # Compute
        w_DA, beta_DA = DA(demands, supply, C)
        w_SE = SE(demands, supply)
        w_DASE = DASE(demands, supply, w_DA)

        # Utilities
        alpha_DA = [alpha_from_alloc(w_DA[i], demands[i]) for i in range(len(demands))]
        alpha_SE = [alpha_from_alloc(w_SE[i], demands[i]) for i in range(len(demands))]
        alpha_DASE = [
            alpha_from_alloc(w_DASE[i], demands[i]) for i in range(len(demands))
        ]

        # Format
        n_agents, n_times = demands.shape
        idx = [f"Agent {i+1}" for i in range(n_agents)]
        cols = [f"t={t+1}" for t in range(n_times)]

        mode_label = "Infinite Storage" if C is None else f"Finite Storage (C = {C})"
        suffix = "" if C is None else "-Finite"

        st.success(f"✓ **{mode_label}**")

        tab1, tab2, tab3 = st.tabs([f"DA{suffix}", f"SE{suffix}", f"DASE{suffix}"])

        with tab1:
            st.markdown(f"### DA{suffix} Allocation")
            st.dataframe(
                pd.DataFrame(w_DA, index=idx, columns=cols).style.format("{:.3f}"),
                use_container_width=True,
            )
            st.dataframe(
                pd.DataFrame({"Agent": idx, "Utility (α)": alpha_DA}),
                use_container_width=True,
                hide_index=True,
            )

        with tab2:
            st.markdown(f"### SE{suffix} Allocation")
            st.dataframe(
                pd.DataFrame(w_SE, index=idx, columns=cols).style.format("{:.3f}"),
                use_container_width=True,
            )
            st.dataframe(
                pd.DataFrame({"Agent": idx, "Utility (α)": alpha_SE}),
                use_container_width=True,
                hide_index=True,
            )

        with tab3:
            st.markdown(f"### DASE{suffix} Allocation")
            st.dataframe(
                pd.DataFrame(w_DASE, index=idx, columns=cols).style.format("{:.3f}"),
                use_container_width=True,
            )
            st.dataframe(
                pd.DataFrame({"Agent": idx, "Utility (α)": alpha_DASE}),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("🔍 Show residual supply after DA"):
                Shat_rem = residual_prefix_budget(supply, w_DA)
                S_rem = prefix_budget_to_step_supply(Shat_rem)
                st.write("**Cumulative residual Ŝ_rem:**", Shat_rem)
                st.write("**Per-step residual S':**", S_rem)

    except Exception as e:
        st.error(f"❌ {str(e)}")

st.caption("CATS model for fair division over time with storage")
