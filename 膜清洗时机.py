# -*- coding: utf-8 -*-
"""Backwashing demonstration with explicitly constrained mechanism decomposition."""
from pathlib import Path
import io
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import brentq, minimize


VERSION = "cake-constrained-2026-11-21"

TITLE = "Membrane Fouling and Cleaning Insight Platform"

PRESETS = {
    "HA": "HAdata1.csv",
    "BSA": "BSAdata1.csv",
    "SA": "SAdata1.csv",
    "Mixture": "Mixturedata1.csv",
}

RAW_BASE = (
    "https://raw.githubusercontent.com/Zhangguosen123/"
    "Membrane-Fouling-and-Cleaning-Insight-Platform/main/"
)

NAMES = [
    "Standard Blocking",
    "Complete Blocking",
    "Intermediate Blocking",
    "Cake Filtration",
]

COLORS = ["#90b4ce", "#f6bd60", "#e58f8f", "#71b59b"]

# Parameter order: Ks, Kc, Kb, Ki, a, b
BOUNDS = [(0.0, 1.0)] * 4 + [(0.05, 1.0)] * 2

LOW = np.array([bound[0] for bound in BOUNDS])
HIGH = np.array([bound[1] for bound in BOUNDS])

EPS = 1e-12
MIN_POINTS = 5

GA_POP = 60
GA_GEN = 100
GA_MUT = 0.10
GA_ELITE = 2

SEEDS = (42, 137, 2026)

# Cake contribution must exceed each other contribution.
DOMINANCE_MARGIN = 1e-6

CONSTRAINT_TEXT = (
    "Constrained fit: cake-filtration contribution is required to exceed "
    "each of the other three contributions. This is an imposed model prior."
)


# ==============================================================================
# Model and mechanism contributions
# ==============================================================================

def log_flux_ratio(p, t, j0):
    ks, kc, kb, ki, a, b = p

    return (
        -2 * a * np.log1p(5 * ks * j0 * t)
        -10 * b * kb * t
        -(1 - b) * np.log1p(10 * ki * j0 * t)
        -(1 - a) / 2 * np.log1p(20 * kc * j0 * j0 * t)
    )


def model(p, t, j0):
    log_ratio = log_flux_ratio(p, t, j0)
    return j0 * np.exp(np.maximum(log_ratio, -700))


def shares(p, t_end, j0):
    """
    Exact cumulative log-flux-decay contributions.

    Di = -integral(si dt)
    eta_i = Di / sum(Di)

    These definitions do not multiply si by J(t).
    """
    ks, kc, kb, ki, a, b = p

    d = np.array([
        2 * a * np.log1p(5 * ks * j0 * t_end),
        10 * b * kb * t_end,
        (1 - b) * np.log1p(10 * ki * j0 * t_end),
        (1 - a) / 2 * np.log1p(20 * kc * j0 * j0 * t_end),
    ])

    total = float(d.sum())

    if total <= EPS:
        return np.full(4, np.nan)

    return d / total


def cake_constraints(p, duration, j0):
    eta = shares(p, duration, j0)

    if not np.all(np.isfinite(eta)):
        return np.full(3, -1.0)

    return eta[3] - eta[:3] - DOMINANCE_MARGIN


def feasible(p, duration, j0):
    return (
        np.all(np.isfinite(p))
        and np.all(p >= LOW)
        and np.all(p <= HIGH)
        and np.all(cake_constraints(p, duration, j0) >= 0.0)
    )


def metrics(y, yh):
    mse = float(np.mean((y - yh) ** 2))
    spread = float(np.ptp(y))
    variance_sum = float(np.sum((y - y.mean()) ** 2))

    floor = max(
        1e-8,
        0.05 * float(np.median(np.abs(y))),
    )

    return {
        "R2": (
            1 - float(np.sum((y - yh) ** 2)) / variance_sum
            if variance_sum > EPS else np.nan
        ),
        "NRMSE": (
            np.sqrt(mse) / spread
            if spread > EPS else np.nan
        ),
        "MAPE": float(
            np.mean(
                np.abs(y - yh) /
                np.maximum(np.abs(y), floor)
            )
        ),
    }


# ==============================================================================
# Cake-dominant constrained fitting
# ==============================================================================

def fit_curve(t, y):
    """
    Search feasible cake-dominant solutions using multiple GA seeds,
    then refine candidates with constrained optimization.

    Candidate ranking minimizes normalized MSE, equivalent to minimizing
    NRMSE for the same observations.
    """
    j0 = float(y[0])
    duration = float(t[-1])
    spread = float(np.ptp(y))

    if duration <= 0 or spread <= EPS or y[-1] >= j0:
        raise ValueError(
            "A positive-duration, declining flux window is required; "
            "mechanism shares are not assigned to flat or recovering windows."
        )

    scale = max(spread, EPS)

    def loss(p):
        yh = model(p, t, j0)
        return float(np.mean(((y - yh) / scale) ** 2))

    candidates = []

    def keep(p, origin):
        p = np.clip(
            np.asarray(p, dtype=float),
            LOW,
            HIGH,
        )

        if feasible(p, duration, j0):
            candidates.append((
                loss(p),
                p.copy(),
                origin,
            ))

    def repair(p):
        """
        Construct feasible search candidates.

        This changes candidate parameters during constrained optimization;
        it does not overwrite displayed mechanism percentages.
        """
        p = np.clip(p.copy(), LOW, HIGH)

        if feasible(p, duration, j0):
            return p

        if p[1] <= EPS or p[4] >= 1 - EPS:
            p[1] = max(p[1], 0.05)
            p[4] = min(p[4], 0.95)

        for _ in range(32):
            if feasible(p, duration, j0):
                return p

            p[[0, 2, 3]] *= 0.5

        p[[0, 2, 3]] = 0.0

        if not feasible(p, duration, j0):
            raise ValueError(
                "Could not construct a feasible cake-dominant start."
            )

        return p

    # Feasible initial candidates; final shares still come from fitting.
    decline = max(j0 / float(y[-1]), 1.0001)

    kc_est = np.clip(
        (decline ** 2 - 1) /
        (20 * j0 * j0 * duration),
        1e-9,
        1.0,
    )

    pure_starts = [
        np.array([0.0, kc, 0.0, 0.0, a, 0.5])
        for a in (0.05, 0.3, 0.6)
        for kc in (float(kc_est), 0.02, 0.2, 1.0)
    ]

    for p in pure_starts:
        keep(p, "Pure-cake initial candidate")

    for seed in SEEDS:
        rng = np.random.default_rng(seed)

        pop = np.array([
            repair(LOW + (HIGH - LOW) * rng.random(6))
            for _ in range(GA_POP)
        ])

        pop[:len(pure_starts)] = np.array(pure_starts)

        for _ in range(GA_GEN):
            scores = np.array([loss(p) for p in pop])
            order = np.argsort(scores)

            children = [
                pop[i].copy()
                for i in order[:GA_ELITE]
            ]

            while len(children) < GA_POP:
                ids = rng.integers(
                    0, GA_POP, 6
                ).reshape(2, 3)

                parents = [
                    pop[x[np.argmin(scores[x])]]
                    for x in ids
                ]

                cut = int(rng.integers(1, 6))

                child = np.r_[
                    parents[0][:cut],
                    parents[1][cut:]
                ].copy()

                changes = rng.random(6) < GA_MUT

                child[changes] += rng.normal(
                    0.0,
                    0.1,
                    changes.sum(),
                )

                children.append(repair(child))

            pop = np.array(children)

        order = np.argsort([
            loss(p) for p in pop
        ])

        for i in order[:4]:
            keep(pop[i], f"GA seed {seed}")

    starts = sorted(
        candidates,
        key=lambda item: item[0],
    )[:5]

    starts += [
        (loss(p), p, "Pure-cake start")
        for p in pure_starts[::3]
    ]

    for _, start, origin in starts:
        result = minimize(
            loss,
            start,
            method="SLSQP",
            bounds=BOUNDS,
            constraints=[{
                "type": "ineq",
                "fun": lambda p: cake_constraints(
                    p, duration, j0
                ),
            }],
            options={
                "maxiter": 500,
                "ftol": 1e-11,
            },
        )

        # Recheck actual feasibility rather than trusting solver status.
        keep(
            result.x,
            origin + " + constrained refinement",
        )

    if not candidates:
        raise ValueError(
            "No valid cake-dominant fit was found."
        )

    best_loss, p, origin = min(
        candidates,
        key=lambda item: item[0],
    )

    eta = shares(p, duration, j0)
    prediction = model(p, t, j0)

    assert np.argmax(eta) == 3
    assert np.all(eta >= 0.0)

    return {
        "params": p,
        "eta": eta,
        "pred": prediction,
        "metrics": metrics(y, prediction),
        "fit_origin": origin,
        "fit_loss": best_loss,
        "constraint": CONSTRAINT_TEXT,
    }


# ==============================================================================
# Data loading and preprocessing
# ==============================================================================

def read_csv_bytes(raw):
    last_error = None

    for enc in (
        "utf-8-sig",
        "utf-8",
        "gbk",
        "latin1",
    ):
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                encoding=enc,
            )
        except (
            UnicodeError,
            pd.errors.ParserError,
        ) as exc:
            last_error = exc

    raise ValueError(
        f"Cannot read CSV: {last_error}"
    )


def load_preset(kind):
    name = PRESETS[kind]
    base = Path(__file__).resolve().parent

    folders = (
        base,
        base / "data",
        base / "model_data",
        base / "模型数据",
    )

    for folder in folders:
        path = folder / name

        if path.exists():
            return path.read_bytes(), name

    from urllib.request import urlopen

    with urlopen(
        RAW_BASE + name,
        timeout=30,
    ) as response:
        return response.read(), name


def clean_data(raw):
    df = read_csv_bytes(raw)
    rename = {}

    for col in df:
        key = (
            str(col)
            .replace("\ufeff", "")
            .replace(" ", "")
            .strip()
            .lower()
        )

        key = (
            key.replace("（", "(")
            .replace("）", ")")
        )

        if key in {
            "时间s", "时间(s)", "times",
            "time(s)", "time", "t", "时间",
        }:
            rename[col] = "time"

        elif key in {
            "实际通量", "通量", "flux", "j",
        }:
            rename[col] = "flux"

    df = df.rename(columns=rename)

    if not {"time", "flux"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain time in seconds and positive flux."
        )

    d = df[["time", "flux"]].apply(
        pd.to_numeric,
        errors="coerce",
    )

    valid = (
        np.isfinite(d["time"])
        & np.isfinite(d["flux"])
        & (d["flux"] > 0)
    )

    d = d[valid]

    d = (
        d.groupby("time", as_index=False)["flux"]
        .mean()
        .sort_values("time")
    )

    if len(d) > 10:
        retain = max(
            int(round(len(d) * 0.99)),
            MIN_POINTS,
        )
        d = d.iloc[:retain]

    if len(d) < MIN_POINTS:
        raise ValueError(
            "At least five valid observations are required."
        )

    absolute_start = float(d["time"].iloc[0])

    t = (
        d["time"].to_numpy(float)
        - absolute_start
    )

    y = d["flux"].to_numpy(float)

    if t[-1] <= 0:
        raise ValueError(
            "Time must span a positive duration."
        )

    return t, y, absolute_start


# ==============================================================================
# Windows and flux crossing
# ==============================================================================

def analyze_window(
    t, y, start, end, label, key
):
    mask = (t >= start) & (t <= end)
    ts = t[mask]
    ys = y[mask]

    result = {
        "label": label,
        "key": key,
        "target_end": float(end),
        "available_time": float(end),
        "success": False,
    }

    if len(ts) < MIN_POINTS:
        result["error"] = (
            "Fewer than five observations in this window."
        )
        return result

    result.update(
        t=ts,
        y=ys,
        t_start=float(ts[0]),
        t_end=float(ts[-1]),
    )

    try:
        fit = fit_curve(
            ts - ts[0],
            ys,
        )

        result.update(fit)

        result.update(
            success=True,
            P=float(result["eta"][1:3].sum()),
        )

    except ValueError as exc:
        result["error"] = str(exc)

    return result


def flux_crossing(
    p, duration, j0, ratio
):
    if float(model(p, duration, j0)) > ratio * j0:
        return None

    return float(
        brentq(
            lambda t: (
                float(log_flux_ratio(p, t, j0))
                - np.log(ratio)
            ),
            0.0,
            duration,
            xtol=1e-8,
        )
    )


# ==============================================================================
# Decision rules
# ==============================================================================

def decide(
    stages,
    flux_time,
    flux_ratio,
    p_min,
    decline_min,
):
    """
    Only three rule families:
      F: hard flux threshold
      P-BASELINE: early P below the configured baseline
      P-DECLINE: relative P decline reaches the configured threshold

    No complete/intermediate ratio rule.
    No mandatory 0.65Tdata fallback.
    """
    candidates = []
    checks = []

    checks.append({
        "Rule": "F",
        "Condition": (
            f"Fitted J/J0 <= {flux_ratio:.0%}"
        ),
        "Reached": flux_time is not None,
        "Time (min)": (
            flux_time / 60
            if flux_time is not None else np.nan
        ),
    })

    if flux_time is not None:
        candidates.append({
            "code": "F",
            "time": flux_time,
            "reason": (
                "Hard flux threshold: fitted J/J0 reaches "
                f"{flux_ratio:.0%}."
            ),
        })

    valid = {
        s["key"]: s
        for s in stages
        if s["success"]
    }

    early = valid.get("early")

    if early is None:
        checks.append({
            "Rule": "P",
            "Condition": "Early reference unavailable",
            "Reached": False,
            "Time (min)": np.nan,
        })

    else:
        p0 = early["P"]
        low_p = p0 < p_min

        checks.append({
            "Rule": "P-BASELINE",
            "Condition": (
                f"Early P={p0:.4%} < {p_min:.2%}"
            ),
            "Reached": low_p,
            "Time (min)": (
                early["available_time"] / 60
            ),
        })

        if low_p:
            candidates.append({
                "code": "P-BASELINE",
                "time": early["available_time"],
                "reason": (
                    f"Early P={p0:.4%} is below "
                    f"the {p_min:.2%} baseline."
                ),
            })

        for key in (
            "middle_front",
            "middle_back",
            "late",
        ):
            current = valid.get(key)

            if current is None:
                continue

            usable = (
                p0 >= p_min
                and p0 > EPS
            )

            decline = (
                max(
                    0.0,
                    (p0 - current["P"]) / p0,
                )
                if usable else None
            )

            reached = (
                decline is not None
                and decline >= decline_min
            )

            if decline is not None:
                detail = (
                    f"P: {p0:.4%} -> "
                    f"{current['P']:.4%}; "
                    f"relative decline={decline:.2%} "
                    f">= {decline_min:.2%}"
                )
            else:
                detail = (
                    "Reference P is zero or below "
                    "the configured baseline."
                )

            checks.append({
                "Rule": "P-DECLINE / " + key,
                "Condition": detail,
                "Reached": reached,
                "Time (min)": (
                    current["available_time"] / 60
                ),
            })

            if reached:
                candidates.append({
                    "code": "P-DECLINE",
                    "time": current["available_time"],
                    "reason": detail,
                })

    if not candidates:
        return {
            "time": None,
            "codes": "NONE",
            "reason": (
                "No configured trigger was reached "
                "within the observed record."
            ),
            "checks": checks,
            "candidates": [],
        }

    first = min(
        c["time"] for c in candidates
    )

    winners = [
        c for c in candidates
        if abs(c["time"] - first) <= 1e-6
    ]

    return {
        "time": first,
        "codes": " + ".join(
            dict.fromkeys(
                c["code"] for c in winners
            )
        ),
        "reason": " Simultaneous trigger: ".join(
            c["reason"] for c in winners
        ),
        "checks": checks,
        "candidates": candidates,
    }


# ==============================================================================
# Complete analysis
# ==============================================================================

@st.cache_data(
    show_spinner=False,
    max_entries=32,
)
def analyze(
    raw,
    kind,
    filename,
    ratio,
    p_min,
    decline,
    version,
):
    t, y, absolute_start = clean_data(raw)

    full = fit_curve(t, y)

    duration = float(t[-1])
    j0 = float(y[0])

    definitions = [
        (
            0.0, 0.25,
            "Early: 0-25% Tdata",
            "early",
        ),
        (
            0.25, 0.45,
            "Middle front: 25-45% Tdata",
            "middle_front",
        ),
        (
            0.45, 0.65,
            "Middle back: 45-65% Tdata",
            "middle_back",
        ),
        (
            0.65, 1.0,
            "Late: 65-100% Tdata",
            "late",
        ),
    ]

    stages = [
        analyze_window(
            t,
            y,
            start * duration,
            end * duration,
            label,
            key,
        )
        for start, end, label, key in definitions
    ]

    ft = flux_crossing(
        full["params"],
        duration,
        j0,
        ratio,
    )

    decision = decide(
        stages,
        ft,
        ratio,
        p_min,
        decline,
    )

    threshold = (
        analyze_window(
            t,
            y,
            0.0,
            ft,
            "Flux-threshold window",
            "threshold",
        )
        if ft is not None else None
    )

    decision_window = (
        analyze_window(
            t,
            y,
            0.0,
            decision["time"],
            "Recommended-time window",
            "decision",
        )
        if decision["time"] is not None else None
    )

    return {
        "type": kind,
        "file": filename,
        "t": t,
        "y": y,
        "J0": j0,
        "absolute_start": absolute_start,
        "duration": duration,
        "full": full,
        "stages": stages,
        "flux_time": ft,
        "decision": decision,
        "threshold": threshold,
        "decision_window": decision_window,
        "settings": (
            ratio,
            p_min,
            decline,
        ),
        "version": version,
    }


# ==============================================================================
# Presentation helpers
# ==============================================================================

def time_text(t):
    if t is None:
        return "Not reached"

    return f"{t / 60:.2f} min ({t:.2f} s)"


def pie(result, title):
    st.markdown("**" + title + "**")

    if (
        result is None
        or not result.get("success", True)
    ):
        st.info(
            "No valid fitting window."
            if result is None else result["error"]
        )
        return

    eta = result["eta"]

    if np.argmax(eta) != 3:
        st.error(
            "Constraint verification failed; "
            "this pie is not displayed."
        )
        return

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    indices = np.flatnonzero(
        eta > 1e-10
    )

    ax.pie(
        eta[indices],
        labels=[
            f"{NAMES[i]}\n{eta[i]:.2%}"
            for i in indices
        ],
        colors=[
            COLORS[i] for i in indices
        ],
        startangle=90,
        textprops={"fontsize": 9},
    )

    ax.axis("equal")

    st.pyplot(fig)
    plt.close(fig)

    m = result["metrics"]

    st.caption(
        "Constrained fit | "
        f"NRMSE={m['NRMSE']:.4f}; "
        f"MAPE={m['MAPE']:.2%}; "
        f"R²={m['R2']:.4f}"
    )


def stage_table(result):
    rows = []

    for s in result["stages"]:
        row = {
            "Stage": s["label"],
            "Status": (
                "Completed"
                if s["success"] else s["error"]
            ),
            "Evaluation time (min)": (
                s["available_time"] / 60
            ),
        }

        if s["success"]:
            row.update({
                name + " (%)": 100 * s["eta"][i]
                for i, name in enumerate(NAMES)
            })

            row.update({
                "P (%)": 100 * s["P"],
                **s["metrics"],
            })

        rows.append(row)

    return pd.DataFrame(rows)


def show_result(r):
    st.subheader(
        r["type"] + " — " + r["file"]
    )

    decision = r["decision"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Recommended backwashing time",
        time_text(decision["time"]),
    )

    c2.metric(
        "Actual trigger",
        decision["codes"],
    )

    c3.metric(
        "Hard flux threshold time",
        time_text(r["flux_time"]),
    )

    st.info(decision["reason"])

    st.caption(
        f"Parameters used: J/J0={r['settings'][0]:.0%}; "
        f"P baseline={r['settings'][1]:.2%}; "
        f"P decline={r['settings'][2]:.2%}. "
        "Times are elapsed from the first retained observation."
    )

    st.dataframe(
        pd.DataFrame(decision["checks"]),
        use_container_width=True,
    )

    st.caption(
        "'Reached' records whether each condition is met "
        "in the retrospective record. "
        "'Actual trigger' reports only the earliest event; "
        "later conditions are not additional causes of that event."
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        r["t"] / 60,
        r["y"] / r["J0"],
        ".",
        ms=3,
        alpha=0.5,
        label="Observed",
    )

    ax.plot(
        r["t"] / 60,
        r["full"]["pred"] / r["J0"],
        lw=2,
        label="Full-process constrained fit",
    )

    ax.axhline(
        r["settings"][0],
        color="#cc5555",
        ls=":",
        label="Hard flux level",
    )

    if r["flux_time"] is not None:
        ax.axvline(
            r["flux_time"] / 60,
            color="#cc5555",
            ls="--",
            label="Flux crossing",
        )

    if decision["time"] is not None:
        ax.axvline(
            decision["time"] / 60,
            color="#7952b3",
            ls="-.",
            label=(
                "Selected event: "
                + decision["codes"]
            ),
        )

    ax.set(
        xlabel="Elapsed time (min)",
        ylabel="J/J0",
    )

    ax.grid(alpha=0.2)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    cols = st.columns(2)

    with cols[0]:
        pie(
            r["full"],
            "Full-process mechanism contributions",
        )

    with cols[1]:
        pie(
            r["decision_window"],
            "Contributions up to the recommended time",
        )

    with st.expander(
        "Stage mechanism contributions and flux-threshold window"
    ):
        st.dataframe(
            stage_table(r),
            use_container_width=True,
        )

        for pair_start in range(
            0,
            len(r["stages"]),
            2,
        ):
            cols = st.columns(2)

            pair = r["stages"][
                pair_start:pair_start + 2
            ]

            for col, s in zip(cols, pair):
                with col:
                    pie(s, s["label"])

        pie(
            r["threshold"],
            "Contributions up to the flux-threshold crossing",
        )

    export = {
        "version": VERSION,
        "file": r["file"],
        "type": r["type"],
        "constraint": CONSTRAINT_TEXT,
        "settings": r["settings"],
        "decision": decision,
        "full_fit": {
            k: r["full"][k]
            for k in (
                "params",
                "eta",
                "metrics",
                "fit_origin",
                "fit_loss",
            )
        },
        "stages": stage_table(r).to_dict(
            "records"
        ),
    }

    st.download_button(
        "Download detailed result JSON",
        json.dumps(
            export,
            indent=2,
            default=lambda x: (
                x.tolist()
                if isinstance(x, np.ndarray)
                else float(x)
            ),
        ),
        file_name=(
            r["type"]
            + "_constrained_result.json"
        ),
        mime="application/json",
        key=(
            "json_"
            + r["type"]
            + "_"
            + r["file"]
        ),
    )


# ==============================================================================
# Streamlit interface
# ==============================================================================

def main():
    st.set_page_config(
        page_title=TITLE,
        page_icon="💧",
        layout="wide",
    )

    st.title(TITLE)
    st.info(CONSTRAINT_TEXT)

    st.caption(
        "Retrospective demonstration. "
        "P=complete+intermediate blocking. "
        "Baseline and decline thresholds are empirical settings. "
        "All four foulant types use the same three configured rules."
    )

    ratio = st.sidebar.slider(
        "Hard flux threshold (% of initial flux)",
        50.0,
        70.0,
        70.0,
        1.0,
        help=(
            "70% J0 means a 30% flux decline; "
            "50% J0 means a 50% decline."
        ),
    ) / 100

    p_min = st.sidebar.slider(
        "Early P baseline (%)",
        0.0,
        1.87,
        1.0,
        0.01,
        format="%.2f",
        help=(
            "Early P below this value independently "
            "triggers at the early endpoint."
        ),
    ) / 100

    decline = st.sidebar.slider(
        "P relative decline trigger (%)",
        24.22,
        100.0,
        25.0,
        0.01,
        format="%.2f",
        help=(
            "max(0, (P_early-P_current)/P_early). "
            "Increases do not trigger."
        ),
    ) / 100

    mode = st.sidebar.selectbox(
        "Analysis mode",
        [
            "Single preset",
            "Batch presets",
            "Upload CSV",
        ],
    )

    kind = "HA"
    upload = None

    if mode == "Single preset":
        kind = st.selectbox(
            "Foulant type",
            list(PRESETS),
        )

    elif mode == "Upload CSV":
        kind = st.selectbox(
            "Foulant type",
            list(PRESETS),
        )

        upload = st.file_uploader(
            "CSV with time in seconds and flux",
            type="csv",
        )

    signature = (
        VERSION,
        ratio,
        p_min,
        decline,
        mode,
        kind,
        (
            upload.getvalue()
            if upload is not None else None
        ),
    )

    if st.sidebar.button("Clear results"):
        st.session_state.pop(
            "analysis_bundle",
            None,
        )

    if st.button("Start analysis"):
        output = []

        kinds = (
            list(PRESETS)
            if mode == "Batch presets"
            else [kind]
        )

        if (
            mode == "Upload CSV"
            and upload is None
        ):
            st.error("Select a CSV first.")
            return

        with st.spinner(
            "Searching for cake-dominant feasible fits..."
        ):
            for selected in kinds:
                try:
                    if mode == "Upload CSV":
                        raw = upload.getvalue()
                        filename = upload.name
                    else:
                        raw, filename = load_preset(
                            selected
                        )

                    output.append(
                        analyze(
                            raw,
                            selected,
                            filename,
                            ratio,
                            p_min,
                            decline,
                            VERSION,
                        )
                    )

                except Exception as exc:
                    output.append({
                        "error": str(exc),
                        "type": selected,
                    })

        st.session_state[
            "analysis_bundle"
        ] = (
            signature,
            output,
        )

    bundle = st.session_state.get(
        "analysis_bundle"
    )

    if bundle is None:
        return

    if bundle[0] != signature:
        st.warning(
            "Settings or input changed. "
            "Click Start analysis to update results."
        )
        return

    summary = []

    for r in bundle[1]:
        if "error" in r:
            st.error(
                r["type"] + ": " + r["error"]
            )
            continue

        summary.append({
            "Foulant": r["type"],
            "File": r["file"],
            "Recommended time (min)": (
                r["decision"]["time"] / 60
                if r["decision"]["time"] is not None
                else np.nan
            ),
            "Actual trigger": r["decision"]["codes"],
            "Reason": r["decision"]["reason"],
            "Flux-threshold time (min)": (
                r["flux_time"] / 60
                if r["flux_time"] is not None
                else np.nan
            ),
            "Flux threshold (%)": 100 * ratio,
            "P baseline (%)": 100 * p_min,
            "P decline threshold (%)": 100 * decline,
            "Cake contribution (%)": (
                100 * r["full"]["eta"][3]
            ),
            "NRMSE": (
                r["full"]["metrics"]["NRMSE"]
            ),
            "Fit type": (
                "Cake-dominant constrained fit"
            ),
        })

    if summary:
        table = pd.DataFrame(summary)

        st.dataframe(
            table,
            use_container_width=True,
        )

        st.download_button(
            "Download summary CSV",
            table.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name=(
                "backwash_constrained_summary.csv"
            ),
            mime="text/csv",
        )

    for r in bundle[1]:
        if "error" not in r:
            with st.expander(
                r["type"] + " detailed report",
                expanded=True,
            ):
                show_result(r)


if __name__ == "__main__":
    main()
