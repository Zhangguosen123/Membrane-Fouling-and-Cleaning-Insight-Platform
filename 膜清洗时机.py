# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import r2_score

# ==============================================================================
# Configuration
# ==============================================================================
APP_TITLE = "Membrane Fouling and Cleaning Insight Platform"

PRESET_FILES = {
    "BSA": "BSAdata1.csv",
    "HA": "HAdata1.csv",
    "SA": "SAdata1.csv",
}

USE_LOG_FIT = True
USE_HUBER = True
HUBER_DELTA = 0.002
GA_POP = 60
GA_GEN = 100
GA_MUT = 0.10
GA_ELITE = 2
RANDOM_SEED = 42
BOUNDS = [(0, 1)] * 4 + [(0.05, 1)] * 2  # Ks, Kc, Kb, Ki, a, b
EPS = 1e-12
EXP_FLOOR = -50.0
MIN_STAGE_POINTS = 5

MECH_NAMES = [
    "Standard Blocking",
    "Complete Blocking",
    "Intermediate Blocking",
    "Cake Filtration",
]


# ==============================================================================
# Four-Mechanism Coupled Model
# ==============================================================================
def stage1_model(params, t, J0):
    Ks, Kc, Kb, Ki, a, b = params

    c1 = 10.0 * Ks * J0 / 2.0
    c2 = 10.0 * Kb
    c3 = 10.0 * Ki * J0
    c4 = 20.0 * Kc * J0 ** 2

    base1 = np.maximum(1.0 + c1 * t, EPS)
    base3 = np.maximum(1.0 + c3 * t, EPS)
    base4 = np.maximum(1.0 + c4 * t, EPS)

    term1 = base1 ** (-2.0 * a)
    term2 = np.exp(np.maximum(-b * c2 * t, EXP_FLOOR))
    term3 = base3 ** (-(1.0 - b))
    term4 = base4 ** (-(1.0 - a) / 2.0)

    J_pred = J0 * term1 * term2 * term3 * term4
    return np.maximum(J_pred, EPS)


def huber_loss(residual, delta):
    abs_r = np.abs(residual)
    quad = 0.5 * abs_r ** 2
    lin = delta * (abs_r - 0.5 * delta)
    return np.where(abs_r <= delta, quad, lin)


def objective(params, t, J_obs, J0):
    J_pred = stage1_model(params, t, J0)
    mask = np.isfinite(J_obs) & np.isfinite(J_pred)

    if mask.sum() < 5:
        return 1e9

    y = J_obs[mask]
    yhat = J_pred[mask]

    if USE_LOG_FIT:
        y = np.maximum(y, EPS)
        yhat = np.maximum(yhat, EPS)
        residual = np.log(y) - np.log(yhat)
    else:
        residual = y - yhat

    if USE_HUBER:
        return np.mean(huber_loss(residual, HUBER_DELTA))
    return np.mean(residual ** 2)


def genetic_algorithm(objective_fn, bounds, t, J_obs, J0):
    rng = np.random.default_rng(RANDOM_SEED)
    dim = len(bounds)

    pop = rng.random((GA_POP, dim))
    for i, (lo, hi) in enumerate(bounds):
        pop[:, i] = lo + pop[:, i] * (hi - lo)

    def fitness(ind):
        try:
            val = float(objective_fn(ind, t, J_obs, J0))
            return val if np.isfinite(val) else 1e9
        except Exception:
            return 1e9

    for _ in range(GA_GEN):
        scores = np.array([fitness(ind) for ind in pop])
        elite_idx = np.argsort(scores)[:GA_ELITE]
        new_pop = pop[elite_idx].copy()

        while len(new_pop) < GA_POP:
            idx1 = rng.integers(0, len(pop), size=3)
            p1 = pop[idx1[np.argmin(scores[idx1])]].copy()

            idx2 = rng.integers(0, len(pop), size=3)
            p2 = pop[idx2[np.argmin(scores[idx2])]].copy()

            cp = rng.integers(1, dim)
            child = np.concatenate([p1[:cp], p2[cp:]])

            for i, (lo, hi) in enumerate(bounds):
                if rng.random() < GA_MUT:
                    child[i] += rng.normal(0, 0.1 * (hi - lo))
                    child[i] = np.clip(child[i], lo, hi)

            new_pop = np.vstack([new_pop, child])

        pop = new_pop

    scores = np.array([fitness(ind) for ind in pop])
    return pop[np.argmin(scores)]


def fit_model(t, J_obs, J0):
    if len(t) < 5:
        return np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5])
    return genetic_algorithm(objective, BOUNDS, t, J_obs, J0)


def calculate_mechanism_contribution(params, t, J0):
    Ks, Kc, Kb, Ki, a, b = params

    c1 = 10.0 * Ks * J0 / 2.0
    c2 = 10.0 * Kb
    c3 = 10.0 * Ki * J0
    c4 = 20.0 * Kc * J0 ** 2

    s1 = -(2.0 * a) * c1 / (1.0 + c1 * t + EPS)
    s2 = -b * c2 * np.ones_like(t)
    s3 = -(1.0 - b) * c3 / (1.0 + c3 * t + EPS)
    s4 = -(1.0 - a) * c4 / (2.0 * (1.0 + c4 * t + EPS))

    Di = []
    for si in [s1, s2, s3, s4]:
        val = -np.trapz(si, t)
        Di.append(max(val, 0.0))

    dsum = sum(Di) + EPS
    return np.array([d / dsum for d in Di])


# ==============================================================================
# Data Loading and Cleaning
# ==============================================================================
def app_base_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def resolve_preset_file(filename):
    base = app_base_dir()
    candidate_dirs = [
        base,
        os.path.join(base, "data"),
        os.path.join(base, "model_data"),
        os.path.join(base, "模型数据"),
    ]

    for directory in candidate_dirs:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path

    searched = "\n".join(candidate_dirs)
    raise FileNotFoundError(
        f"Preset data file was not found: {filename}\n"
        f"Searched folders:\n{searched}"
    )


def read_csv_robust(path_or_buffer, encoding_list=("utf-8-sig", "utf-8", "gbk", "latin1")):
    last_error = None

    for enc in encoding_list:
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            df = pd.read_csv(path_or_buffer, encoding=enc)
            return df, enc
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to read CSV file: {last_error}")


def normalize_cols_to_standard(df):
    def norm_key(c):
        return (
            str(c)
            .replace("\ufeff", "")
            .strip()
            .replace("（", "(")
            .replace("）", ")")
            .replace(" ", "")
            .lower()
        )

    new_names = {}
    for c in df.columns:
        k = norm_key(c)

        if k in {"时间s", "时间(s)", "times", "time(s)", "time", "t", "时间"}:
            new_names[c] = "Time (s)"
        elif k in {"实际通量", "通量", "flux", "j"}:
            new_names[c] = "Flux"

    return df.rename(columns=new_names)


def clean_series(t, J):
    t = np.asarray(t, dtype=float)
    J = np.asarray(J, dtype=float)

    mask = np.isfinite(t) & np.isfinite(J) & (J > 0)
    t = t[mask]
    J = J[mask]

    if len(t) == 0:
        return t, J

    order = np.argsort(t)
    t = t[order]
    J = J[order]

    if len(t) > 10:
        k = max(int(round(len(t) * 0.99)), 5)
        t = t[:k]
        J = J[:k]

    return t, J


def dataframe_to_series(df, filename="uploaded.csv", encoding="uploaded"):
    df = normalize_cols_to_standard(df)

    if "Time (s)" not in df.columns or "Flux" not in df.columns:
        raise ValueError(
            "The data file must contain time and flux columns, such as "
            "'Time (s)' / 'time' and 'Flux' / 'J'."
        )

    t_clean, J_clean = clean_series(df["Time (s)"].values, df["Flux"].values)

    if len(J_clean) < 5:
        raise ValueError(f"{filename} does not contain enough valid data points for fitting.")

    J0 = float(J_clean[0])
    if J0 <= 0:
        raise ValueError(f"The initial flux in {filename} is zero or negative.")

    return t_clean, J_clean, J0, filename, encoding


def load_preset_data(data_type):
    filename = PRESET_FILES[data_type]
    file_path = resolve_preset_file(filename)
    df, enc = read_csv_robust(file_path)
    return dataframe_to_series(df, filename, enc)


# ==============================================================================
# Metrics, Stages, and Cleaning Timing Decision
# ==============================================================================
def calculate_metrics(J_obs, J_pred):
    mask = np.isfinite(J_obs) & np.isfinite(J_pred)

    if mask.sum() < 2:
        return {"R2": np.nan, "NRMSE": np.nan, "MAPE": np.nan}

    y = J_obs[mask]
    yhat = J_pred[mask]

    try:
        r2 = r2_score(y, yhat)
    except Exception:
        r2 = np.nan

    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    y_range = np.max(y) - np.min(y)
    nrmse = rmse / (y_range + EPS) if y_range > 0 else np.nan

    mape_floor = max(1e-8, 0.05 * np.median(np.abs(y)))
    denom = np.maximum(np.abs(y), mape_floor)
    mape = np.mean(np.abs(y - yhat) / denom)

    return {
        "R2": round(float(r2), 3) if np.isfinite(r2) else np.nan,
        "NRMSE": round(float(nrmse), 3) if np.isfinite(nrmse) else np.nan,
        "MAPE": round(float(mape), 3) if np.isfinite(mape) else np.nan,
    }


def analyze_interval(t_abs, J_obs, label):
    if len(t_abs) < MIN_STAGE_POINTS:
        return {
            "success": False,
            "label": label,
            "error": f"Insufficient valid points: {len(t_abs)}",
        }

    t_rel = t_abs - t_abs[0]
    J0 = float(J_obs[0])

    params = fit_model(t_rel, J_obs, J0)
    J_pred = stage1_model(params, t_rel, J0)
    eta = calculate_mechanism_contribution(params, t_rel, J0)
    metrics = calculate_metrics(J_obs, J_pred)

    dominant_idx = int(np.argmax(eta))

    return {
        "success": True,
        "label": label,
        "t_abs": t_abs,
        "t_rel": t_rel,
        "J_obs": J_obs,
        "J0": J0,
        "params": params,
        "J_pred": J_pred,
        "eta": eta,
        "metrics": metrics,
        "dominant_idx": dominant_idx,
        "dominant_mechanism": MECH_NAMES[dominant_idx],
        "dominant_ratio": float(eta[dominant_idx]),
        "intermediate_ratio": float(eta[2]),
        "cake_ratio": float(eta[3]),
        "t_start": float(t_abs[0]),
        "t_end": float(t_abs[-1]),
        "flux_start": float(J_obs[0]),
        "flux_end": float(J_obs[-1]),
    }


def find_time_when_curve_below(t, J_pred, target_flux):
    idx = np.where(J_pred <= target_flux)[0]

    if len(idx) == 0:
        return None, None

    first_idx = int(idx[0])

    if first_idx == 0:
        return float(t[0]), first_idx

    t1, t2 = t[first_idx - 1], t[first_idx]
    j1, j2 = J_pred[first_idx - 1], J_pred[first_idx]

    if abs(j2 - j1) < EPS:
        return float(t2), first_idx

    crossing_time = t1 + (t2 - t1) * (target_flux - j1) / (j2 - j1)
    return float(crossing_time), first_idx


def find_cleaning_time_by_flux(t, J_pred, J0, ratio=0.7):
    target_flux = J0 * ratio
    crossing_time, crossing_idx = find_time_when_curve_below(t, J_pred, target_flux)

    if crossing_time is None:
        return float(t[-1]), float(J_pred[-1]), len(t) - 1, False

    return crossing_time, float(target_flux), crossing_idx, True


def get_time_at_flux_ratio(t, J_pred, J0, ratio):
    if ratio >= 1.0:
        return float(t[0])
    target_flux = J0 * ratio
    crossing_time, _ = find_time_when_curve_below(t, J_pred, target_flux)
    return crossing_time


def slice_by_time(t, J, start_time, end_time):
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], J[mask]


def build_three_stage_analysis(t, J, J_pred_full, J0):
    t0 = float(t[0])
    tend = float(t[-1])

    t90 = get_time_at_flux_ratio(t, J_pred_full, J0, 0.9)
    t80 = get_time_at_flux_ratio(t, J_pred_full, J0, 0.8)
    t70 = get_time_at_flux_ratio(t, J_pred_full, J0, 0.7)

    if t90 is None:
        t90 = tend
    if t80 is None:
        t80 = tend
    if t70 is None:
        t70 = tend

    raw_stages = [
        ("Stage I: 100%-90% Flux Interval", "100%-90%", t0, t90),
        ("Stage II: 90%-80% Flux Interval", "90%-80%", t90, t80),
        ("Stage III: 80%-70% Flux Interval", "80%-70%", t80, t70),
    ]

    stage_results = []
    for stage_name, flux_range, start_time, end_time in raw_stages:
        if end_time <= start_time:
            stage_results.append({
                "success": False,
                "label": stage_name,
                "flux_range": flux_range,
                "t_start": start_time,
                "t_end": end_time,
                "error": "This flux interval has insufficient time span.",
            })
            continue

        t_seg, J_seg = slice_by_time(t, J, start_time, end_time)
        res = analyze_interval(t_seg, J_seg, stage_name)
        res["flux_range"] = flux_range
        res["target_start_time"] = start_time
        res["target_end_time"] = end_time
        stage_results.append(res)

    return stage_results


def recommend_cleaning_strategy(eta, stage_type="full"):
    dominant_idx = int(np.argmax(eta))
    dominant = MECH_NAMES[dominant_idx]
    dominant_ratio = eta[dominant_idx] * 100

    if stage_type == "early":
        if dominant_idx == 3:
            return (
                f"Dominant mechanism: {dominant} ({dominant_ratio:.1f}%).\n"
                f"Recommended action: hydraulic backwashing at 0.08-0.10 MPa for 3-5 min.\n"
                f"Rationale: surface cake fouling is usually more reversible at this stage."
            )
        if dominant_idx in (0, 1):
            return (
                f"Dominant mechanism: {dominant} ({dominant_ratio:.1f}%).\n"
                f"Recommended action: timely low-intensity backwashing, with mild acid cleaning if needed.\n"
                f"Rationale: pore-related fouling may increase irreversible fouling risk if operation continues."
            )
        if dominant_idx == 2:
            return (
                f"Dominant mechanism: {dominant} ({dominant_ratio:.1f}%).\n"
                f"Recommended action: mild backwashing and close tracking of intermediate-blocking attenuation.\n"
                f"Rationale: intermediate blocking reflects the interfacial bridging structure associated with the reversible-to-irreversible transition."
            )

    if dominant_idx == 3:
        return (
            f"Dominant mechanism: {dominant} ({dominant_ratio:.1f}%).\n"
            f"Recommended action: hydraulic backwashing, optionally combined with low-dose oxidative or alkaline cleaning.\n"
            f"Operational note: shorten the filtration cycle to reduce cake-layer compaction."
        )
    if dominant_idx in (0, 1):
        return (
            f"Dominant mechanism: {dominant} ({dominant_ratio:.1f}%).\n"
            f"Recommended action: backwashing plus mild acid or chelating cleaning.\n"
            f"Operational note: pore-related fouling is more likely to leave irreversible residues, so cleaning should not be delayed."
        )
    if dominant_idx == 2:
        return (
            f"Dominant mechanism: {dominant} ({dominant_ratio:.1f}%).\n"
            f"Recommended action: backwashing plus mild alkaline cleaning to disrupt pore-entrance bridging.\n"
            f"Operational note: intervention is recommended when intermediate blocking continuously decreases toward the threshold."
        )

    return "Mixed mechanisms are involved. A combined mild cleaning protocol is recommended."


def decide_backwash_timing(stage_results, intermediate_threshold, flux_cleaning_time, flux_ratio):
    valid_stages = [s for s in stage_results if s.get("success")]

    if not valid_stages:
        return {
            "triggered": False,
            "decision_time": flux_cleaning_time,
            "decision_stage": "Stage-resolved diagnosis unavailable",
            "level": "warning",
            "message": "Stage-resolved mechanism analysis is unavailable. The flux-based cleaning point is used instead.",
            "basis": f"Cleaning is suggested when flux decreases to {flux_ratio * 100:.0f}% of the initial flux.",
        }

    for s in valid_stages:
        if s["intermediate_ratio"] <= intermediate_threshold:
            return {
                "triggered": True,
                "decision_time": s["t_end"],
                "decision_stage": s["label"],
                "level": "danger",
                "message": (
                    f"Backwashing is recommended near the end of {s['label']}. "
                    f"The intermediate-blocking contribution is {s['intermediate_ratio'] * 100:.2f}%, "
                    f"which is below the threshold of {intermediate_threshold * 100:.2f}%."
                ),
                "basis": (
                    "Intermediate blocking represents the bridging structure at the pore entrance and the cake-membrane interface. "
                    "When its contribution approaches disappearance, the interfacial structure is considered consolidated, "
                    "indicating a transition from reversible to irreversible fouling. Therefore, backwashing should be performed at this stage."
                ),
            }

    min_stage = min(valid_stages, key=lambda x: x["intermediate_ratio"])
    return {
        "triggered": False,
        "decision_time": flux_cleaning_time,
        "decision_stage": f"No intermediate-blocking trigger; use the {flux_ratio * 100:.0f}% flux point",
        "level": "normal",
        "message": (
            f"The intermediate-blocking contribution remains above the threshold of {intermediate_threshold * 100:.2f}% "
            f"in all three stages. The lowest value appears in {min_stage['label']} "
            f"({min_stage['intermediate_ratio'] * 100:.2f}%)."
        ),
        "basis": (
            f"No near-disappearance of intermediate blocking is detected. Operation can continue until flux decreases to "
            f"{flux_ratio * 100:.0f}% of the initial flux, or until the next monitoring cycle indicates threshold crossing."
        ),
    }


def analyze_dataset(t, J, J0, filename, data_type, intermediate_threshold=0.05, flux_ratio=0.7):
    full = analyze_interval(t, J, "Full Process")
    if not full["success"]:
        raise RuntimeError(full["error"])

    J_pred_full = full["J_pred"]
    cleaning_time_flux, cleaning_flux, cleaning_idx, reached_flux_threshold = find_cleaning_time_by_flux(
        t, J_pred_full, J0, flux_ratio
    )

    t_partial = t[: cleaning_idx + 1]
    J_partial = J[: cleaning_idx + 1]
    partial = analyze_interval(t_partial, J_partial, f"100%-{int(flux_ratio * 100)}% Stage")

    stages = build_three_stage_analysis(t, J, J_pred_full, J0)
    decision = decide_backwash_timing(stages, intermediate_threshold, cleaning_time_flux, flux_ratio)

    full_strategy = recommend_cleaning_strategy(full["eta"], "full")
    partial_strategy = (
        recommend_cleaning_strategy(partial["eta"], "early")
        if partial.get("success")
        else "The 100%-70% stage contains insufficient valid data for cleaning recommendation."
    )

    return {
        "success": True,
        "filename": filename,
        "data_type": data_type,
        "J0": J0,
        "t": t,
        "J": J,
        "full": full,
        "partial": partial,
        "stages": stages,
        "decision": decision,
        "flux_cleaning_time": cleaning_time_flux,
        "flux_cleaning_flux": cleaning_flux,
        "flux_threshold_reached": reached_flux_threshold,
        "flux_ratio": flux_ratio,
        "intermediate_threshold": intermediate_threshold,
        "full_strategy": full_strategy,
        "partial_strategy": partial_strategy,
    }


def analyze_preset_file(data_type, intermediate_threshold, flux_ratio):
    t, J, J0, filename, _ = load_preset_data(data_type)
    return analyze_dataset(t, J, J0, filename, data_type, intermediate_threshold, flux_ratio)


def analyze_uploaded_file(uploaded_file, intermediate_threshold, flux_ratio):
    df, enc = read_csv_robust(uploaded_file)
    t, J, J0, filename, _ = dataframe_to_series(df, uploaded_file.name, enc)
    return analyze_dataset(t, J, J0, filename, "Custom Upload", intermediate_threshold, flux_ratio)


# ==============================================================================
# Visualization Helpers
# ==============================================================================
def percent(x):
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def metric_text(x):
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.3f}"


def draw_mechanism_pie(eta, title):
    sizes = [max(v * 100, 0.0) for v in eta]
    labels = [f"{name} {value:.1f}%" for name, value in zip(MECH_NAMES, sizes) if value > 0]
    values = [value for value in sizes if value > 0]

    fig, ax = plt.subplots(figsize=(5, 4))
    if sum(values) <= 0:
        ax.text(0.5, 0.5, "No valid contribution", ha="center", va="center")
        ax.axis("off")
    else:
        ax.pie(values, labels=labels, startangle=90)
        ax.axis("equal")
    ax.set_title(title)
    st.pyplot(fig)


def make_stage_table(stage_results):
    rows = []
    for s in stage_results:
        if s.get("success"):
            rows.append({
                "Stage": s["label"],
                "Flux Interval": s["flux_range"],
                "Start Time (s)": round(s["t_start"], 2),
                "End Time (s)": round(s["t_end"], 2),
                "Standard Blocking": percent(s["eta"][0]),
                "Complete Blocking": percent(s["eta"][1]),
                "Intermediate Blocking": percent(s["eta"][2]),
                "Cake Filtration": percent(s["eta"][3]),
                "Dominant Mechanism": f"{s['dominant_mechanism']} ({s['dominant_ratio'] * 100:.1f}%)",
                "NRMSE": s["metrics"]["NRMSE"],
                "MAPE": s["metrics"]["MAPE"],
                "Status": "Completed",
            })
        else:
            rows.append({
                "Stage": s.get("label", "N/A"),
                "Flux Interval": s.get("flux_range", "N/A"),
                "Start Time (s)": round(float(s.get("t_start", 0)), 2),
                "End Time (s)": round(float(s.get("t_end", 0)), 2),
                "Standard Blocking": "N/A",
                "Complete Blocking": "N/A",
                "Intermediate Blocking": "N/A",
                "Cake Filtration": "N/A",
                "Dominant Mechanism": "N/A",
                "NRMSE": "N/A",
                "MAPE": "N/A",
                "Status": s.get("error", "Failed"),
            })
    return pd.DataFrame(rows)


def draw_flux_curve(res):
    t = res["t"]
    J = res["J"]
    full = res["full"]
    partial = res["partial"]
    decision = res["decision"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, J, "o", ms=3, color="gray", alpha=0.6, label="Observed Flux")
    ax.plot(t, full["J_pred"], "-", lw=2.2, color="#d97706", label="Full-Process Fitting")

    if partial.get("success"):
        ax.plot(partial["t_abs"], partial["J_pred"], "-", lw=2.4, color="#16a34a", label="100%-70% Stage Fitting")

    flux_line = res["J0"] * res["flux_ratio"]
    ax.axhline(
        y=flux_line,
        color="#dc2626",
        linestyle=":",
        lw=2,
        label=f"{res['flux_ratio'] * 100:.0f}% Initial Flux Threshold",
    )
    ax.axvline(
        x=res["flux_cleaning_time"],
        color="#dc2626",
        linestyle="--",
        alpha=0.7,
        label="Flux-Based Cleaning Point",
    )

    if decision["triggered"]:
        ax.axvline(
            x=decision["decision_time"],
            color="#7c3aed",
            linestyle="-.",
            lw=2.2,
            label="Intermediate-Blocking Backwash Point",
        )

    stage_colors = ["#dcfce7", "#fef9c3", "#fee2e2"]
    for i, s in enumerate(res["stages"]):
        if s.get("success"):
            ax.axvspan(
                s["t_start"],
                s["t_end"],
                color=stage_colors[i],
                alpha=0.25,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flux")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    y_min = max(min(np.min(J), flux_line) * 0.8, 0)
    y_max = max(res["J0"] * 1.15, np.max(J) * 1.05)
    ax.set_ylim(y_min, y_max)

    st.pyplot(fig)


# ==============================================================================
# Streamlit Interface
# ==============================================================================
def main():
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💧",
        layout="wide",
    )

    st.title(APP_TITLE)

    st.sidebar.header("Settings")

    intermediate_threshold_percent = st.sidebar.slider(
        "Intermediate-Blocking Backwash Threshold (%)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help=(
            "When the intermediate-blocking contribution in any stage falls below this threshold, "
            "the app recommends backwashing intervention."
        ),
    )
    intermediate_threshold = intermediate_threshold_percent / 100.0

    flux_ratio_percent = st.sidebar.slider(
        "Flux-Based Cleaning Threshold (% of Initial Flux)",
        min_value=50,
        max_value=90,
        value=70,
        step=5,
    )
    flux_ratio = flux_ratio_percent / 100.0

    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Single Preset File", "Batch Analysis of Preset Files", "Upload Custom CSV"],
    )

    if "all_results" not in st.session_state:
        st.session_state.all_results = []

    if st.sidebar.button("Clear Results"):
        st.session_state.all_results = []

    if analysis_mode == "Single Preset File":
        st.header("Single Preset File Analysis")
        data_type = st.selectbox("Select Data Type", ["BSA", "HA", "SA"])

        st.caption(f"Preset file: {PRESET_FILES[data_type]}")

        if st.button("Start Analysis"):
            with st.spinner(f"Analyzing {PRESET_FILES[data_type]} ..."):
                try:
                    result = analyze_preset_file(data_type, intermediate_threshold, flux_ratio)
                    st.session_state.all_results = [result]
                except Exception as e:
                    st.error(str(e))

    elif analysis_mode == "Batch Analysis of Preset Files":
        st.header("Batch Analysis of Preset Files")
        st.info("The app will analyze BSAdata1.csv, HAdata1.csv, and SAdata1.csv from the GitHub repository.")

        if st.button("Start Batch Analysis"):
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, data_type in enumerate(["BSA", "HA", "SA"]):
                status_text.text(f"Analyzing ({i + 1}/3): {PRESET_FILES[data_type]}")
                try:
                    result = analyze_preset_file(data_type, intermediate_threshold, flux_ratio)
                except Exception as e:
                    result = {
                        "success": False,
                        "filename": PRESET_FILES[data_type],
                        "data_type": data_type,
                        "error": str(e),
                    }
                all_results.append(result)
                progress_bar.progress((i + 1) / 3)
                time.sleep(0.1)

            status_text.text("Batch analysis completed.")
            st.session_state.all_results = all_results

    else:
        st.header("Upload Custom CSV")
        uploaded_file = st.file_uploader("Upload a CSV file containing time and flux data", type="csv")

        if uploaded_file is not None and st.button("Analyze Uploaded File"):
            with st.spinner(f"Analyzing {uploaded_file.name} ..."):
                try:
                    result = analyze_uploaded_file(uploaded_file, intermediate_threshold, flux_ratio)
                    st.session_state.all_results = [result]
                except Exception as e:
                    st.error(str(e))

    all_results = st.session_state.all_results

    if not all_results:
        return

    st.markdown("---")
    st.header("Analysis Summary")

    summary_rows = []
    for res in all_results:
        if not res.get("success"):
            summary_rows.append({
                "File": res.get("filename", "N/A"),
                "Type": res.get("data_type", "N/A"),
                "Initial Flux": "N/A",
                "Full-Process Dominant Mechanism": "N/A",
                "100%-70% Stage Dominant Mechanism": "N/A",
                "Intermediate-Blocking Threshold": f"{intermediate_threshold_percent:.1f}%",
                "Recommended Cleaning Stage": "Analysis Failed",
                "Recommended Time (s)": "N/A",
                "Status": res.get("error", "Failed"),
            })
            continue

        full = res["full"]
        partial = res["partial"]
        decision = res["decision"]

        partial_dom = (
            f"{partial['dominant_mechanism']} ({partial['dominant_ratio'] * 100:.1f}%)"
            if partial.get("success")
            else "N/A"
        )

        summary_rows.append({
            "File": res["filename"],
            "Type": res["data_type"],
            "Initial Flux": f"{res['J0']:.6g}",
            "Full-Process Dominant Mechanism": f"{full['dominant_mechanism']} ({full['dominant_ratio'] * 100:.1f}%)",
            "100%-70% Stage Dominant Mechanism": partial_dom,
            "Intermediate-Blocking Threshold": f"{intermediate_threshold_percent:.1f}%",
            "Recommended Cleaning Stage": decision["decision_stage"],
            "Recommended Time (s)": f"{decision['decision_time']:.2f}",
            "Status": "Backwash Triggered" if decision["triggered"] else "Not Triggered; Flux-Based Point Used",
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    csv = summary_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download Summary CSV",
        data=csv,
        file_name="membrane_cleaning_decision_summary.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.header("Detailed Report")

    for res in all_results:
        if not res.get("success"):
            with st.expander(f"Analysis Failed: {res.get('filename', 'N/A')}"):
                st.error(res.get("error", "Unknown error"))
            continue

        with st.expander(f"Detailed Report: {res['filename']}", expanded=True):
            full = res["full"]
            partial = res["partial"]
            decision = res["decision"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Basic Information")
                st.write(f"Data Type: {res['data_type']}")
                st.write(f"Initial Flux: {res['J0']:.6g}")
                st.write(f"Flux-Based Threshold: {res['flux_ratio'] * 100:.0f}%")
                st.write(f"Flux-Based Cleaning Time: {res['flux_cleaning_time']:.2f} s")

            with col2:
                st.subheader("Full-Process Fitting")
                st.write(f"R2: {metric_text(full['metrics']['R2'])}")
                st.write(f"NRMSE: {metric_text(full['metrics']['NRMSE'])}")
                st.write(f"MAPE: {metric_text(full['metrics']['MAPE'])}")
                st.write(f"Dominant Mechanism: {full['dominant_mechanism']}")

            with col3:
                st.subheader("Backwashing Decision")
                if decision["triggered"]:
                    st.error(decision["message"])
                else:
                    st.info(decision["message"])
                st.write(f"Recommended Time: {decision['decision_time']:.2f} s")
                st.write(f"Recommended Stage: {decision['decision_stage']}")

            st.markdown("#### Decision Basis")
            st.write(decision["basis"])

            st.markdown("---")
            st.subheader("Three-Stage Mechanism Analysis")
            stage_df = make_stage_table(res["stages"])
            st.dataframe(stage_df, use_container_width=True)

            st.markdown("---")
            st.subheader("Mechanism Contribution Comparison")

            col1, col2 = st.columns(2)

            with col1:
                draw_mechanism_pie(full["eta"], "Full-Process Mechanism Contribution")
                st.info(res["full_strategy"])

            with col2:
                if partial.get("success"):
                    draw_mechanism_pie(partial["eta"], "100%-70% Stage Mechanism Contribution")
                else:
                    st.warning(partial.get("error", "Insufficient data in the 100%-70% stage."))
                st.info(res["partial_strategy"])

            st.markdown("---")
            st.subheader("Flux Decay Fitting and Cleaning Intervention Point")
            draw_flux_curve(res)


if __name__ == "__main__":
    main()