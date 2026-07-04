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
    "Mixture": "Mixturedata1.csv",
}

PRESET_RAW_URLS = {
    "Mixturedata1.csv": (
        "https://raw.githubusercontent.com/Zhangguosen123/"
        "Membrane-Fouling-and-Cleaning-Insight-Platform/main/Mixturedata1.csv"
    )
}

DEFAULT_FORCED_FLUX_RATIO = 0.70
DEFAULT_INTERMEDIATE_MIN = 0.05
DEFAULT_INTERMEDIATE_DROP = 0.20
COMPLETE_TO_INTERMEDIATE_TRIGGER = 1.0

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

INDUSTRIAL_CHECKPOINTS = {
    "early_end": 0.25,
    "middle_front_end": 0.45,
    "middle_end": 0.65,
    "late_end": 1.00,
}

MECH_NAMES = [
    "Standard Blocking",
    "Complete Blocking",
    "Intermediate Blocking",
    "Cake Filtration",
]


# ==============================================================================
# Numerical Helpers
# ==============================================================================
def trapezoid_integral(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def percent(x):
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def metric_text(x):
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.3f}"


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
    if len(t) < MIN_STAGE_POINTS:
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
        val = -trapezoid_integral(si, t)
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

    if filename in PRESET_RAW_URLS:
        return PRESET_RAW_URLS[filename]

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

    if len(J_clean) < MIN_STAGE_POINTS:
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
# Metrics, Stages, and Cleaning Decision
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


def build_observed_schedule(t):
    t0 = float(t[0])
    t_end = float(t[-1])
    Tdata = t_end - t0

    if Tdata <= 0:
        raise ValueError("The time-series duration must be positive.")

    return {
        "t0": t0,
        "Tdata": Tdata,
        "early_end": t0 + INDUSTRIAL_CHECKPOINTS["early_end"] * Tdata,
        "middle_front_end": t0 + INDUSTRIAL_CHECKPOINTS["middle_front_end"] * Tdata,
        "middle_end": t0 + INDUSTRIAL_CHECKPOINTS["middle_end"] * Tdata,
        "late_end": t_end,
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
        "complete_ratio": float(eta[1]),
        "cake_ratio": float(eta[3]),
        "complete_to_intermediate_ratio": (
            float(eta[1] / eta[2]) if eta[2] > EPS else np.inf if eta[1] > EPS else 0.0
        ),
        "t_start": float(t_abs[0]),
        "t_end": float(t_abs[-1]),
        "flux_start": float(J_obs[0]),
        "flux_end": float(J_obs[-1]),
    }


def slice_time_window(t, J, start_time, end_time):
    mask = (t >= start_time) & (t <= end_time)
    t_seg = t[mask]
    J_seg = J[mask]

    if len(t_seg) >= MIN_STAGE_POINTS:
        return t_seg, J_seg

    idx_start = int(np.searchsorted(t, start_time, side="left"))
    idx_end = int(np.searchsorted(t, end_time, side="right"))

    idx_start = max(0, min(idx_start, len(t) - 1))
    idx_end = max(idx_start + MIN_STAGE_POINTS, idx_end)
    idx_end = min(idx_end, len(t))

    return t[idx_start:idx_end], J[idx_start:idx_end]


def build_industrial_stage_analysis(t, J, observed_schedule):
    windows = [
        (
            "Early stage diagnosis (0-25%Tdata)",
            "0-25%Tdata",
            observed_schedule["t0"],
            observed_schedule["early_end"],
            "early",
        ),
        (
            "Middle stage first-half diagnosis (25-45%Tdata)",
            "25-45%Tdata",
            observed_schedule["early_end"],
            observed_schedule["middle_front_end"],
            "middle_front",
        ),
        (
            "Middle stage second-half diagnosis (45-65%Tdata)",
            "45-65%Tdata",
            observed_schedule["middle_front_end"],
            observed_schedule["middle_end"],
            "middle_back",
        ),
    ]

    results = []
    for label, ratio_text, start_time, end_time, key in windows:
        t_seg, J_seg = slice_time_window(t, J, start_time, end_time)
        res = analyze_interval(t_seg, J_seg, label)
        res["stage_ratio"] = ratio_text
        res["target_start"] = float(start_time)
        res["target_end"] = float(end_time)
        res["stage_key"] = key
        results.append(res)

    return results


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


def find_flux_threshold_time_from_fit(params, J0, t0, Tdata, flux_ratio):
    t_rel_grid = np.linspace(0.0, float(Tdata), 800)
    J_grid = stage1_model(params, t_rel_grid, J0)
    target_flux = J0 * flux_ratio

    crossing_rel_time, crossing_idx = find_time_when_curve_below(t_rel_grid, J_grid, target_flux)

    if crossing_rel_time is None:
        return float(t0 + Tdata), float(J_grid[-1]), len(t_rel_grid) - 1, False

    return float(t0 + crossing_rel_time), float(target_flux), crossing_idx, True


def build_flux_threshold_window(t, J, threshold_time):
    start_time = float(t[0])
    end_time = float(threshold_time)

    if end_time <= start_time:
        end_time = start_time + max((t[-1] - t[0]) * 0.05, EPS)

    t_seg, J_seg = slice_time_window(t, J, start_time, min(end_time, float(t[-1])))
    return analyze_interval(t_seg, J_seg, "Flux-threshold window")


def recommend_cleaning_strategy(eta):
    dominant_idx = int(np.argmax(eta))
    dominant = MECH_NAMES[dominant_idx]
    dominant_ratio = eta[dominant_idx] * 100

    if dominant_idx == 3:
        return (
            f"Dominant mechanism: {dominant} ({dominant_ratio:.1f}%). "
            f"Recommended action: hydraulic backwashing is preferred. "
            f"If flux recovery is limited, low-dose oxidative or alkaline cleaning may be considered."
        )

    if dominant_idx in (0, 1):
        return (
            f"Dominant mechanism: {dominant} ({dominant_ratio:.1f}%). "
            f"Recommended action: backwashing combined with mild acid or chelating cleaning. "
            f"Pore-related fouling is more likely to cause irreversible residues, so delayed cleaning should be avoided."
        )

    if dominant_idx == 2:
        return (
            f"Dominant mechanism: {dominant} ({dominant_ratio:.1f}%). "
            f"Recommended action: timely backwashing to disrupt pore-entrance bridging and the cake-membrane interface."
        )

    return "Mixed fouling mechanisms are involved. A combined mild cleaning protocol is recommended."


def decide_backwash_timing(stage_results,
                           observed_schedule,
                           flux_threshold_time,
                           flux_threshold_reached,
                           flux_ratio,
                           intermediate_threshold,
                           intermediate_drop_threshold):
    stage_by_key = {s.get("stage_key"): s for s in stage_results if s.get("success")}

    def cap_by_flux(candidate_time, candidate_stage, candidate_message, candidate_basis, rule_code):
        if flux_threshold_reached and flux_threshold_time <= candidate_time:
            return {
                "triggered": True,
                "decision_time": float(flux_threshold_time),
                "decision_stage": "Forced flux-threshold backwash",
                "level": "danger",
                "rule_code": "F",
                "message": (
                    f"Forced backwashing is recommended at {flux_threshold_time:.2f} s because "
                    f"the fitted flux reaches {flux_ratio * 100:.1f}% of the initial flux."
                ),
                "basis": (
                    "The industrial flux-decline threshold has priority over all mechanism-based rules. "
                    "Once membrane flux reaches the configured 50-70% J0 threshold, the filtration cycle should not continue."
                ),
            }

        return {
            "triggered": True,
            "decision_time": float(candidate_time),
            "decision_stage": candidate_stage,
            "level": "warning",
            "rule_code": rule_code,
            "message": candidate_message,
            "basis": candidate_basis,
        }

    early_end = observed_schedule["early_end"]
    mid_front_end = observed_schedule["middle_front_end"]
    mid_end = observed_schedule["middle_end"]

    if flux_threshold_reached and flux_threshold_time <= early_end:
        return cap_by_flux(
            early_end,
            "Forced flux-threshold backwash",
            "",
            "",
            "F",
        )

    early = stage_by_key.get("early")
    middle_front = stage_by_key.get("middle_front")
    middle_back = stage_by_key.get("middle_back")

    if early is not None:
        if early["complete_to_intermediate_ratio"] > COMPLETE_TO_INTERMEDIATE_TRIGGER:
            return cap_by_flux(
                early_end,
                "Early-stage backwash",
                (
                    f"Backwashing is recommended at the actual early-stage endpoint ({early_end:.2f} s). "
                    f"The complete/intermediate blocking ratio is {early['complete_to_intermediate_ratio']:.2f}, above 1.00."
                ),
                (
                    "Complete blocking exceeds intermediate blocking in the early 25% data window. "
                    "This indicates stronger pore blocking risk, so early intervention is preferred."
                ),
                "E-CI",
            )

        if early["intermediate_ratio"] < intermediate_threshold:
            return cap_by_flux(
                early_end,
                "Early-stage backwash",
                (
                    f"Backwashing is recommended at the actual early-stage endpoint ({early_end:.2f} s). "
                    f"The early intermediate-blocking contribution is {early['intermediate_ratio'] * 100:.2f}%, "
                    f"below the {intermediate_threshold * 100:.2f}% threshold."
                ),
                (
                    "Low intermediate blocking in the early 25% data window suggests that interfacial bridging is weak. "
                    "Early hydraulic backwashing is preferred before fouling consolidates."
                ),
                "E-IB",
            )

    if middle_front is not None:
        if middle_front["complete_to_intermediate_ratio"] > COMPLETE_TO_INTERMEDIATE_TRIGGER:
            return cap_by_flux(
                mid_front_end,
                "Middle first-half backwash",
                (
                    f"Backwashing is recommended at the middle first-half endpoint ({mid_front_end:.2f} s). "
                    f"The complete/intermediate blocking ratio is {middle_front['complete_to_intermediate_ratio']:.2f}, above 1.00."
                ),
                (
                    "The complete/intermediate blocking ratio exceeds 1 during the first half of the middle stage. "
                    "The cycle should stop at this checkpoint unless the flux threshold is reached earlier."
                ),
                "MF-CI",
            )

        if early is not None:
            drop = early["intermediate_ratio"] - middle_front["intermediate_ratio"]
            if drop >= intermediate_drop_threshold:
                return cap_by_flux(
                    mid_front_end,
                    "Middle first-half backwash",
                    (
                        f"Backwashing is recommended at the middle first-half endpoint ({mid_front_end:.2f} s). "
                        f"Intermediate blocking decreases by {drop * 100:.2f} percentage points from early stage "
                        f"to the first half of middle stage."
                    ),
                    (
                        "A clear decline in intermediate blocking suggests a transition from pore-entrance/interfacial "
                        "blocking to cake-layer control. Backwashing at this checkpoint helps avoid cake compaction."
                    ),
                    "MF-IB-DROP",
                )

    if middle_back is not None:
        if middle_back["complete_to_intermediate_ratio"] > COMPLETE_TO_INTERMEDIATE_TRIGGER:
            return cap_by_flux(
                mid_end,
                "Middle second-half backwash",
                (
                    f"Backwashing is recommended at the middle endpoint ({mid_end:.2f} s). "
                    f"The complete/intermediate blocking ratio is {middle_back['complete_to_intermediate_ratio']:.2f}, above 1.00."
                ),
                (
                    "The complete/intermediate blocking ratio exceeds 1 in the second half of the middle stage. "
                    "Backwashing should be performed by the 0.65Tdata checkpoint."
                ),
                "MB-CI",
            )

    return cap_by_flux(
        mid_end,
        "Middle-end backwash",
        (
            f"No earlier mechanism trigger is stronger than the control rules. "
            f"Backwashing is recommended at the middle-stage endpoint ({mid_end:.2f} s)."
        ),
        (
            "The operation remains bounded by the actual-data 0.65Tdata checkpoint and the forced flux threshold. "
            "Late-stage backwashing is not considered in this industrial strategy."
        ),
        "M-END",
    )


def analyze_dataset(t, J, J0, filename, data_type,
                    flux_ratio,
                    intermediate_threshold,
                    intermediate_drop_threshold):
    observed_schedule = build_observed_schedule(t)

    t_rel = t - t[0]
    full_params = fit_model(t_rel, J, J0)
    full_pred = stage1_model(full_params, t_rel, J0)
    full_eta = calculate_mechanism_contribution(full_params, t_rel, J0)
    full_metrics = calculate_metrics(J, full_pred)

    full_dom_idx = int(np.argmax(full_eta))
    full = {
        "params": full_params,
        "J_pred": full_pred,
        "eta": full_eta,
        "metrics": full_metrics,
        "dominant_idx": full_dom_idx,
        "dominant_mechanism": MECH_NAMES[full_dom_idx],
        "dominant_ratio": float(full_eta[full_dom_idx]),
    }

    flux_threshold_time, flux_threshold_flux, _, flux_threshold_reached = find_flux_threshold_time_from_fit(
        full_params,
        J0,
        observed_schedule["t0"],
        observed_schedule["Tdata"],
        flux_ratio,
    )

    threshold_stage = build_flux_threshold_window(t, J, flux_threshold_time)
    industrial_stages = build_industrial_stage_analysis(t, J, observed_schedule)

    decision = decide_backwash_timing(
        industrial_stages,
        observed_schedule,
        flux_threshold_time,
        flux_threshold_reached,
        flux_ratio,
        intermediate_threshold,
        intermediate_drop_threshold,
    )

    return {
        "success": True,
        "filename": filename,
        "data_type": data_type,
        "t": t,
        "J": J,
        "t_rel": t_rel,
        "J0": J0,
        "observed_schedule": observed_schedule,
        "full": full,
        "fixed_stages": industrial_stages,
        "threshold_stage": threshold_stage,
        "decision": decision,
        "flux_threshold_time": flux_threshold_time,
        "flux_threshold_flux": flux_threshold_flux,
        "flux_threshold_reached": flux_threshold_reached,
        "flux_ratio": flux_ratio,
        "intermediate_threshold": intermediate_threshold,
        "intermediate_drop_threshold": intermediate_drop_threshold,
        "full_strategy": recommend_cleaning_strategy(full["eta"]),
        "threshold_strategy": (
            recommend_cleaning_strategy(threshold_stage["eta"])
            if threshold_stage.get("success")
            else "The flux-threshold operation window contains insufficient valid data."
        ),
    }


def analyze_preset_file(data_type, flux_ratio, intermediate_threshold, intermediate_drop_threshold):
    t, J, J0, filename, _ = load_preset_data(data_type)
    return analyze_dataset(
        t, J, J0, filename, data_type,
        flux_ratio,
        intermediate_threshold,
        intermediate_drop_threshold,
    )


def analyze_uploaded_file(uploaded_file, data_type, flux_ratio, intermediate_threshold, intermediate_drop_threshold):
    df, enc = read_csv_robust(uploaded_file)
    t, J, J0, filename, _ = dataframe_to_series(df, uploaded_file.name, enc)
    return analyze_dataset(
        t, J, J0, filename, data_type,
        flux_ratio,
        intermediate_threshold,
        intermediate_drop_threshold,
    )


# ==============================================================================
# Visualization Helpers
# ==============================================================================
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


def make_fixed_stage_table(stage_results):
    rows = []

    for s in stage_results:
        if s.get("success"):
            rows.append({
                "Stage": s["label"],
                "Fixed Ratio": s["stage_ratio"],
                "Start Time (s)": round(s["t_start"], 2),
                "End Time (s)": round(s["t_end"], 2),
                "Standard Blocking": percent(s["eta"][0]),
                "Complete Blocking": percent(s["eta"][1]),
                "Intermediate Blocking": percent(s["eta"][2]),
                "Cake Filtration": percent(s["eta"][3]),
                "Complete/Intermediate": (
                    "Inf" if np.isinf(s["complete_to_intermediate_ratio"])
                    else f"{s['complete_to_intermediate_ratio']:.3f}"
                ),
                "Dominant Mechanism": f"{s['dominant_mechanism']} ({s['dominant_ratio'] * 100:.1f}%)",
                "NRMSE": s["metrics"]["NRMSE"],
                "MAPE": s["metrics"]["MAPE"],
                "Status": "Completed",
            })
        else:
            rows.append({
                "Stage": s.get("label", "N/A"),
                "Fixed Ratio": s.get("stage_ratio", "N/A"),
                "Start Time (s)": "N/A",
                "End Time (s)": "N/A",
                "Standard Blocking": "N/A",
                "Complete Blocking": "N/A",
                "Intermediate Blocking": "N/A",
                "Cake Filtration": "N/A",
                "Complete/Intermediate": "N/A",
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
    threshold_stage = res["threshold_stage"]
    decision = res["decision"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, J, "o", ms=3, color="gray", alpha=0.6, label="Observed Flux")
    ax.plot(t, full["J_pred"], "-", lw=2.2, color="#d97706", label="Full-Process Fitting")

    if threshold_stage.get("success"):
        ax.plot(
            threshold_stage["t_abs"],
            threshold_stage["J_pred"],
            "-",
            lw=2.2,
            color="#16a34a",
            label=f"Flux-Threshold Window Fitting ({res['flux_ratio'] * 100:.0f}% J0)",
        )

    flux_line = res["J0"] * res["flux_ratio"]
    ax.axhline(
        y=flux_line,
        color="#dc2626",
        linestyle=":",
        lw=2,
        label=f"Forced {res['flux_ratio'] * 100:.0f}% Initial Flux Reference",
    )

    ax.axvline(
        x=res["flux_threshold_time"],
        color="#dc2626",
        linestyle="--",
        alpha=0.7,
        label="Forced Flux-Threshold Deadline",
    )

    ax.axvline(
        x=decision["decision_time"],
        color="#7c3aed",
        linestyle="-.",
        lw=2.4,
        label="Recommended Industrial Backwash Point",
    )

    schedule = res["observed_schedule"]
    for label, x in [
        ("0.25Tdata", schedule["early_end"]),
        ("0.45Tdata", schedule["middle_front_end"]),
        ("0.65Tdata", schedule["middle_end"]),
    ]:
        ax.axvline(x=x, color="#64748b", linestyle=":", alpha=0.5)
        ax.text(
            x,
            ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else np.max(J),
            label,
            rotation=90,
            va="top",
            ha="right",
        )

    stage_colors = ["#dcfce7", "#fef9c3", "#fee2e2"]
    for i, s in enumerate(res["fixed_stages"]):
        if s.get("success"):
            ax.axvspan(
                s["t_start"],
                s["t_end"],
                color=stage_colors[i],
                alpha=0.25,
                label=s["stage_ratio"] + " Stage" if i == 0 else None,
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

    flux_ratio_percent = st.sidebar.slider(
        "Forced Backwash Flux Threshold (% of initial flux)",
        min_value=50.0,
        max_value=70.0,
        value=70.0,
        step=1.0,
        help=(
            "Industrial hard limit. When fitted flux reaches this percentage of J0, "
            "backwashing is forced regardless of mechanism diagnosis."
        ),
    )
    flux_ratio = flux_ratio_percent / 100.0

    intermediate_threshold_percent = st.sidebar.slider(
        "Early Intermediate-Blocking Minimum (%)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help=(
            "If the early-stage intermediate-blocking contribution is below this value, "
            "backwashing is recommended at 0.25Tdata unless the flux threshold is reached earlier."
        ),
    )
    intermediate_threshold = intermediate_threshold_percent / 100.0

    intermediate_drop_percent = st.sidebar.slider(
        "Intermediate-Blocking Decline Trigger (%)",
        min_value=5.0,
        max_value=35.0,
        value=20.0,
        step=5.0,
        help=(
            "If intermediate blocking decreases by more than this percentage from early stage "
            "to the first half of middle stage, backwashing is recommended at 0.45Tdata."
        ),
    )
    intermediate_drop_threshold = intermediate_drop_percent / 100.0

    st.sidebar.caption(
        "Stage checkpoints are calculated only from the actual time-series duration: "
        "0.25Tdata, 0.45Tdata, and 0.65Tdata."
    )

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
        data_type = st.selectbox("Select Data Type", ["BSA", "HA", "SA", "Mixture"])

        st.caption(f"Preset file: {PRESET_FILES[data_type]}")

        if st.button("Start Analysis"):
            with st.spinner(f"Analyzing {PRESET_FILES[data_type]} ..."):
                try:
                    result = analyze_preset_file(
                        data_type,
                        flux_ratio,
                        intermediate_threshold,
                        intermediate_drop_threshold,
                    )
                    st.session_state.all_results = [result]
                except Exception as e:
                    st.error(str(e))

    elif analysis_mode == "Batch Analysis of Preset Files":
        st.header("Batch Analysis of Preset Files")
        st.info("The app will analyze BSAdata1.csv, HAdata1.csv, SAdata1.csv, and Mixturedata1.csv.")

        if st.button("Start Batch Analysis"):
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            batch_types = ["BSA", "HA", "SA", "Mixture"]
            for i, data_type in enumerate(batch_types):
                status_text.text(f"Analyzing ({i + 1}/{len(batch_types)}): {PRESET_FILES[data_type]}")
                try:
                    result = analyze_preset_file(
                        data_type,
                        flux_ratio,
                        intermediate_threshold,
                        intermediate_drop_threshold,
                    )
                except Exception as e:
                    result = {
                        "success": False,
                        "filename": PRESET_FILES[data_type],
                        "data_type": data_type,
                        "error": str(e),
                    }

                all_results.append(result)
                progress_bar.progress((i + 1) / len(batch_types))
                time.sleep(0.1)

            status_text.text("Batch analysis completed.")
            st.session_state.all_results = all_results

    else:
        st.header("Upload Custom CSV")
        upload_data_type = st.selectbox("Pollutant Type for Uploaded CSV", ["BSA", "HA", "SA", "Mixture"])
        uploaded_file = st.file_uploader("Upload a CSV file containing time and flux data", type="csv")

        if uploaded_file is not None and st.button("Analyze Uploaded File"):
            with st.spinner(f"Analyzing {uploaded_file.name} ..."):
                try:
                    result = analyze_uploaded_file(
                        uploaded_file,
                        upload_data_type,
                        flux_ratio,
                        intermediate_threshold,
                        intermediate_drop_threshold,
                    )
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
                "Actual Tdata (s)": "N/A",
                "Full-Process Dominant Mechanism": "N/A",
                "Flux Threshold": f"{flux_ratio_percent:.1f}% J0",
                "Intermediate-Blocking Threshold": f"{intermediate_threshold_percent:.1f}%",
                "Intermediate Decline Trigger": f"{intermediate_drop_percent:.1f}%",
                "Mechanism-Guided Cleaning Stage": "Analysis Failed",
                "Recommended Time (s)": "N/A",
                "Flux-Threshold Deadline (s)": "N/A",
                "Status": res.get("error", "Failed"),
            })
            continue

        full = res["full"]
        decision = res["decision"]
        schedule = res["observed_schedule"]

        summary_rows.append({
            "File": res["filename"],
            "Type": res["data_type"],
            "Initial Flux": f"{res['J0']:.6g}",
            "Actual Tdata (s)": f"{schedule['Tdata']:.2f}",
            "Full-Process Dominant Mechanism": f"{full['dominant_mechanism']} ({full['dominant_ratio'] * 100:.1f}%)",
            "Flux Threshold": f"{res['flux_ratio'] * 100:.1f}% J0",
            "Intermediate-Blocking Threshold": f"{intermediate_threshold_percent:.1f}%",
            "Intermediate Decline Trigger": f"{intermediate_drop_percent:.1f}%",
            "Mechanism-Guided Cleaning Stage": decision["decision_stage"],
            "Recommended Time (s)": f"{decision['decision_time']:.2f}",
            "Flux-Threshold Deadline (s)": f"{res['flux_threshold_time']:.2f}",
            "Status": decision.get("rule_code", "Triggered"),
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
            threshold_stage = res["threshold_stage"]
            decision = res["decision"]
            schedule = res["observed_schedule"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Basic Information")
                st.write(f"Data Type: {res['data_type']}")
                st.write(f"Initial Flux: {res['J0']:.6g}")
                st.write(f"Actual Total Time Tdata: {schedule['Tdata']:.2f} s")
                st.write("Stage Checkpoints: 0.25Tdata / 0.45Tdata / 0.65Tdata")
                st.write(f"0.25Tdata Early End: {schedule['early_end']:.2f} s")
                st.write(f"0.45Tdata Middle First-Half End: {schedule['middle_front_end']:.2f} s")
                st.write(f"0.65Tdata Middle End: {schedule['middle_end']:.2f} s")
                st.write(f"Forced Flux Threshold: {res['flux_ratio'] * 100:.1f}% of J0")

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
                st.write(f"Flux-Threshold Deadline: {res['flux_threshold_time']:.2f} s")
                st.write(f"Decision Rule: {decision.get('rule_code', 'N/A')}")

            st.markdown("#### Decision Basis")
            st.write(decision["basis"])

            st.markdown("#### Time Reference")
            time_df = pd.DataFrame([{
                "Pollutant": res["data_type"],
                "Actual Tdata (s)": round(schedule["Tdata"], 2),
                "0.25Tdata Early End (s)": round(schedule["early_end"], 2),
                "0.45Tdata Middle First-Half End (s)": round(schedule["middle_front_end"], 2),
                "0.65Tdata Middle End (s)": round(schedule["middle_end"], 2),
                "Final Time (s)": round(schedule["late_end"], 2),
            }])
            st.dataframe(time_df, use_container_width=True)

            st.markdown("---")
            st.subheader("Industrial Stage Mechanism Analysis")
            fixed_stage_df = make_fixed_stage_table(res["fixed_stages"])
            st.dataframe(fixed_stage_df, use_container_width=True)

            st.markdown("---")
            st.subheader("Mechanism Contribution Pies for Industrial Stages")

            stage_cols = st.columns(3)
            for col, stage_res in zip(stage_cols, res["fixed_stages"]):
                with col:
                    if stage_res.get("success"):
                        draw_mechanism_pie(
                            stage_res["eta"],
                            f"{stage_res['label']}",
                        )
                    else:
                        st.warning(stage_res.get("error", "Stage analysis failed."))

            st.markdown("---")
            st.subheader("Full Process and Flux-Threshold Window")

            col1, col2 = st.columns(2)

            with col1:
                draw_mechanism_pie(full["eta"], "Full-Process Mechanism Contribution")
                st.info(res["full_strategy"])

            with col2:
                if threshold_stage.get("success"):
                    draw_mechanism_pie(
                        threshold_stage["eta"],
                        f"Flux-Threshold Window Contribution ({res['flux_ratio'] * 100:.0f}% J0)",
                    )
                    st.info(res["threshold_strategy"])
                else:
                    st.warning(threshold_stage.get("error", "Insufficient data in flux-threshold window."))

            st.markdown("---")
            st.subheader("Flux Decay Fitting and Cleaning Intervention Point")
            draw_flux_curve(res)


if __name__ == "__main__":
    main()