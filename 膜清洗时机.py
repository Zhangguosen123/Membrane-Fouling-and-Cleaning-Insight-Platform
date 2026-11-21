# -*- coding: utf-8 -*-
"""
Membrane Fouling and Cleaning Insight Platform

本版：
1. 保留0.25T、0.45T、0.65T阶段检查及展示。
2. 仅删除“完全堵塞/中间堵塞 > 1”的冲洗触发规则。
3. 通量阈值范围50%–70% J0，默认50% J0。
4. P基线范围0–1.87%，默认1%。
5. P相对变动幅度范围24.22%–100%，默认25%。
6. 恢复原始相对变动定义：
   abs(P_middle_front - P_early) / P_early
   升高和下降均计入。
7. 保留0.65T阶段兜底，冲洗原因与实际最早事件对应。
8. 滤饼主导属于明确施加的拟合约束，不直接修改饼图比例。

时间输入单位：秒。
T：预处理后保留数据的实际时间跨度。

这是基于完整历史曲线的回溯演示：
全程拟合和T均来自已提供的完整序列。
"""

from pathlib import Path
import io
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.optimize import brentq, minimize


VERSION = "cake-constrained-checkpoints-restored-2026-11-21"

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

COLORS = [
    "#90b4ce",
    "#f6bd60",
    "#e58f8f",
    "#71b59b",
]

# 参数顺序：Ks, Kc, Kb, Ki, a, b
BOUNDS = [(0.0, 1.0)] * 4 + [(0.05, 1.0)] * 2

LOW = np.array([b[0] for b in BOUNDS])
HIGH = np.array([b[1] for b in BOUNDS])

EPS = 1e-12
MIN_POINTS = 5

GA_POP = 60
GA_GEN = 100
GA_MUT = 0.10
GA_ELITE = 2

SEEDS = (42, 137, 2026)

DOMINANCE_MARGIN = 1e-6

CONSTRAINT_TEXT = (
    "Constrained fit: cake-filtration contribution is required to exceed "
    "each of the other three contributions. This is an imposed model prior."
)


# =============================================================================
# Unified model and mechanism contributions
# =============================================================================

def log_flux_ratio(p, t, j0):
    ks, kc, kb, ki, a, b = p

    return (
        -2 * a * np.log1p(5 * ks * j0 * t)
        -10 * b * kb * t
        -(1 - b) * np.log1p(10 * ki * j0 * t)
        -(1 - a) / 2 * np.log1p(20 * kc * j0 * j0 * t)
    )


def model(p, t, j0):
    return j0 * np.exp(
        np.maximum(log_flux_ratio(p, t, j0), -700)
    )


def shares(p, t_end, j0):
    """
    四种机制的累积贡献。
    使用各机制对 -log(J/J0) 的解析积分。
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

    # 滤饼贡献必须大于其余三个单项贡献。
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
            if variance_sum > EPS
            else np.nan
        ),
        "NRMSE": (
            np.sqrt(mse) / spread
            if spread > EPS
            else np.nan
        ),
        "MAPE": float(
            np.mean(
                np.abs(y - yh)
                / np.maximum(np.abs(y), floor)
            )
        ),
    }


# =============================================================================
# Constrained fitting
# =============================================================================

def fit_curve(t, y):
    """
    多随机种子遗传算法 + SLSQP约束优化。

    仅在满足滤饼主导约束的候选解中，
    根据归一化均方误差选择最佳结果。

    不保证求得数学上的全局最优解。
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
            candidates.append(
                (loss(p), p.copy(), origin)
            )

    def repair(p):
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

    decline = max(j0 / float(y[-1]), 1.0001)

    kc_est = np.clip(
        (decline ** 2 - 1)
        / (20 * j0 * j0 * duration),
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
            repair(
                LOW + (HIGH - LOW) * rng.random(6)
            )
            for _ in range(GA_POP)
        ])

        pop[:len(pure_starts)] = np.array(pure_starts)

        for _ in range(GA_GEN):
            scores = np.array([
                loss(p) for p in pop
            ])

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

        # 无论优化器状态如何，均独立检查最终解是否满足约束。
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
    pred = model(p, t, j0)

    assert np.argmax(eta) == 3
    assert np.all(eta >= 0.0)

    return {
        "params": p,
        "eta": eta,
        "pred": pred,
        "metrics": metrics(y, pred),
        "fit_origin": origin,
        "fit_loss": best_loss,
        "constraint": CONSTRAINT_TEXT,
    }


# =============================================================================
# Input and preprocessing
# =============================================================================

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

    for folder in (
        base,
        base / "data",
        base / "model_data",
        base / "模型数据",
    ):
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

        key = key.replace(
            "（", "("
        ).replace(
            "）", ")"
        )

        if key in {
            "时间s",
            "时间(s)",
            "times",
            "time(s)",
            "time",
            "t",
            "时间",
        }:
            rename[col] = "time"

        elif key in {
            "实际通量",
            "通量",
            "flux",
            "j",
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

    d = d[
        np.isfinite(d["time"])
        & np.isfinite(d["flux"])
        & (d["flux"] > 0)
    ]

    d = (
        d.groupby(
            "time",
            as_index=False,
        )["flux"]
        .mean()
        .sort_values("time")
    )

    # 保留原程序末尾1%截除处理。
    if len(d) > 10:
        retained = max(
            int(round(len(d) * 0.99)),
            MIN_POINTS,
        )
        d = d.iloc[:retained]

    if len(d) < MIN_POINTS:
        raise ValueError(
            "At least five valid observations are required."
        )

    absolute_start = float(
        d["time"].iloc[0]
    )

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


# =============================================================================
# Window analysis and flux crossing
# =============================================================================

def analyze_window(
    t,
    y,
    start,
    end,
    label,
    key,
):
    mask = (
        (t >= start)
        & (t <= end)
    )

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
        result.update(
            fit_curve(
                ts - ts[0],
                ys,
            )
        )

        result.update(
            success=True,
            P=float(
                result["eta"][1:3].sum()
            ),
        )

    except ValueError as exc:
        result["error"] = str(exc)

    return result


def flux_crossing(
    p,
    duration,
    j0,
    ratio,
):
    if (
        float(model(p, duration, j0))
        > ratio * j0
    ):
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


# =============================================================================
# Backwashing decision
# =============================================================================

def decide(
    stages,
    flux_time,
    flux_ratio,
    p_min,
    change_min,
    duration,
):
    """
    保留原阶段规则：
    F：通量阈值。
    E-P：0.25T早期P基线。
    MF-P-CHANGE：0.45T相对变动。
    M-END：0.65T兜底。

    完全/中间比值不参与决策。
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
            if flux_time is not None
            else np.nan
        ),
    })

    if flux_time is not None:
        candidates.append({
            "code": "F",
            "time": flux_time,
            "stage": "Hard flux threshold",
            "reason": (
                "Hard flux threshold: fitted J/J0 "
                f"reaches {flux_ratio:.0%}."
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
            "Condition": (
                "Early reference unavailable"
            ),
            "Reached": False,
            "Time (min)": np.nan,
        })

    else:
        p0 = early["P"]
        low_p = p0 < p_min

        checks.append({
            "Rule": "P-BASELINE",
            "Condition": (
                f"Early P={p0:.4%} "
                f"< {p_min:.2%}"
            ),
            "Reached": low_p,
            "Time (min)": (
                early["available_time"] / 60
            ),
        })

        if low_p:
            candidates.append({
                "code": "E-P",
                "time": early["available_time"],
                "stage": (
                    "Early endpoint (0.25T)"
                ),
                "reason": (
                    f"Early P={p0:.4%} is below "
                    f"the {p_min:.2%} baseline."
                ),
            })

        current = valid.get("middle_front")

        if current is not None:
            # 恢复原程序定义：
            # 升高与下降均计入非负相对变动。
            change = (
                abs(current["P"] - p0) / p0
                if p0 > EPS
                else None
            )

            reached = (
                change is not None
                and change + 1e-12 >= change_min
            )

            if change is not None:
                detail = (
                    f"P: {p0:.4%} "
                    f"-> {current['P']:.4%}; "
                    f"relative change={change:.2%}; "
                    f"threshold={change_min:.2%}"
                )
            else:
                detail = (
                    "Relative change is undefined: "
                    "fitted early P is zero."
                )

            checks.append({
                "Rule": "MF-P-CHANGE",
                "Condition": detail,
                "Reached": reached,
                "Time (min)": (
                    current["available_time"] / 60
                ),
            })

            if reached:
                candidates.append({
                    "code": "MF-P-CHANGE",
                    "time": (
                        current["available_time"]
                    ),
                    "stage": (
                        "Middle first-half endpoint "
                        "(0.45T)"
                    ),
                    "reason": detail,
                })

    # 恢复原0.65T兜底。
    fallback_time = 0.65 * duration

    candidates.append({
        "code": "M-END",
        "time": fallback_time,
        "stage": (
            "Middle endpoint (0.65T)"
        ),
        "reason": (
            "Scheduled backwashing at the retained "
            "0.65T checkpoint. No earlier configured "
            "trigger was reached."
        ),
    })

    first = min(
        c["time"]
        for c in candidates
    )

    winners = [
        c
        for c in candidates
        if abs(c["time"] - first) <= 1e-6
    ]

    # 实际触发与兜底恰好重合时，只将实际条件列为原因。
    active = [
        c
        for c in winners
        if c["code"] != "M-END"
    ]

    if active:
        winners = active

    checks.append({
        "Rule": "M-END",
        "Condition": (
            "Scheduled fallback at 0.65T"
        ),
        "Reached": (
            winners[0]["code"] == "M-END"
        ),
        "Time (min)": fallback_time / 60,
    })

    return {
        "time": first,
        "codes": " + ".join(
            dict.fromkeys(
                c["code"]
                for c in winners
            )
        ),
        "stage": " + ".join(
            dict.fromkeys(
                c["stage"]
                for c in winners
            )
        ),
        "reason": (
            " Simultaneous trigger: ".join(
                c["reason"]
                for c in winners
            )
        ),
        "checks": checks,
        "candidates": candidates,
    }


# =============================================================================
# Complete dataset analysis
# =============================================================================

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
    change,
    version,
):
    t, y, absolute_start = clean_data(raw)

    full = fit_curve(t, y)

    duration = float(t[-1])
    j0 = float(y[0])

    stage_definitions = [
        (
            0.0,
            0.25,
            "Early: 0-25% Tdata",
            "early",
        ),
        (
            0.25,
            0.45,
            "Middle front: 25-45% Tdata",
            "middle_front",
        ),
        (
            0.45,
            0.65,
            "Middle back: 45-65% Tdata",
            "middle_back",
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
        for start, end, label, key
        in stage_definitions
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
        change,
        duration,
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
        if ft is not None
        else None
    )

    if (
        ft is not None
        and abs(decision["time"] - ft) <= 1e-6
    ):
        decision_window = threshold

    else:
        decision_window = analyze_window(
            t,
            y,
            0.0,
            decision["time"],
            "Recommended-time window",
            "decision",
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
            change,
        ),
        "version": version,
    }


# =============================================================================
# Display helpers
# =============================================================================

def time_text(t):
    if t is None:
        return "Not reached"

    return (
        f"{t / 60:.2f} min "
        f"({t:.2f} s)"
    )


def strategy(result):
    if (
        result is None
        or not result.get("success", True)
    ):
        return (
            "No valid mechanism estimate is "
            "available for this window."
        )

    i = int(np.argmax(result["eta"]))

    if i == 3:
        return (
            "Cake filtration is the largest fitted "
            "contribution. Hydraulic backwashing is "
            "the first cleaning option to evaluate. "
            "Actual recovery and any chemical-cleaning "
            "choice require separate tests and "
            "membrane-compatibility checks."
        )

    return (
        "Select a cleaning strategy using "
        "independent recovery measurements."
    )


def pie(result, title):
    st.markdown("**" + title + "**")

    if (
        result is None
        or not result.get("success", True)
    ):
        st.info(
            "No valid fitting window."
            if result is None
            else result["error"]
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
            COLORS[i]
            for i in indices
        ],
        startangle=90,
        textprops={
            "fontsize": 9,
        },
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
                if s["success"]
                else s["error"]
            ),
            "Evaluation time (min)": (
                s["available_time"] / 60
            ),
            "Evaluation time (s)": (
                s["available_time"]
            ),
        }

        if s["success"]:
            row.update({
                name + " (%)": (
                    100 * s["eta"][i]
                )
                for i, name in enumerate(NAMES)
            })

            complete, intermediate = (
                s["eta"][1:3]
            )

            # 只保留描述性显示，不参与冲洗决策。
            if intermediate > EPS:
                diagnostic_ratio = (
                    complete / intermediate
                )
            elif complete > EPS:
                diagnostic_ratio = np.inf
            else:
                diagnostic_ratio = 0.0

            row.update({
                "P (%)": 100 * s["P"],
                "Complete/intermediate (display only)": (
                    diagnostic_ratio
                ),
                "NRMSE": (
                    s["metrics"]["NRMSE"]
                ),
                "MAPE (%)": (
                    100 * s["metrics"]["MAPE"]
                ),
                "R2": (
                    s["metrics"]["R2"]
                ),
                "Observed window start (s)": (
                    s["t_start"]
                ),
                "Observed window end (s)": (
                    s["t_end"]
                ),
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

    st.write(
        "Selected stage: "
        + decision["stage"]
    )

    st.caption(
        "Parameters used: "
        f"J/J0={r['settings'][0]:.0%}; "
        f"P baseline={r['settings'][1]:.2%}; "
        f"P change={r['settings'][2]:.2%}. "
        "Times are elapsed from the first "
        "retained observation."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            "**Data and checkpoints**"
        )

        st.write(
            "Initial flux J0: {:.6g}".format(
                r["J0"]
            )
        )

        st.write(
            "Tdata: "
            + time_text(r["duration"])
        )

        for fraction in (
            0.25,
            0.45,
            0.65,
        ):
            st.write(
                f"{fraction:.2f}T: "
                + time_text(
                    fraction * r["duration"]
                )
            )

    with c2:
        st.markdown(
            "**Full-process fit**"
        )

        for name, value in (
            r["full"]["metrics"].items()
        ):
            st.write(
                f"{name}: {value:.5f}"
            )

        st.write(
            "Largest contribution: "
            "Cake Filtration"
        )

        st.write(
            "Cake contribution: "
            f"{r['full']['eta'][3]:.2%}"
        )

    with c3:
        st.markdown(
            "**Decision basis**"
        )

        st.write(
            "Choose the earliest of the hard "
            "flux threshold, early P baseline, "
            "P relative change at 0.45T, "
            "and 0.65T fallback."
        )

        st.write(
            "The complete/intermediate ratio "
            "is descriptive only; it never "
            "triggers backwashing."
        )

    references = pd.DataFrame([
        {
            "Reference": label,
            "Elapsed time (s)": (
                f * r["duration"]
            ),
            "Elapsed time (min)": (
                f * r["duration"] / 60
            ),
        }
        for label, f in [
            ("Early end: 0.25T", 0.25),
            (
                "Middle first-half end: 0.45T",
                0.45,
            ),
            ("Middle end: 0.65T", 0.65),
            ("Data endpoint: Tdata", 1.0),
        ]
    ])

    st.markdown(
        "**Time references**"
    )

    st.dataframe(
        references,
        use_container_width=True,
    )

    st.markdown(
        "**Rule evaluation and actual trigger**"
    )

    st.dataframe(
        pd.DataFrame(decision["checks"]),
        use_container_width=True,
    )

    st.caption(
        "'Reached' records whether each condition "
        "is met in the retrospective record. "
        "'Actual trigger' reports only the "
        "earliest event; later conditions are "
        "not additional causes of that event. "
        "M-END is a scheduled fallback, "
        "not a mechanism trigger."
    )

    # -------------------------------------------------------------------------
    # Flux figure
    # -------------------------------------------------------------------------

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
        label=(
            "Full-process constrained fit"
        ),
    )

    threshold = r["threshold"]

    if (
        threshold is not None
        and threshold["success"]
    ):
        ax.plot(
            threshold["t"] / 60,
            threshold["pred"] / r["J0"],
            color="#169b62",
            lw=1.8,
            label=(
                "Flux-threshold window fit"
            ),
        )

    for fraction in (
        0.25,
        0.45,
        0.65,
    ):
        x = (
            fraction
            * r["duration"]
            / 60
        )

        ax.axvline(
            x,
            color="#64748b",
            ls=":",
            alpha=0.6,
        )

        ax.text(
            x,
            0.98,
            f"{fraction:.2f}T",
            rotation=90,
            transform=(
                ax.get_xaxis_transform()
            ),
            ha="right",
            va="top",
        )

    for start, end, color, label in [
        (
            0.0,
            0.25,
            "#dcfce7",
            "Early",
        ),
        (
            0.25,
            0.45,
            "#fef9c3",
            "Middle first half",
        ),
        (
            0.45,
            0.65,
            "#fee2e2",
            "Middle second half",
        ),
    ]:
        ax.axvspan(
            start * r["duration"] / 60,
            end * r["duration"] / 60,
            color=color,
            alpha=0.3,
            label=label,
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

    # -------------------------------------------------------------------------
    # Whole-process and threshold-window pies
    # -------------------------------------------------------------------------

    cols = st.columns(2)

    with cols[0]:
        pie(
            r["full"],
            "Full-process mechanism contributions",
        )
        st.write(
            strategy(r["full"])
        )

    with cols[1]:
        pie(
            r["threshold"],
            "Flux-threshold window mechanism contributions",
        )
        st.write(
            strategy(r["threshold"])
        )

    # -------------------------------------------------------------------------
    # Original three diagnostic windows
    # -------------------------------------------------------------------------

    st.markdown(
        "**Industrial-stage mechanism contributions**"
    )

    table = stage_table(r)

    st.dataframe(
        table,
        use_container_width=True,
    )

    st.caption(
        "Complete/intermediate is retained as "
        "a descriptive column only. It is "
        "excluded from every timing decision."
    )

    cols = st.columns(3)

    for col, s in zip(
        cols,
        r["stages"],
    ):
        with col:
            pie(s, s["label"])

    with st.expander(
        "Contributions up to the selected backwashing time"
    ):
        pie(
            r["decision_window"],
            "Recommended-time window",
        )

        st.write(
            strategy(r["decision_window"])
        )

    # -------------------------------------------------------------------------
    # Downloads
    # -------------------------------------------------------------------------

    st.download_button(
        "Download stage results CSV",
        table.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=(
            r["type"]
            + "_stage_results.csv"
        ),
        mime="text/csv",
        key=(
            "stages_"
            + r["type"]
            + "_"
            + r["file"]
        ),
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
        "stages": (
            stage_table(r)
            .to_dict("records")
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


# =============================================================================
# Streamlit interface
# =============================================================================

def main():
    st.set_page_config(
        page_title=TITLE,
        page_icon="💧",
        layout="wide",
    )

    st.title(TITLE)

    st.info(
        CONSTRAINT_TEXT
    )

    st.caption(
        "Retrospective demonstration. "
        "P=complete+intermediate blocking. "
        "Baseline and relative-change thresholds "
        "are empirical settings. All four foulant "
        "types use the same configured rules, "
        "with the retained 0.65T scheduled fallback."
    )

    ratio = st.sidebar.slider(
        "Hard flux threshold (% of initial flux)",
        min_value=50.0,
        max_value=70.0,
        value=50.0,
        step=1.0,
        help=(
            "70% J0 means a 30% flux decline; "
            "50% J0 means a 50% decline."
        ),
    ) / 100

    p_min = st.sidebar.slider(
        "Early Pore-Mouth Blocking Minimum (%)",
        min_value=0.0,
        max_value=1.87,
        value=1.0,
        step=0.01,
        format="%.2f",
        help=(
            "Early P below this value independently "
            "triggers at the early endpoint, 0.25T, "
            "unless an earlier flux trigger occurs."
        ),
    ) / 100

    change = st.sidebar.slider(
        "Pore-Mouth Blocking Relative Change Trigger (%)",
        min_value=24.22,
        max_value=100.0,
        value=25.0,
        step=0.01,
        format="%.2f",
        help=(
            "At 0.45T only: "
            "abs(P_middle_front-P_early)/P_early. "
            "Increases and decreases both count. "
            "The denominator is the fitted early P, "
            "not the baseline setting."
        ),
    ) / 100

    st.sidebar.caption(
        "Checkpoints: 0.25T, 0.45T, 0.65T. "
        "T is the retained time-series duration "
        "after preprocessing. No separate "
        "late-stage timing trigger is added."
    )

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
        change,
        mode,
        kind,
        (
            upload.getvalue()
            if upload is not None
            else None
        ),
    )

    if st.sidebar.button(
        "Clear results"
    ):
        st.session_state.pop(
            "analysis_bundle",
            None,
        )

    if st.button(
        "Start analysis"
    ):
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
            st.error(
                "Select a CSV first."
            )
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
                        raw, filename = (
                            load_preset(selected)
                        )

                    result = analyze(
                        raw,
                        selected,
                        filename,
                        ratio,
                        p_min,
                        change,
                        VERSION,
                    )

                    output.append(result)

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

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = []

    for r in bundle[1]:
        if "error" in r:
            st.error(
                r["type"]
                + ": "
                + r["error"]
            )
            continue

        summary.append({
            "Foulant": r["type"],
            "File": r["file"],
            "Initial flux J0": r["J0"],
            "Tdata (s)": r["duration"],
            "0.25T (s)": (
                0.25 * r["duration"]
            ),
            "0.45T (s)": (
                0.45 * r["duration"]
            ),
            "0.65T (s)": (
                0.65 * r["duration"]
            ),
            "Selected stage": (
                r["decision"]["stage"]
            ),
            "Recommended time (min)": (
                r["decision"]["time"] / 60
            ),
            "Recommended time (s)": (
                r["decision"]["time"]
            ),
            "Actual trigger": (
                r["decision"]["codes"]
            ),
            "Reason": (
                r["decision"]["reason"]
            ),
            "Flux-threshold time (min)": (
                r["flux_time"] / 60
                if r["flux_time"] is not None
                else np.nan
            ),
            "Flux threshold (%)": (
                100 * ratio
            ),
            "P baseline (%)": (
                100 * p_min
            ),
            "P relative-change threshold (%)": (
                100 * change
            ),
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
        table = pd.DataFrame(
            summary
        )

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

    # -------------------------------------------------------------------------
    # Detailed reports
    # -------------------------------------------------------------------------

    for r in bundle[1]:
        if "error" not in r:
            with st.expander(
                r["type"] + " detailed report",
                expanded=True,
            ):
                show_result(r)


if __name__ == "__main__":
    main()
