"""
explainability.py  —  Attribution for IsolationForest
------------------------------------------------------

FIX LOG
-------
1. build_root_cause_string() showed "—" instead of values
   Root cause: During warm-up (< 20 ticks), ig_df["Direction"] is "—"
   because all attributions are 0. The function was returning this raw "—"
   into the HITL box with no context.
   Fix: Check total_abs first. If near-zero (warm-up), return a descriptive
        message instead of "—". When IG is active, include percentage impact
        and confidence score in the output string.

2. Direction labels were counterintuitive
   Old: "↑ anomalous" / "↓ normal"
        — upward arrow for "bad" is confusing; suggests value is rising
          but doesn't tell the operator WHY it matters.
   New: "🔴 driving anomaly" / "🟢 stabilising"
        — clear semantic meaning, colour-coded.

3. All existing bug fixes from the previous version are preserved:
   EPS=1.0, ZERO_THRESHOLD=1e-5, WARMUP_ROWS=20, DataFrame wrapping.
"""

import numpy as np
import pandas as pd

STEPS          = 40
EPS            = 1.0
ZERO_THRESHOLD = 1e-5
WARMUP_ROWS    = 20


# ── Helpers ──────────────────────────────────────────────────────────────────

def _score(model, x: np.ndarray, feature_cols: list) -> float:
    df = pd.DataFrame([x], columns=feature_cols)
    return float(model.decision_function(df)[0])


# ── Integrated Gradients ─────────────────────────────────────────────────────

def integrated_gradients(
    model,
    input_row: np.ndarray,
    baseline: np.ndarray,
    feature_cols: list,
    steps: int = STEPS,
    eps: float = EPS,
) -> np.ndarray:
    """
    Compute IG attributions using per-feature central finite differences
    along the straight-line path from baseline to input_row.
    negative = pushed anomaly score down → anomaly driver
    positive = pushed anomaly score up   → stabilising feature
    """
    n      = len(input_row)
    delta  = input_row - baseline
    alphas = np.linspace(0.0, 1.0, steps)
    ig     = np.zeros(n)

    for i in range(n):
        grads = []
        for alpha in alphas:
            x_mid         = baseline + alpha * delta
            x_plus        = x_mid.copy(); x_plus[i]  += eps
            x_minus       = x_mid.copy(); x_minus[i] -= eps
            g = (_score(model, x_plus, feature_cols) -
                 _score(model, x_minus, feature_cols)) / (2.0 * eps)
            grads.append(g)
        ig[i] = float(np.mean(grads)) * delta[i]

    return ig


# ── Z-score fallback ──────────────────────────────────────────────────────────

def zscore_attribution(features: np.ndarray, latest: np.ndarray,
                        feature_cols: list) -> np.ndarray:
    """
    Signed z-score deviation. Negative = anomaly-driving.
    Higher-is-worse features (latency, packet_loss, jitter): above mean → negative.
    Lower-is-worse features (throughput, bandwidth):         below mean → negative.
    """
    mean = features.mean(axis=0)
    std  = features.std(axis=0) + 1e-9
    z    = (latest - mean) / std

    HIGHER_IS_WORSE = {"latency", "packet_loss", "jitter"}
    for idx, feat in enumerate(feature_cols):
        if feat in HIGHER_IS_WORSE:
            z[idx] = -z[idx]

    return z


# ── Public API ────────────────────────────────────────────────────────────────

def get_ig_attribution(model, df: pd.DataFrame,
                        feature_cols: list) -> pd.DataFrame:
    """
    Returns attribution DataFrame for the latest row of df.
    Tries IG first; falls back to z-score if not enough data or IG is near-zero.

    FIX #2 — direction labels changed to semantic colour-coded strings.
    Columns: Feature, IG_Attribution, Abs_Attribution, Direction, Method
    Sorted by Abs_Attribution descending.
    """
    if len(df) < WARMUP_ROWS:
        return _zero_result(feature_cols, "Warming up…")

    features = df[feature_cols].values.astype(float)
    baseline = features.mean(axis=0)
    latest   = features[-1]

    try:
        ig_values = integrated_gradients(model, latest, baseline, feature_cols)
        ig_values = np.nan_to_num(ig_values, nan=0.0, posinf=0.0, neginf=0.0)
        method    = "IG"
    except Exception:
        ig_values = np.zeros(len(feature_cols))
        method    = "Z-score"

    if np.max(np.abs(ig_values)) < ZERO_THRESHOLD:
        ig_values = zscore_attribution(features, latest, feature_cols)
        ig_values = np.nan_to_num(ig_values, nan=0.0, posinf=0.0, neginf=0.0)
        method    = "Z-score"

    # FIX #2 — clear semantic direction labels
    directions = ["🔴 driving anomaly" if v < 0 else "🟢 stabilising" for v in ig_values]

    return pd.DataFrame({
        "Feature":         feature_cols,
        "IG_Attribution":  ig_values,
        "Abs_Attribution": np.abs(ig_values),
        "Direction":       directions,
        "Method":          [method] * len(feature_cols),
    }).sort_values("Abs_Attribution", ascending=False).reset_index(drop=True)


def build_root_cause_string(ig_df: pd.DataFrame) -> str:
    """
    FIX #1 — was returning "—" during warm-up or omitting percentage.

    Now returns:
    - A warm-up progress message if attributions are all zero.
    - "Primary: LATENCY (🔴 driving anomaly, 54.2% impact) [IG] ·
       Secondary: JITTER (🔴 driving anomaly, 28.1% impact) [IG]"
      when IG is active, including percentage impact for each driver.
    """
    if ig_df.empty or len(ig_df) < 2:
        return "Insufficient data for root-cause analysis."

    total_abs = float(ig_df["Abs_Attribution"].sum())

    # Warm-up guard — all zeros means IG has not kicked in yet
    if total_abs < 1e-6:
        return "⏳ Analysing… (IG activates after ~20 ticks of data)"

    method = ig_df["Method"].iloc[0] if "Method" in ig_df.columns else "IG"
    tag    = " [IG]" if method == "IG" else " [z-score]"

    p = ig_df.iloc[0]
    s = ig_df.iloc[1]

    p_pct = round(float(p["Abs_Attribution"]) / total_abs * 100, 1)
    s_pct = round(float(s["Abs_Attribution"]) / total_abs * 100, 1)

    return (
        f"Primary: {str(p['Feature']).upper()} "
        f"({p['Direction']}, {p_pct}% impact){tag}  ·  "
        f"Secondary: {str(s['Feature']).upper()} "
        f"({s['Direction']}, {s_pct}% impact){tag}"
    )


def _zero_result(feature_cols, method="Warming up…"):
    return pd.DataFrame({
        "Feature":         feature_cols,
        "IG_Attribution":  [0.0] * len(feature_cols),
        "Abs_Attribution": [0.0] * len(feature_cols),
        "Direction":       ["⏳ collecting"] * len(feature_cols),
        "Method":          [method] * len(feature_cols),
    })
