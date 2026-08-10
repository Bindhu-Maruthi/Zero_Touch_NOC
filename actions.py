"""
actions.py
----------
Decision and corrective-action layer.

decide_action()        — rule-based severity classifier (CRITICAL / AUTO-HEALING / NORMAL).
                         Still used to gate HITL — the RL agent decides *what* to do,
                         the rule-based classifier decides *when* human oversight is needed.

rl_corrective_action() — asks the Q-Learning agent for the best corrective action
                         and performs one online learning step using the previous row.

corrective_action()    — legacy rule-based fallback (kept for compatibility/testing).

--- REAL DATA RECALIBRATION ---
Original thresholds were tuned for SIMULATED data:
  latency:     normal ~50ms,  anomaly injected at +70-110ms
  packet_loss: normal ~1%,    anomaly injected at +3-6%
  jitter:      normal ~5ms,   anomaly injected at +10-25ms

Real Windows ping data (Hyderabad, India) typical ranges:
  latency:     30 – 80ms   (Google/Cloudflare DNS)
  packet_loss: 0 – 5%      (WiFi fluctuation is common)
  jitter:      20 – 80ms   (WiFi jitter is naturally high)

New thresholds reflect these real-world ranges so CRITICAL only fires
on genuine network problems, not normal WiFi variation.
"""

import pandas as pd


# ── Severity classifier (drives HITL gating) ─────────────────────────────────

def decide_action(row) -> str:
    """
    Returns 'CRITICAL' | 'AUTO-HEALING' | 'NORMAL'.
    Independent of the RL agent — used to gate HITL in app.py.

    Thresholds recalibrated for real Windows/WiFi measurements:

    CRITICAL    — something is genuinely broken
      latency     > 200ms  (was 120ms — real DNS ping rarely exceeds 200ms)
      packet_loss > 15%    (was 5%   — WiFi drops 5% regularly, 15% is a real problem)
      jitter      > 100ms  (NEW      — 50-80ms is normal WiFi, 100ms+ is degraded)

    AUTO-HEALING — degraded but not broken
      latency     > 120ms  (was 80ms)
      packet_loss > 8%     (was 1.5%)
      jitter      > 60ms   (NEW — elevated but not critical)

    NORMAL       — everything within expected real-world range
    """
    latency     = float(row["latency"])
    packet_loss = float(row["packet_loss"])
    jitter      = float(row["jitter"])

    if latency > 200 or packet_loss > 15 or jitter > 100:
        return "CRITICAL"
    elif latency > 120 or packet_loss > 8 or jitter > 60:
        return "AUTO-HEALING"
    else:
        return "NORMAL"


# ── RL-based corrective action ────────────────────────────────────────────────

def rl_corrective_action(
    agent,
    current_row: pd.Series,
    prev_row,
) -> tuple:
    """
    Ask the Q-Learning agent for the best corrective action and do one
    online learning step (so the agent improves every tick).

    Parameters
    ----------
    agent       : QLearningAgent instance (held in st.session_state)
    current_row : latest telemetry Series
    prev_row    : previous telemetry Series, or None on the first tick

    Returns
    -------
    (action_id: int, labels: list[str])
      action_id  0=REDUCE, 1=MAINTAIN, 2=RESET
      labels     human-readable strings for the UI
    """
    from rl_agent import discretise, compute_reward, get_rl_action_label

    current_state = discretise(current_row)
    action        = agent.choose_action(current_state)

    if prev_row is not None:
        prev_state = discretise(prev_row)
        reward     = compute_reward(prev_row, current_row, action)
        agent.learn(prev_state, action, reward, current_state)
        agent.save()

    return action, [get_rl_action_label(action)]


# ── Rule-based fallback ───────────────────────────────────────────────────────

def corrective_action(row) -> list:
    """
    Legacy rule-based corrective actions.
    Used as a safe fallback if the RL agent is unavailable.
    Thresholds updated to match real data ranges.
    """
    actions = []

    if float(row["latency"]) > 200:
        actions.append("🔧 High latency detected — check DNS or switch network")

    if float(row["packet_loss"]) > 15:
        actions.append("🚦 High packet loss — possible WiFi interference")

    if float(row["jitter"]) > 100:
        actions.append("📶 Severe jitter — connection unstable")

    if float(row["latency"]) > 300 and float(row["packet_loss"]) > 10:
        actions.append("🔄 Network reset recommended")

    if not actions:
        actions.append("✅ Network stable")

    return actions
