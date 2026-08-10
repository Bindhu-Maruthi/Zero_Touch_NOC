"""
rl_agent.py
-----------
Q-Learning agent that learns optimal corrective actions from network telemetry.

FIX LOG
-------
1. Q_TABLE_PATH / META_PATH — relative path bug
   Old: Q_TABLE_PATH = "q_table.npy"  (relative to cwd — changes with launch dir)
   New: __file__-based absolute path (same pattern as logger.py / incident_manager.py)
   Effect: Agent no longer "forgets" everything when Streamlit is launched from
           a different working directory.

2. _normalise() is now public (no underscore removed from export)
   app.py imports it for stress_proxy so both use the identical formula.
   No logic change — just ensures consistency between the chart and the agent.
"""

import numpy as np
import os
import pandas as pd

# ── Paths — absolute so they survive any working-directory ────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
Q_TABLE_PATH = os.path.join(_BASE_DIR, "q_table.npy")   # FIX #1
META_PATH    = os.path.join(_BASE_DIR, "q_meta.npy")    # FIX #1

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_STATES        = 50
N_ACTIONS       = 3      # REDUCE, MAINTAIN, RESET
LEARNING_RATE   = 0.10
DISCOUNT_FACTOR = 0.90
EPSILON_START   = 0.30
EPSILON_MIN     = 0.05
EPSILON_DECAY   = 0.995

# Target thresholds for reward shaping — calibrated for real WiFi
LATENCY_TARGET     = 100.0   # ms
PACKET_LOSS_TARGET = 5.0     # %

ACTION_LABELS = {
    0: "🔧 Bandwidth reduced (RL)",
    1: "✅ Network stable (RL)",
    2: "🔄 Network reset triggered (RL)",
}

ACTION_NAMES = {0: "REDUCE", 1: "MAINTAIN", 2: "RESET"}

# Feature ranges for state discretisation
FEATURE_RANGES = {
    "latency":     (0,   500),
    "packet_loss": (0,   30),
    "throughput":  (0,   200),
    "jitter":      (0,   150),
    "bandwidth":   (0,   120),
}

# Weights for composite stress score
FEATURE_WEIGHTS = {
    "latency":     0.30,
    "packet_loss": 0.30,
    "throughput":  0.15,
    "jitter":      0.15,
    "bandwidth":   0.10,
}


# ── State normalisation (public — used by app.py for stress_proxy) ────────────

def normalise(row: pd.Series) -> float:
    """
    Compute a normalised 0–1 stress score from a telemetry row.
    0 = perfectly healthy, 1 = maximally stressed.
    PUBLIC: imported by app.py so the dashboard chart and the agent use
            the identical stress calculation (was inconsistent before).
    """
    score = 0.0
    for feat, (lo, hi) in FEATURE_RANGES.items():
        val  = float(row.get(feat, 0))
        val  = np.clip(val, lo, hi)
        norm = (val - lo) / (hi - lo + 1e-9)
        if feat == "throughput":
            norm = 1.0 - norm   # low throughput = high stress
        score += FEATURE_WEIGHTS[feat] * norm
    return float(np.clip(score, 0.0, 1.0))


# Keep private alias for backwards compat with any internal callers
_normalise = normalise


def discretise(row: pd.Series) -> int:
    """Map a telemetry row to a discrete state index 0 … N_STATES-1."""
    stress = normalise(row)
    bucket = int(stress * (N_STATES - 1))
    return int(np.clip(bucket, 0, N_STATES - 1))


# ── Reward function ───────────────────────────────────────────────────────────

def compute_reward(prev_row: pd.Series, curr_row: pd.Series, action: int) -> float:
    """
    Reward the agent for improving network health.
    +ve → latency / packet_loss moved toward targets.
    −ve → they worsened.
    """
    lat_improvement  = float(prev_row.get("latency",     0)) - float(curr_row.get("latency",     0))
    loss_improvement = float(prev_row.get("packet_loss", 0)) - float(curr_row.get("packet_loss", 0))

    reward = (lat_improvement * 0.3) + (loss_improvement * 2.0)

    if action == 1:
        if (float(curr_row.get("latency",     0)) < LATENCY_TARGET and
                float(curr_row.get("packet_loss", 0)) < PACKET_LOSS_TARGET):
            reward += 1.0

    if action == 2:
        if (float(curr_row.get("latency",     0)) < LATENCY_TARGET and
                float(curr_row.get("packet_loss", 0)) < PACKET_LOSS_TARGET):
            reward -= 3.0

    return float(np.clip(reward, -10.0, 10.0))


# ── Agent class ───────────────────────────────────────────────────────────────

class QLearningAgent:
    """Tabular Q-Learning agent with ε-greedy exploration and persistent Q-table."""

    def __init__(self):
        self.n_states  = N_STATES
        self.n_actions = N_ACTIONS

        if os.path.exists(Q_TABLE_PATH):
            self.q_table = np.load(Q_TABLE_PATH)
        else:
            self.q_table = np.zeros((N_STATES, N_ACTIONS))

        if os.path.exists(META_PATH):
            meta = np.load(META_PATH)
            self.epsilon     = float(meta[0])
            self.total_steps = int(meta[1])
        else:
            self.epsilon     = EPSILON_START
            self.total_steps = 0

    def choose_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state: int, action: int, reward: float, next_state: int) -> None:
        best_next = np.max(self.q_table[next_state])
        td_target = reward + DISCOUNT_FACTOR * best_next
        td_error  = td_target - self.q_table[state][action]
        self.q_table[state][action] += LEARNING_RATE * td_error
        self.epsilon     = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.total_steps += 1

    def save(self) -> None:
        np.save(Q_TABLE_PATH, self.q_table)
        np.save(META_PATH, np.array([self.epsilon, self.total_steps]))

    def get_policy_summary(self) -> pd.DataFrame:
        rows = []
        for s in range(self.n_states):
            stress_mid  = (s + 0.5) / self.n_states
            best_action = int(np.argmax(self.q_table[s]))
            rows.append({
                "State bucket": s,
                "Stress level": round(stress_mid, 2),
                "Greedy action": ACTION_NAMES[best_action],
                "Q(REDUCE)":    round(self.q_table[s][0], 3),
                "Q(MAINTAIN)":  round(self.q_table[s][1], 3),
                "Q(RESET)":     round(self.q_table[s][2], 3),
            })
        return pd.DataFrame(rows)

    @property
    def exploration_rate(self) -> float:
        return round(self.epsilon * 100, 1)

    @property
    def steps_trained(self) -> int:
        return self.total_steps


def get_rl_action_label(action: int) -> str:
    return ACTION_LABELS.get(action, "Unknown action")
