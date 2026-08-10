"""
model.py
--------
Returns a fresh (unfitted) IsolationForest instance.
The model is fitted in app.py against the rolling 200-row window on every cycle.

Tuned for REAL network data (not simulation):
- contamination raised from 0.05 → 0.10: real networks have more natural
  variation so we expect ~10% of readings to look unusual, not 5%.
- n_estimators raised from 100 → 200: more trees = more stable decision
  boundary when the input distribution shifts (as real data does).
- random_state removed: with real data the distribution changes over time,
  a fixed seed gives false confidence in stability.
"""

from sklearn.ensemble import IsolationForest


def load_model() -> IsolationForest:
    return IsolationForest(
        n_estimators=200,
        contamination=0.10,   # expect ~10% anomalous ticks in real traffic
        max_samples="auto",
    )
