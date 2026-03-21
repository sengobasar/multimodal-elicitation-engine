# src/behavior/feature_builder.py

import numpy as np
from .latency import compute_latency_stats

def build_behavior_vector(session_data: dict) -> np.ndarray:
    """
    session_data must contain:
    - response_times: list[float]
    - session_start: float
    - session_end: float
    """

    response_times = session_data.get("response_times", [])
    session_start = session_data.get("session_start", 0)
    session_end = session_data.get("session_end", 0)

    mean_latency, std_latency = compute_latency_stats(response_times)

    session_duration = session_end - session_start
    question_count = len(response_times)

    return np.array([
        mean_latency,
        std_latency,
        session_duration,
        question_count
    ])