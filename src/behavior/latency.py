# src/behavior/latency.py

import numpy as np

def compute_latency_stats(response_times):
    """
    response_times: list of latency values (seconds)
    """

    if not response_times:
        return 0.0, 0.0

    mean_latency = np.mean(response_times)
    std_latency = np.std(response_times)

    return mean_latency, std_latency