import numpy as np

def build_behavior_vector(session_data: dict) -> np.ndarray:
    """
    session_data must contain:
    - response_times: list of latencies
    """

    latencies = session_data.get("response_times", [])

    if not latencies:
        return np.array([0.0, 0.0])

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    return np.array([mean_latency, std_latency])