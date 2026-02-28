import numpy as np

def heuristic_projection(implicit_vector: np.ndarray) -> np.ndarray:

    # Last 3 elements
    sentiment = implicit_vector[-3]
    sr = implicit_vector[-2]
    cd = implicit_vector[-1]

    # Mood influenced by sentiment and distortion
    mood_est = 3 + (sentiment * 2) - (cd * 2)

    # Sleep slightly affected by negative tone
    sleep_est = 3 - (sentiment * 1.5)

    # Stress increases with negative sentiment + distortion
    stress_est = 3 - (sentiment * 2) + (cd * 3)

    return np.array([sleep_est, mood_est, stress_est])