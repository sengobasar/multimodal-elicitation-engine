import numpy as np

def compute_final_risk(explicit_vector,
                       consistency_score,
                       behavior_vector):

    # Base risk from explicit average
    base_risk = np.mean(explicit_vector)

    # Behavior influence
    behavior_score = np.mean(behavior_vector)

    # Weighted fusion
    risk_final = (
        base_risk
        + 0.2 * consistency_score
        + 0.1 * behavior_score
    )

    return risk_final