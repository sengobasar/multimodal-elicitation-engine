import numpy as np

def compute_consistency(explicit_vector: np.ndarray,
                        projected_vector: np.ndarray) -> float:
    return np.linalg.norm(explicit_vector - projected_vector)