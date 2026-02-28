import numpy as np
from src.text_features.feature_builder import build_implicit_vector
from src.projection.heuristic_projection import heuristic_projection
from src.projection.consistency import compute_consistency


def run_session():

    explicit_vector = np.array([4, 4, 2])  # sleep, mood, stress

    text = input("Enter user text: ")

    implicit_vector = build_implicit_vector(text)

    projected = heuristic_projection(implicit_vector)

    consistency = compute_consistency(explicit_vector, projected)

    print("\nExplicit:", explicit_vector)
    print("Projected:", projected)
    print("Consistency:", consistency)


if __name__ == "__main__":
    run_session()