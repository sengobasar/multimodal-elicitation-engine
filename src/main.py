# ==========================================================
# Main Multimodal Risk Pipeline
# ==========================================================

def run_pipeline(explicit: list | dict, text: str, latencies: list, affect=None):
    """
    explicit: questionnaire scores (list of raw scores or dict of module averages)
    text: user free text
    latencies: response times
    affect: fusion vector from face + voice pipeline
    """
    import numpy as np
    from .text_features.feature_builder import build_implicit_vector
    from .projection.heuristic_projection import heuristic_projection
    from .projection.consistency import compute_consistency
    from .behavior.feature_builder import build_behavior_vector
    from .fusion.risk_model import compute_final_risk, classify_risk

    try:

        # -------------------------------------------------
        # Explicit Vector (Questionnaire)
        # -------------------------------------------------

        if isinstance(explicit, list):
            explicit_vector = np.array(explicit, dtype=float)
            explicit_input = explicit_vector
        else:
            # Module-based dictionary
            explicit_input = explicit
            # For backward compatibility with other internal components that might need a list
            explicit_vector = np.array(list(explicit.values()), dtype=float)

        # -------------------------------------------------
        # Implicit Text Features
        # -------------------------------------------------

        if text is None:
            text = ""

        implicit_vector = build_implicit_vector(text)

        projected = heuristic_projection(implicit_vector)

        try:
            consistency = compute_consistency(
                explicit_vector,
                projected
            )
        except Exception as ce:
            print(f"Consistency computation failed: {ce}")
            consistency = 0.5 

        # -------------------------------------------------
        # Behavioral Features (Response Timing)
        # -------------------------------------------------

        session_data = {
            "response_times": latencies if latencies else [],
            "session_start": 0,
            "session_end": sum(latencies) if latencies else 0,
        }

        behavior_vector = build_behavior_vector(session_data)

        # -------------------------------------------------
        # Affect Features (Face + Voice)
        # -------------------------------------------------

        if affect is None or len(affect) == 0:
            affect_vector = np.zeros(16)
        else:
            affect_vector = np.array(affect, dtype=float)

        # -------------------------------------------------
        # Multimodal Risk Fusion
        # -------------------------------------------------

        risk_final = compute_final_risk(
            explicit_input,
            consistency,
            behavior_vector,
            affect_vector
        )

        risk_level = classify_risk(risk_final)

        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        return {
            "explicit": explicit_input if isinstance(explicit_input, dict) else explicit_vector.tolist(),
            "projected": projected.tolist() if 'projected' in locals() else [],
            "consistency": float(consistency),
            "behavior_vector": behavior_vector.tolist(),
            "affect_vector": affect_vector.tolist(),
            "risk_score": float(risk_final),
            "risk_level": risk_level
        }

    except Exception as e:

        print("Pipeline error:", e)

        # Safe fallback output
        return {
            "explicit": explicit,
            "projected": [],
            "consistency": 0.0,
            "behavior_vector": [],
            "affect_vector": [],
            "risk_score": 0.0,
        }