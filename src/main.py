import numpy as np
from .behavior.feature_builder import build_behavior_vector
from .text_features.feature_builder import build_implicit_vector
from .projection.heuristic_projection import heuristic_projection as _heuristic_projection
from .projection.consistency import compute_consistency
from .fusion.risk_model import compute_final_risk, classify_risk

def heuristic_projection(implicit_vector: np.ndarray) -> np.ndarray:
    """
    Wrapper for the modular heuristic projection.
    """
    return _heuristic_projection(implicit_vector)

def run_pipeline(module_averages: dict, free_text: str, latencies: list, affect_features: list) -> dict:
    """
    Mental State Analysis Pipeline.
    Ties together behavior, text, and affect modalities to produce a clinical risk assessment.
    
    Inputs:
        module_averages: dict of 6 Likert scores (1-5)
        free_text: raw text input from the user
        latencies: list of response times in seconds
        affect_features: fused vector of face and voice features
    """
    
    # 1. Behavioral Features: Build vector [mean_latency, std_latency, duration, count]
    session_data = {
        "response_times": latencies,
        "session_start": 0,
        "session_end": sum(latencies) if latencies else 0
    }
    behavior_vector = build_behavior_vector(session_data)
    
    # 2. Implicit (Text) Features: Sentiment, self-reference, and distortions
    implicit_vector = build_implicit_vector(free_text)
    
    # 3. Projection & Consistency Analysis
    # Maps text features to the same space as questionnaire scores for comparison
    projected_vector = _heuristic_projection(implicit_vector)
    
    # Explicit subvector for comparison [sleep, mood, anxiety]
    explicit_subvector = np.array([
        module_averages.get("sleep", 3.0),
        module_averages.get("mood", 3.0),
        module_averages.get("anxiety", 3.0)
    ])
    
    # Compute consistency (distance inverted and normalized)
    dist = compute_consistency(explicit_subvector, projected_vector)
    # Max distance in [1, 5]^3 space is ~6.93
    norm_dist = np.clip(dist / 7.0, 0.0, 1.0)
    
    # 4. Final Fusion and Risk Calculation
    risk_score = compute_final_risk(
        explicit_scores=module_averages,
        consistency_score=norm_dist,
        behavior_vector=behavior_vector,
        affect_vector=np.array(affect_features) if affect_features else None
    )
    
    risk_level = classify_risk(risk_score)
    
    return {
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "consistency": 1.0 - float(norm_dist)
    }
