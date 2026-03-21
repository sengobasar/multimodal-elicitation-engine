# src/fusion/risk_model.py

import numpy as np


def compute_weighted_explicit_score(module_scores: dict) -> float:
    """
    Computes a weighted distress score from 6 core modules.
    Weights based on clinical significance in general distress assessment.
    Likert scores are on a 1-5 scale. We normalize to [0, 1]:
        normalized = (score - 1) / (5 - 1)
    So rating=1 (best) → 0.0, rating=5 (worst) → 1.0.
    """
    weights = {
        "mood": 0.20,
        "anxiety": 0.20,
        "social": 0.20,
        "sleep": 0.15,
        "energy": 0.15,
        "cognitive": 0.10
    }

    total_weighted_score = 0.0
    total_weight_used = 0.0

    for module, weight in weights.items():
        raw_score = module_scores.get(module, 1.0)
        # Normalize 1–5 Likert to [0, 1]
        normalized = (raw_score - 1.0) / 4.0
        normalized = max(0.0, min(1.0, normalized))
        total_weighted_score += normalized * weight
        total_weight_used += weight

    if total_weight_used == 0:
        return 0.0

    return total_weighted_score / total_weight_used


def classify_risk(risk_score: float) -> str:
    """
    Classifies the final risk score into categorical levels.
    """
    if risk_score >= 0.65:
        return "HIGH"
    elif risk_score >= 0.35:
        return "MODERATE"
    else:
        return "LOW"


def _normalize_affect_vector(affect_vector: np.ndarray) -> float:
    """
    Normalizes the affect feature vector to a [0, 1] scalar.

    The affect vector has heterogeneous features with very different scales:
      [0]   valence        (-1 to +1)
      [1]   arousal        (-1 to +1)
      [2]   distress       (0 to 1)
      [3]   energy_rms     (0 to ~50)
      [4]   pitch_hz / 100 (0 to ~5)
      [5]   consistency    (-1 to +1)
      [6-7] jitter/shimmer (0 to 1)
      [8+]  MFCCs          (can be -200 to +200)

    We clip each group to its expected range, rescale to [0, 1], then average.
    """
    if affect_vector.size == 0:
        return 0.0

    normalized_parts = []

    # [0] valence: -1..+1 → invert (low valence = high distress)
    if affect_vector.size > 0:
        valence_distress = (1.0 - affect_vector[0]) / 2.0
        normalized_parts.append(np.clip(valence_distress, 0.0, 1.0))

    # [1] arousal: -1..+1 → absolute value (high arousal = more activation)
    if affect_vector.size > 1:
        arousal_norm = (abs(affect_vector[1]) + 1.0) / 2.0
        normalized_parts.append(np.clip(arousal_norm, 0.0, 1.0))

    # [2] distress: 0..1
    if affect_vector.size > 2:
        normalized_parts.append(np.clip(affect_vector[2], 0.0, 1.0))

    # [3] energy RMS: clip at 50
    if affect_vector.size > 3:
        normalized_parts.append(np.clip(affect_vector[3] / 50.0, 0.0, 1.0))

    # [4] pitch (normalized): already divided by ~100
    if affect_vector.size > 4:
        normalized_parts.append(np.clip(abs(affect_vector[4]) / 5.0, 0.0, 1.0))

    # [5] consistency: treat as 0..1 directly
    if affect_vector.size > 5:
        normalized_parts.append(np.clip(abs(affect_vector[5]), 0.0, 1.0))

    # [6-7] jitter/shimmer: 0..1
    for i in range(6, min(8, affect_vector.size)):
        normalized_parts.append(np.clip(affect_vector[i], 0.0, 1.0))

    # [8+] MFCCs: these live in roughly -200..+200 range; normalize to [0,1]
    for i in range(8, affect_vector.size):
        mfcc_norm = (affect_vector[i] + 200.0) / 400.0
        normalized_parts.append(np.clip(mfcc_norm, 0.0, 1.0))

    return float(np.mean(normalized_parts)) if normalized_parts else 0.0


def compute_final_risk(
    explicit_scores: dict | np.ndarray,
    consistency_score: float,
    behavior_vector: np.ndarray,
    affect_vector: np.ndarray | None,
    alpha1: float = 0.45,   # Questionnaire is primary signal
    alpha2: float = 0.20,   # Consistency (text/answers agreement)
    alpha3: float = 0.15,   # Behavioral (response latency)
    alpha4: float = 0.20    # Affect (face + voice)
) -> float:
    """
    Multimodal deterministic fusion model.

    Modalities
    ----------
    explicit_scores : questionnaire signals (dict of 6 modules or raw vector)
    consistency     : agreement between explicit and implicit text signals
    behavior        : response latency dynamics
    affect          : face + voice emotional features
    """

    # ------------------------------------------------
    # Explicit modality (Weighted + Normalized)
    # ------------------------------------------------
    if isinstance(explicit_scores, dict):
        explicit_val = compute_weighted_explicit_score(explicit_scores)
    else:
        # Raw vector provided: assume already 1-5 scale, normalize each element
        explicit_vector = np.asarray(explicit_scores, dtype=float)
        if explicit_vector.size > 0:
            normalized = (explicit_vector - 1.0) / 4.0
            explicit_val = float(np.mean(np.clip(normalized, 0.0, 1.0)))
        else:
            explicit_val = 0.0

    # ------------------------------------------------
    # Behavioral modality (Scaled)
    # ------------------------------------------------
    behavior_vector = np.asarray(behavior_vector)
    if behavior_vector.size >= 4:
        # mean_latency (clip at 10s), std_latency (clip at 5s),
        # duration (clip at 600s), count (clip at 100)
        scaled_behavior = [
            min(behavior_vector[0] / 10.0, 1.0),
            min(behavior_vector[1] / 5.0, 1.0),
            min(behavior_vector[2] / 600.0, 1.0),
            min(behavior_vector[3] / 100.0, 1.0)
        ]
        behavior_val = float(np.mean(scaled_behavior))
    else:
        behavior_val = float(np.mean(behavior_vector)) if behavior_vector.size > 0 else 0.0

    # ------------------------------------------------
    # Affect modality (face + voice) — properly normalized
    # ------------------------------------------------
    if affect_vector is None:
        affect_val = 0.0
    else:
        affect_val = _normalize_affect_vector(np.asarray(affect_vector, dtype=float))

    # ------------------------------------------------
    # Final fusion (Clamped to [0, 1])
    # ------------------------------------------------
    risk_final = (
        alpha1 * explicit_val
        + alpha2 * float(consistency_score)
        + alpha3 * behavior_val
        + alpha4 * affect_val
    )

    return float(min(max(risk_final, 0.0), 1.0))
