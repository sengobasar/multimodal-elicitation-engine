import numpy as np
from .embedding import get_embedding
from .sentiment import get_sentiment
from .self_reference import get_self_reference_ratio
from .distortion import get_distortion_ratio

def build_implicit_vector(text: str) -> np.ndarray:
    embedding = get_embedding(text)
    sentiment = get_sentiment(text)
    sr = get_self_reference_ratio(text)
    cd = get_distortion_ratio(text)

    return np.concatenate([
        embedding,
        np.array([sentiment, sr, cd])
    ])