# src/text_features/embedding.py

import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np

_tokenizer = None
_model = None

def get_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print("Loading DistilBert model and tokenizer (lazy)...")
        _tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        _model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    return _tokenizer, _model

def get_embedding(text: str) -> np.ndarray:
    tokenizer, model = get_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding