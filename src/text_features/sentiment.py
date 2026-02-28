# src/text_features/sentiment.py

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download once (safe if already downloaded)
nltk.download("vader_lexicon")

# Initialize analyzer once
sia = SentimentIntensityAnalyzer()

def get_sentiment(text: str) -> float:
    """
    Returns compound sentiment score in range [-1, 1]
    -1 = very negative
     0 = neutral
     1 = very positive
    """
    scores = sia.polarity_scores(text)
    return scores["compound"]