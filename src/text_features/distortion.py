import re

DISTORTION_WORDS = ["always", "never", "nothing", "completely"]

def get_distortion_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    count = sum(word in DISTORTION_WORDS for word in words)
    return count / len(words)