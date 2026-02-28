import re

def get_self_reference_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    count = sum(word in ["i", "me", "my"] for word in words)
    return count / len(words)