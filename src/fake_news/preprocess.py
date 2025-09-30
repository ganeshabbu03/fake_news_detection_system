import re
from typing import Iterable, List

_whitespace_re = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    text = text.strip()
    text = _whitespace_re.sub(" ", text)
    return text


def preprocess_texts(texts: Iterable[str]) -> List[str]:
    return [normalize_whitespace(t) for t in texts]
