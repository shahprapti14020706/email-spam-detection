"""Deterministic English preprocessing without downloaded NLTK resources."""
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PREPROCESSING_VERSION = "regex-english-stopwords-porter-v1"
_stemmer = PorterStemmer()

def clean_text(text):
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return " ".join(_stemmer.stem(t) for t in tokens if t not in ENGLISH_STOP_WORDS)
