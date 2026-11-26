"""Text preprocessing: cleaning, tokenization, stopword removal, lemmatization."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOP = set(stopwords.words("english"))
_LEMMA = WordNetLemmatizer()
_EMAIL = re.compile(r"\S+@\S+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_WORD = re.compile(r"[^a-z0-9+#.\s]")


def clean_text(text: str) -> str:
    """Lowercase, strip emails/URLs and special characters."""
    text = text.lower()
    text = _EMAIL.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens (keeps c++, c#, node.js style tokens)."""
    return [t.strip(".") for t in text.split() if t.strip(".")]


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize."""
    tokens = tokenize(clean_text(text))
    tokens = [t for t in tokens if t not in _STOP and len(t) > 1]
    return " ".join(_LEMMA.lemmatize(t) for t in tokens)
