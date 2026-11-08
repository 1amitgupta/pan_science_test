"""System 1 question-type classifier: TF-IDF + LogisticRegression, LLM fallback when unsure."""
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from . import gemini_client
from .schemas import QuestionTypeOut

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent / 'data' / 'question_types.json'
FALLBACK_THRESHOLD = 0.45


@dataclass(frozen=True)
class Classification:
    label: str
    probability: float
    source: str  # 'system1' | 'llm' | 'system1_low_confidence'


@lru_cache(maxsize=1)
def get_model():
    data = json.loads(DATA_PATH.read_text())
    texts = [t for label in data for t in data[label]]
    labels = [label for label in data for _ in data[label]]
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
        LogisticRegression(C=20.0, max_iter=2000),
    )
    model.fit(texts, labels)
    return model


def predict(question):
    """System 1 only: (label, probability)."""
    model = get_model()
    probs = model.predict_proba([question])[0]
    best = int(probs.argmax())
    return str(model.classes_[best]), float(probs[best])


def classify(question):
    label, prob = predict(question)
    if prob >= FALLBACK_THRESHOLD:
        return Classification(label, prob, 'system1')
    logger.info('classifier fallback to LLM (system1=%s p=%.2f)', label, prob)
    try:
        out = gemini_client.generate(
            f'Classify the question into one type.\n'
            f'direct = a single fact; policy = what the rules say; procedural = how to do something; '
            f'decision = whether the asker qualifies/is allowed given their situation; '
            f'comparison = differences between two or more things.\n\nQuestion: {question}',
            model=settings.GEMINI_FAST_MODEL, schema=QuestionTypeOut, temperature=0.0,
        )
        return Classification(out.label, prob, 'llm')
    except gemini_client.GeminiError as exc:
        logger.warning('classifier LLM fallback failed, keeping system1 label: %s', exc)
        return Classification(label, prob, 'system1_low_confidence')
