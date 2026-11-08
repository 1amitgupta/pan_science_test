"""System 1 decision executors (boolean rules, choice, score), run in parallel."""
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from django.conf import settings

from . import gemini_client
from .decomposer import find_hit, label
from .schemas import DecisionResult, RatingOut, choice_schema

logger = logging.getLogger(__name__)

UNKNOWN = 'unknown'
MAX_WORKERS = 4
SCORE_SIM_WEIGHT = 0.5
_TRUE = {'true', 'yes', 'y', '1'}
_FALSE = {'false', 'no', 'n', '0', 'none'}
_NUMBER = re.compile(r'-?\d+(?:\.\d+)?')


def _norm(name):
    return re.sub(r'[\s_\-]+', ' ', (name or '').strip().lower())


def _coerce(raw, kind):
    """Parse a fact/threshold string; None if it can't be read as `kind`."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw if kind == 'bool' else None
    text = str(raw).strip().lower()
    if not text:
        return None
    if kind == 'number':
        m = _NUMBER.search(text.replace(',', ''))
        return float(m.group()) if m else None
    if kind == 'bool':
        return True if text in _TRUE else False if text in _FALSE else None
    return text


def evaluate_rule(decision, facts):
    """Deterministic boolean rule: facts maps name → UserFact. Missing/unreadable fact → unknown."""
    result = DecisionResult(id=decision.id, kind='boolean', question=decision.question)
    fact = facts.get(_norm(decision.fact))
    op = decision.operator
    kind = fact.type if fact else 'string'
    value = _coerce(fact.value, kind) if fact else None
    if value is None:
        return result.model_copy(update={'value': UNKNOWN})
    if op in ('in', 'not_in'):
        options = {_coerce(x, kind) for x in str(decision.threshold or '').split(',')} - {None}
        if not options:
            return result.model_copy(update={'value': UNKNOWN})
        ok = value in options
        ok = ok if op == 'in' else not ok
    else:
        threshold = _coerce(decision.threshold, kind)
        if threshold is None or (op in ('>=', '<=') and kind != 'number'):
            return result.model_copy(update={'value': UNKNOWN})
        ok = {
            '>=': lambda: value >= threshold, '<=': lambda: value <= threshold,
            '==': lambda: value == threshold, '!=': lambda: value != threshold,
        }[op]()
    return result.model_copy(update={'value': bool(ok), 'confidence': 1.0})


def run_choice(decision, hit):
    out = gemini_client.generate(
        f'Evidence {label(hit)}:\n{hit.text}\n\nQuestion: {decision.question}\n'
        'Pick the option best supported by the evidence.',
        model=settings.GEMINI_FAST_MODEL, schema=choice_schema(decision.options), temperature=0.0,
    )
    return DecisionResult(id=decision.id, kind='choice', question=decision.question,
                          value=out.choice, confidence=round(max(hit.similarity, 0.0), 3))


def run_score(decision, hit):
    out = gemini_client.generate(
        f'Evidence {label(hit)}:\n{hit.text}\n\nStatement/question: {decision.question}\n'
        'Rate how strongly the evidence supports a positive answer: 0 = not at all, 4 = fully.',
        model=settings.GEMINI_FAST_MODEL, schema=RatingOut, temperature=0.0,
    )
    sim = min(max(hit.similarity, 0.0), 1.0)
    score = SCORE_SIM_WEIGHT * sim + (1 - SCORE_SIM_WEIGHT) * out.rating / 4
    return DecisionResult(id=decision.id, kind='score', question=decision.question,
                          value=round(score, 3), confidence=round(score, 3))


def run_one(decision, facts, hits):
    """Execute one decision; never raises (errors become result.error)."""
    hit = find_hit(hits, decision.source_document, decision.source_page)
    ref = label(hit) if hit else None
    try:
        if decision.kind == 'boolean':
            result = evaluate_rule(decision, facts)
        elif hit is None:
            raise ValueError('no evidence chunk')
        elif decision.kind == 'choice':
            result = run_choice(decision, hit)
        else:
            result = run_score(decision, hit)
    except Exception as exc:  # one failed decision must not sink the pipeline
        logger.warning('decision %s (%s) failed: %r', decision.id, decision.kind, exc)
        return DecisionResult(id=decision.id, kind=decision.kind, question=decision.question,
                              value=None, evidence_ref=ref, error='evaluation failed')
    return result.model_copy(update={'evidence_ref': ref})


def run_decisions(decomposition, hits):
    facts = {_norm(f.name): f for f in decomposition.user_facts}
    decisions = decomposition.decisions
    if not decisions:
        return []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(decisions))) as pool:
        return list(pool.map(lambda d: run_one(d, facts, hits), decisions))
