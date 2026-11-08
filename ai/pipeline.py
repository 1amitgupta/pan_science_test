"""Runs the answer pipeline for one validated question; the view handles input validation."""
import logging
import time
import uuid

from assistant.models import QueryLog
from knowledge import retrieval

from . import answerer, classifier, decisions, decomposer, gemini_client
from .combiner import combine
from .schemas import CombinedResult

logger = logging.getLogger(__name__)

NO_ANSWER_MESSAGE = "I couldn't find anything about that in this organization's documents."


class PipelineUnavailable(Exception):
    """Gemini could not be used; carries the request_id for the friendly error response."""

    def __init__(self, request_id):
        super().__init__(request_id)
        self.request_id = request_id


def _log(org, question, qtype, outcome, started, decisions_json=None, refs=None):
    QueryLog.objects.create(
        organization=org, question=question, question_type=qtype or '', outcome=outcome,
        decisions=decisions_json or {}, references=refs or [],
        latency_ms=int((time.monotonic() - started) * 1000),
    )


def _decide(question, hits):
    """Steps 5–7: decompose, run decisions in parallel, combine. Bad LLM JSON → needs_review."""
    try:
        plan = decomposer.decompose(question, hits)
    except gemini_client.GeminiBadResponse as exc:
        logger.warning('decomposition failed: %s', exc)
        combined = CombinedResult(verdict='needs_review', outcome='needs_review',
                                  summary=['The conditions could not be evaluated automatically.'])
        return [], combined
    results = decisions.run_decisions(plan, hits)
    if not results:
        combined = CombinedResult(verdict='needs_review', outcome='needs_review',
                                  summary=['No checkable conditions were found in the documents.'],
                                  missing_facts=plan.missing_facts)
        return [], combined
    combined = combine(results, plan.combine.rule, plan.combine.required_ids, plan.missing_facts)
    return results, combined


def ask(org, question):
    """Answer one question for `org`; returns the API payload. Raises PipelineUnavailable."""
    started = time.monotonic()
    request_id = uuid.uuid4().hex[:12]
    qtype = None
    try:
        cls = classifier.classify(question)
        qtype = cls.label
        hits = retrieval.retrieve(org, question)
        if not hits:
            _log(org, question, qtype, QueryLog.Outcome.NO_ANSWER, started)
            return {'answer': NO_ANSWER_MESSAGE, 'outcome': 'no_answer', 'question_type': qtype,
                    'decisions': [], 'references': [], 'request_id': request_id}

        results, combined = ([], None)
        if qtype == 'decision':
            results, combined = _decide(question, hits)
        text, refs = answerer.answer(question, hits, combined)
    except gemini_client.GeminiError as exc:
        logger.error('ask %s failed (type=%s): %r', request_id, qtype, exc)
        _log(org, question, qtype, QueryLog.Outcome.ERROR, started)
        raise PipelineUnavailable(request_id) from exc

    outcome = combined.outcome if combined else 'answered'
    decision_rows = [r.model_dump() for r in results]
    log_json = {'results': decision_rows, 'classifier': {'p': round(cls.probability, 3), 'source': cls.source}}
    if combined:
        log_json.update(verdict=combined.verdict, missing_facts=combined.missing_facts)
    _log(org, question, qtype, outcome, started, log_json, refs)
    return {
        'answer': text, 'outcome': outcome, 'question_type': qtype,
        'verdict': combined.verdict if combined else None,
        'missing_facts': combined.missing_facts if combined else [],
        'decisions': decision_rows, 'references': refs, 'request_id': request_id,
    }
