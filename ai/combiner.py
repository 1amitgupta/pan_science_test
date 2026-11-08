"""Combine decision results into a verdict — pure Python, no I/O."""
from .schemas import CombinedResult

UNKNOWN = 'unknown'
WEIGHTED_PASS = 0.5


def _describe(r):
    if r.error:
        return f'{r.id}: could not be evaluated'
    if r.kind == 'score':
        return f'{r.id} (support score): {r.value}'
    if r.kind == 'choice':
        return f'{r.id} (choice): {r.value}'
    return f'{r.id}: {"met" if r.value is True else "not met" if r.value is False else "unknown"}'


def combine(results, rule='all_required', required_ids=(), missing_facts=()):
    """Verdict from DecisionResults. Errors/unknowns → needs_review; no booleans → no verdict."""
    summary = [_describe(r) for r in results]
    missing = list(dict.fromkeys(missing_facts))
    booleans = [r for r in results if r.kind == 'boolean']
    required = [r for r in booleans if r.id in set(required_ids)] or booleans

    def review():
        return CombinedResult(verdict='needs_review', outcome='needs_review', summary=summary, missing_facts=missing)

    if any(r.error for r in results):
        return review()
    if not required:
        return CombinedResult(verdict=None, outcome='answered', summary=summary, missing_facts=missing)

    values = [r.value for r in required]
    unknown_ids = [r.id for r in required if r.value not in (True, False)]
    if rule == 'any':
        if True in values:
            verdict = 'eligible'
        elif unknown_ids:
            return review()
        else:
            verdict = 'not_eligible'
    elif rule == 'weighted':
        if unknown_ids:
            return review()
        scores = [1.0 if v else 0.0 for v in values]
        scores += [float(r.value) for r in results if r.kind == 'score']
        verdict = 'eligible' if sum(scores) / len(scores) >= WEIGHTED_PASS else 'not_eligible'
    else:  # all_required
        if False in values:
            verdict = 'not_eligible'
        elif unknown_ids:
            return review()
        else:
            verdict = 'eligible'
    return CombinedResult(verdict=verdict, outcome='answered', summary=summary, missing_facts=missing)
