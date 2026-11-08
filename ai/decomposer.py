"""Decision decomposition (LLM, structured output): question + retrieved rules → decisions."""
import logging

from . import gemini_client
from .schemas import Decomposition

logger = logging.getLogger(__name__)

MAX_DECISIONS = 6

SYSTEM = (
    'You turn an eligibility/decision question into structured, checkable decisions. '
    'Use ONLY conditions stated in the provided context; never invent rules. '
    'Every decision must cite the source_document and source_page of the chunk it comes from. '
    'user_facts are facts the user stated about themselves (value as a string; null if not given). '
    'boolean decisions compare one user fact (by its exact name) to a threshold using operator; '
    'for in/not_in give threshold as a comma-separated list. Use fact/operator/threshold only for '
    'boolean (null otherwise), options only for choice ([] otherwise). '
    'List in missing_facts any fact a rule needs that the user did not state. '
    'combine.required_ids lists the boolean decisions that must all hold.'
)


def label(hit):
    return f'[{hit.doc_name} p.{hit.page}]'


def format_context(hits):
    return '\n\n'.join(f'{label(h)}\n{h.text}' for h in hits)


def find_hit(hits, document, page):
    """The retrieved chunk a decision cites (doc name match is case/extension-insensitive)."""
    def norm(name):
        name = (name or '').strip().lower()
        return name[:-4] if name.endswith('.pdf') else name
    for h in hits:
        if norm(h.doc_name) == norm(document) and h.page == page:
            return h
    return None


def decompose(question, hits):
    prompt = f'Context:\n{format_context(hits)}\n\nQuestion: {question}'
    result = gemini_client.generate(prompt, system=SYSTEM, schema=Decomposition, temperature=0.0)
    return sanitize(result, hits)


def sanitize(result, hits):
    """Keep only decisions grounded in a retrieved chunk, with unique ids and usable fields."""
    kept, seen = [], set()
    for d in result.decisions:
        if d.id in seen or find_hit(hits, d.source_document, d.source_page) is None:
            logger.info('dropping ungrounded/duplicate decision %r', d.id)
            continue
        if d.kind == 'boolean' and not (d.fact and d.operator):
            continue
        if d.kind == 'choice' and len(set(d.options)) < 2:
            continue
        seen.add(d.id)
        kept.append(d)
    kept = kept[:MAX_DECISIONS]
    ids = {d.id for d in kept}
    required = [i for i in result.combine.required_ids if i in ids]
    if not required:
        required = [d.id for d in kept if d.kind == 'boolean']
    return result.model_copy(update={
        'decisions': kept,
        'combine': result.combine.model_copy(update={'required_ids': required}),
    })
