"""Grounded answer generation and citation parsing."""
import re

from . import gemini_client
from .decomposer import format_context

FALLBACK_REFS = 3
_CITATION = re.compile(r'\[([^\[\]]+?)\s+p\.\s*(\d+)\]')

SYSTEM = (
    'You answer visitor questions for an organization using ONLY the provided context. '
    'If the context is insufficient, say so plainly. Cite every fact with its label exactly as given, '
    'e.g. [Policy.pdf p.3]. Be concise; plain text, "- " bullets allowed. '
    'If a decision result is provided, explain it using the context; never change the verdict.'
)


def build_prompt(question, hits, combined=None):
    parts = [f'Context:\n{format_context(hits)}']
    if combined is not None:
        lines = [f'Verdict: {combined.verdict or "no overall verdict"}']
        lines += [f'- {s}' for s in combined.summary]
        if combined.missing_facts:
            lines.append('Missing facts (ask the user for these): ' + ', '.join(combined.missing_facts))
        parts.append('Decision result (computed by rules; do not change it):\n' + '\n'.join(lines))
    parts.append(f'Question: {question}')
    return '\n\n'.join(parts)


def parse_references(text, hits):
    """Cited [doc p.N] labels that match retrieved chunks, in order; fallback: top hits."""
    known = {(h.doc_name.lower(), h.page): h.doc_name for h in hits}
    refs, seen = [], set()
    for name, page in _CITATION.findall(text):
        key = (name.strip().lower(), int(page))
        if key in known and key not in seen:
            seen.add(key)
            refs.append({'document': known[key], 'page': key[1]})
    if refs:
        return refs
    for h in hits:
        key = (h.doc_name.lower(), h.page)
        if key not in seen and len(refs) < FALLBACK_REFS:
            seen.add(key)
            refs.append({'document': h.doc_name, 'page': h.page})
    return refs


def answer(question, hits, combined=None):
    text = gemini_client.generate(build_prompt(question, hits, combined), system=SYSTEM).strip()
    return text, parse_references(text, hits)
