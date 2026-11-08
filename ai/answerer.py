"""Grounded answer generation and citation parsing."""
import re

from . import gemini_client
from .decomposer import format_context

FALLBACK_REFS = 3
_BRACKET = re.compile(r'\[([^\[\]]+)\]')
_CITE = re.compile(r'^\s*(.+?)\s+p\.\s*(\d+)\s*$')

SYSTEM = (
    'You answer visitor questions for an organization using ONLY the provided context. '
    'If the context is insufficient, say so plainly. Cite every fact with its label exactly as given, '
    'e.g. [Policy.pdf p.3]. If a decision result is provided, explain it using the context; '
    'never change the verdict.\n'
    'Formatting (Markdown only, never raw HTML): start with a 1-2 sentence summary; use "### " '
    'headings for distinct sections, **bold** for key terms, "- " or "1. " lists (indent 2 spaces '
    'to nest). When listing several items that share attributes (products, prices, plans, dates, '
    'comparisons), use a Markdown pipe table with a header row and one item per row; do not put '
    'citations inside table cells; cite the table once on the line directly below it. '
    'Put citations at the end of a sentence or bullet; each citation in its own brackets, never '
    'combined (write [A.pdf p.1] [A.pdf p.2], not [A.pdf p.1, A.pdf p.2]). Keep it concise.'
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


def _citations(text):
    """(name, page) pairs from [doc p.N] and merged [doc p.N, doc p.M] brackets."""
    for group in _BRACKET.findall(text):
        whole = _CITE.match(group)
        parts = [_CITE.match(p) for p in re.split(r'[;,]', group)]
        if all(parts):
            yield from (m.groups() for m in parts)
        elif whole:
            yield whole.groups()


def parse_references(text, hits):
    """Cited [doc p.N] labels that match retrieved chunks, in order; fallback: top hits."""
    known = {(h.doc_name.lower(), h.page): h.doc_name for h in hits}
    refs, seen = [], set()
    for name, page in _citations(text):
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
