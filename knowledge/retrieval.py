"""Hybrid retrieval (cosine + BM25, fused with RRF) with a grounding gate (SPEC S4)."""
import re
from dataclasses import dataclass, field

import numpy as np
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from rank_bm25 import BM25Okapi

from ai import gemini_client

from .models import Chunk, Document

RRF_K = 60
_TOKEN = re.compile(r'\w+')


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_name: str
    page: int
    text: str
    score: float  # fused RRF score
    similarity: float = 0.0  # raw cosine similarity
    document_id: str = ''


@dataclass
class SearchResult:
    hits: list = field(default_factory=list)
    best_similarity: float = 0.0

    @property
    def is_grounded(self):
        return bool(self.hits) and self.best_similarity >= settings.RETRIEVAL_MIN_SIMILARITY


def tokenize(text):
    return _TOKEN.findall(text.lower())


def _cosine(matrix, query):
    norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
    norms[norms == 0] = 1.0
    return (matrix @ query) / norms


def _rrf(rankings, n):
    fused = np.zeros(n)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] += 1.0 / (RRF_K + rank)
    return fused


def search(org, query, k=6):
    """Ungated hybrid search over one org's chunks; also reports raw best cosine similarity."""
    if settings.VECTOR_BACKEND != 'numpy':
        raise ImproperlyConfigured('Only VECTOR_BACKEND=numpy is implemented.')
    query = (query or '').strip()
    if not query:
        return SearchResult()
    doc_names = dict(
        Document.objects.filter(organization=org, status=Document.Status.READY).values_list('id', 'original_name')
    )
    chunks = [
        c for c in Chunk.objects.filter(organization=org).only('id', 'text', 'embedding', 'page_number', 'document')
        if c.document_id in doc_names and c.embedding
    ]
    if not chunks:
        return SearchResult()

    query_vec = np.asarray(gemini_client.embed_query(query), dtype=float)
    matrix = np.asarray([c.embedding for c in chunks], dtype=float)
    if matrix.shape[1] != query_vec.shape[0]:
        raise gemini_client.GeminiBadResponse('query embedding dimension does not match stored chunks')
    sims = _cosine(matrix, query_vec)
    rankings = [list(np.argsort(-sims, kind='stable'))]

    query_tokens = tokenize(query)
    if query_tokens:
        bm25 = BM25Okapi([tokenize(c.text) or ['_'] for c in chunks])
        scores = bm25.get_scores(query_tokens)
        rankings.append([i for i in np.argsort(-scores, kind='stable') if scores[i] > 0])

    fused = _rrf(rankings, len(chunks))
    top = np.argsort(-fused, kind='stable')[:k]
    hits = [
        Hit(chunk_id=str(chunks[i].pk), doc_name=doc_names[chunks[i].document_id], page=chunks[i].page_number,
            text=chunks[i].text, score=float(fused[i]), similarity=float(sims[i]),
            document_id=str(chunks[i].document_id))
        for i in top
    ]
    return SearchResult(hits=hits, best_similarity=float(sims.max()))


def retrieve(org, query, k=6):
    """Gated retrieval: [] when best cosine similarity is below RETRIEVAL_MIN_SIMILARITY."""
    result = search(org, query, k)
    return result.hits if result.is_grounded else []
