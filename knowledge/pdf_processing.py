"""Per-page text extraction, page-bounded chunking, and batched embedding (SPEC S3)."""
import logging
import re

from pypdf import PdfReader

from ai import gemini_client

from .models import Chunk, Document

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MSG_NO_TEXT = 'No extractable text found in this PDF.'
MSG_EMBED_UNAVAILABLE = 'The AI service is unavailable right now. Delete and re-upload to retry.'
MSG_EMBED_BAD = 'The AI service returned an invalid response. Delete and re-upload to retry.'

_WS = re.compile(r'\s+')


def normalize(text):
    return _WS.sub(' ', text or '').strip()


def extract_pages(path):
    """Return [(page_number starting at 1, normalized text)] for pages that have text."""
    reader = PdfReader(path)
    if reader.is_encrypted:
        reader.decrypt('')
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = normalize(page.extract_text())
        if text:
            pages.append((number, text))
    return pages


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split one page's text into ~size-char windows with overlap, preferring word boundaries."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(' ', start + size // 2, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        if text[nxt - 1] != ' ':  # don't start mid-word
            space = text.find(' ', nxt, end)
            if space != -1:
                nxt = space + 1
        start = nxt
    return chunks


def chunk_pages(pages):
    """Chunk each page on its own so no chunk ever crosses a page boundary."""
    return [(number, piece) for number, text in pages for piece in chunk_text(text)]


def _fail(doc, message):
    doc.status = Document.Status.FAILED
    doc.error_message = message
    doc.save(update_fields=['status', 'error_message'])


def process_document(doc, path=None):
    """Extract → chunk → embed → store chunks; sets doc status around the work."""
    from .services import absolute_path  # local import: services imports this module

    doc.status = Document.Status.PROCESSING
    doc.error_message = ''
    doc.save(update_fields=['status', 'error_message'])

    pieces = chunk_pages(extract_pages(path or absolute_path(doc.stored_path)))
    if not pieces:
        return _fail(doc, MSG_NO_TEXT)
    try:
        vectors = gemini_client.embed_texts([text for _, text in pieces], task_type='RETRIEVAL_DOCUMENT')
    except gemini_client.GeminiUnavailable:
        return _fail(doc, MSG_EMBED_UNAVAILABLE)
    except gemini_client.GeminiBadResponse:
        return _fail(doc, MSG_EMBED_BAD)

    Chunk.objects.filter(organization=doc.organization_id, document=doc).delete()
    try:
        Chunk.objects.bulk_create([
            Chunk(organization_id=doc.organization_id, document=doc, page_number=number,
                  chunk_index=i, text=text, embedding=vector, token_count=max(1, len(text) // 4))
            for i, ((number, text), vector) in enumerate(zip(pieces, vectors))
        ])
    except Exception:
        Chunk.objects.filter(organization=doc.organization_id, document=doc).delete()
        raise
    doc.status = Document.Status.READY
    doc.save(update_fields=['status'])
    logger.info('Processed document %s into %d chunks', doc.pk, len(pieces))
