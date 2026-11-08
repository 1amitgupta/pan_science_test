import logging
import uuid
from pathlib import Path

from django.conf import settings
from pypdf import PdfReader

from .models import Chunk, Document
from .pdf_processing import process_document

logger = logging.getLogger(__name__)

MAX_DOCUMENTS = 10
MAX_PAGES = 20
MAX_BYTES = 10 * 1024 * 1024

MSG_NOT_PDF = 'Only PDF files are accepted.'
MSG_TOO_LARGE = 'File is too large: the maximum size is 10 MB.'
MSG_TOO_MANY_DOCS = 'Document limit reached: an organization can have at most 10 PDFs. Delete one first.'
MSG_CORRUPT = 'Invalid or corrupted PDF.'
MSG_TOO_MANY_PAGES = 'PDF has {pages} pages: the maximum is 20.'
MSG_NO_TEXT = 'PDF contains no extractable text (scanned?).'


class UploadError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def validate_pdf(organization, uploaded):
    """Run the S3 checks in order; return the page count or raise UploadError."""
    if not uploaded or not uploaded.name.lower().endswith('.pdf'):
        raise UploadError(MSG_NOT_PDF)
    uploaded.seek(0)
    if uploaded.read(4) != b'%PDF':
        raise UploadError(MSG_NOT_PDF)
    if uploaded.size > MAX_BYTES:
        raise UploadError(MSG_TOO_LARGE, status=413)
    if Document.objects.filter(organization=organization).count() >= MAX_DOCUMENTS:
        raise UploadError(MSG_TOO_MANY_DOCS, status=409)
    try:
        uploaded.seek(0)
        reader = PdfReader(uploaded)
        if reader.is_encrypted and not reader.decrypt(''):
            raise ValueError('encrypted')
        pages = list(reader.pages)
    except Exception:
        raise UploadError(MSG_CORRUPT)
    if not pages:
        raise UploadError(MSG_CORRUPT)
    if len(pages) > MAX_PAGES:
        raise UploadError(MSG_TOO_MANY_PAGES.format(pages=len(pages)))
    try:
        has_text = any((page.extract_text() or '').strip() for page in pages)
    except Exception:
        raise UploadError(MSG_CORRUPT)
    if not has_text:
        raise UploadError(MSG_NO_TEXT)
    return len(pages)


def absolute_path(stored_path):
    return Path(settings.PRIVATE_STORAGE_DIR) / stored_path


def store_upload(organization, uploaded):
    """Validate, save to private storage, create the Document, then process it."""
    page_count = validate_pdf(organization, uploaded)
    stored_path = f'{organization.pk}/{uuid.uuid4().hex}.pdf'
    target = absolute_path(stored_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    uploaded.seek(0)
    with open(target, 'wb') as fh:
        for chunk in uploaded.chunks():
            fh.write(chunk)
    try:
        doc = Document.objects.create(
            organization=organization,
            original_name=Path(uploaded.name).name[:255],
            stored_path=stored_path,
            size_bytes=uploaded.size,
            page_count=page_count,
        )
    except Exception:
        target.unlink(missing_ok=True)
        raise
    try:
        process_document(doc)
    except Exception:
        logger.exception('Processing failed for document %s', doc.pk)
        doc.status = Document.Status.FAILED
        doc.error_message = 'Processing failed unexpectedly. Delete and re-upload to retry.'
        doc.save(update_fields=['status', 'error_message'])
    return doc


def delete_document(doc):
    Chunk.objects.filter(organization=doc.organization_id, document=doc).delete()
    path = absolute_path(doc.stored_path)
    doc.delete()
    path.unlink(missing_ok=True)
