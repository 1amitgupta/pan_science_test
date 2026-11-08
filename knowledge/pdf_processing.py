from .models import Document


def process_document(doc):
    """Phase 2 stub: marks the document ready. Real extract/chunk/embed lands in Phase 3."""
    doc.status = Document.Status.READY
    doc.error_message = ''
    doc.save(update_fields=['status', 'error_message'])
