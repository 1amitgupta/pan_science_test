from django.db import models
from django_mongodb_backend.fields import ArrayField


class Document(models.Model):
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        READY = 'ready', 'Ready'
        FAILED = 'failed', 'Failed'

    organization = models.ForeignKey(
        'accounts.Organization', on_delete=models.CASCADE, related_name='documents', db_index=False
    )
    original_name = models.CharField(max_length=255)
    stored_path = models.CharField(max_length=500)
    size_bytes = models.PositiveBigIntegerField()
    page_count = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING)
    error_message = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['organization'], name='document_org_idx')]

    def __str__(self):
        return self.original_name


class Chunk(models.Model):
    # organization duplicates document.organization so retrieval filters on one indexed field.
    organization = models.ForeignKey(
        'accounts.Organization', on_delete=models.CASCADE, related_name='chunks', db_index=False
    )
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks', db_index=False)
    page_number = models.PositiveIntegerField()
    chunk_index = models.PositiveIntegerField()
    text = models.TextField()
    embedding = ArrayField(models.FloatField())
    token_count = models.PositiveIntegerField(default=0)

    class Meta:
        indexes = [
            models.Index(fields=['organization'], name='chunk_org_idx'),
            models.Index(fields=['document'], name='chunk_document_idx'),
        ]
        constraints = [
            models.UniqueConstraint(fields=['document', 'chunk_index'], name='unique_chunk_index_per_document'),
        ]

    def __str__(self):
        return f'{self.document_id}#{self.chunk_index}'
