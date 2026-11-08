import hashlib
import secrets

from django.db import models


def hash_token(raw_token):
    return hashlib.sha256(raw_token.encode()).hexdigest()


class AssistantToken(models.Model):
    organization = models.OneToOneField('accounts.Organization', on_delete=models.CASCADE, related_name='assistant_token')
    token_hash = models.CharField(max_length=64, unique=True)
    token_prefix = models.CharField(max_length=8)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    @classmethod
    def issue(cls, organization):
        """Create or replace the org's token; the raw value is returned once and never stored."""
        raw_token = secrets.token_urlsafe(32)
        token, _ = cls.objects.update_or_create(
            organization=organization,
            defaults={'token_hash': hash_token(raw_token), 'token_prefix': raw_token[:8], 'is_active': True},
        )
        return token, raw_token

    @classmethod
    def lookup(cls, raw_token):
        """Active token for a raw value (compared by hash), or None."""
        if not raw_token or len(raw_token) > 128:
            return None
        return cls.objects.select_related('organization').filter(
            token_hash=hash_token(raw_token), is_active=True
        ).first()

    def __str__(self):
        return f'{self.token_prefix}… ({self.organization})'


class QueryLog(models.Model):
    class Outcome(models.TextChoices):
        ANSWERED = 'answered', 'Answered'
        NO_ANSWER = 'no_answer', 'No answer'
        NEEDS_REVIEW = 'needs_review', 'Needs review'
        ERROR = 'error', 'Error'

    organization = models.ForeignKey(
        'accounts.Organization', on_delete=models.CASCADE, related_name='query_logs', db_index=False
    )
    question = models.TextField()
    question_type = models.CharField(max_length=32, blank=True)
    outcome = models.CharField(max_length=12, choices=Outcome.choices)
    decisions = models.JSONField(default=dict, blank=True)
    references = models.JSONField(default=list, blank=True)
    latency_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['organization', 'created_at'], name='querylog_org_created_idx')]

    def __str__(self):
        return f'{self.outcome}: {self.question[:40]}'
