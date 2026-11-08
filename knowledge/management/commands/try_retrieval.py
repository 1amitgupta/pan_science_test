from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from accounts.models import Organization
from ai.gemini_client import GeminiError
from knowledge.retrieval import search


class Command(BaseCommand):
    help = 'Run hybrid retrieval for one organization and print the top hits.'

    def add_arguments(self, parser):
        parser.add_argument('org_slug')
        parser.add_argument('query')
        parser.add_argument('-k', type=int, default=6)

    def handle(self, org_slug, query, k, **options):
        org = Organization.objects.filter(slug=org_slug).first()
        if org is None:
            raise CommandError(f'No organization with slug "{org_slug}".')
        try:
            result = search(org, query, k)
        except GeminiError as exc:
            raise CommandError(f'Gemini error: {exc}')
        threshold = settings.RETRIEVAL_MIN_SIMILARITY
        gate = 'PASS' if result.is_grounded else 'FAIL (no relevant knowledge)'
        self.stdout.write(f'best_similarity={result.best_similarity:.3f} threshold={threshold:.2f} gate={gate}')
        if not result.hits:
            self.stdout.write('No chunks indexed for this organization.')
        for n, hit in enumerate(result.hits, start=1):
            snippet = hit.text[:160] + ('…' if len(hit.text) > 160 else '')
            self.stdout.write(
                f'{n}. {hit.doc_name} p.{hit.page}  rrf={hit.score:.4f} cos={hit.similarity:.3f}\n   {snippet}'
            )
