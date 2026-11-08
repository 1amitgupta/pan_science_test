import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from accounts.models import Organization
from ai import pipeline


def refs_text(refs):
    return '; '.join(f"{r['document']} p.{r['page']}" for r in refs) or '-'


class Command(BaseCommand):
    help = 'Run demo/test_questions.json through the pipeline and print a compact result table.'

    def add_arguments(self, parser):
        parser.add_argument('--file', default=str(Path(settings.BASE_DIR) / 'demo' / 'test_questions.json'))

    def handle(self, file, **options):
        questions = json.loads(Path(file).read_text())
        orgs = {o.slug: o for o in Organization.objects.filter(slug__in={q['org'] for q in questions})}
        missing = {q['org'] for q in questions} - set(orgs)
        if missing:
            raise CommandError(f'Unknown org(s) {sorted(missing)}; run `python manage.py seed_demo` first.')

        rows = [('id', 'org', 'expected', 'actual', 'type', 'refs')]
        passed = 0
        for q in questions:
            try:
                result = pipeline.ask(orgs[q['org']], q['question'])
                actual, qtype, refs = result['outcome'], result['question_type'], refs_text(result['references'])
            except pipeline.PipelineUnavailable as exc:
                actual, qtype, refs = 'error', '-', f'request {exc.request_id}'
            ok = actual == q['expected']
            passed += ok
            mark = '' if ok else ' ✗'
            rows.append((q['id'], q['org'], q['expected'], actual + mark, f"{qtype} (exp {q['type']})", refs))

        widths = [max(len(str(r[i])) for r in rows) for i in range(5)]
        for row in rows:
            cells = [str(c).ljust(w) for c, w in zip(row[:5], widths)] + [row[5]]
            self.stdout.write(' | '.join(cells))
        self.stdout.write(f'\n{passed}/{len(questions)} outcomes match expected.')
