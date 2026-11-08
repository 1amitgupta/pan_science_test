from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management.base import BaseCommand
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from accounts.models import Organization
from accounts.services import register_organization
from knowledge.demo_content import DEMO_ORGS, DEMO_PASSWORD
from knowledge.models import Document
from knowledge.services import delete_document, store_upload


def build_pdf(path, title, pages):
    """Render one PDF; each item in `pages` becomes its own page."""
    styles = getSampleStyleSheet()
    kinds = {'h': styles['Heading2'], 'p': styles['BodyText'], 'b': styles['BodyText']}
    story = [Paragraph(title, styles['Title'])]
    for n, page in enumerate(pages):
        if n:
            story.append(PageBreak())
        for kind, text in page:
            story.append(Paragraph(f'• {text}' if kind == 'b' else text, kinds[kind]))
            story.append(Spacer(1, 4))
    SimpleDocTemplate(str(path), pagesize=A4, title=title).build(story)


def remove_demo_org(email):
    """Delete a previous demo run's user, org, documents (incl. private files) and logs."""
    user = get_user_model().objects.filter(username=email).first()
    if user is None:
        return
    org = Organization.objects.filter(memberships__user=user).first()
    if org is not None:
        for doc in Document.objects.filter(organization=org):
            delete_document(doc)
        org.delete()
    user.delete()


class Command(BaseCommand):
    help = 'Generate demo PDFs, (re)create 2 demo orgs, upload + process docs, print URLs and logins.'

    def add_arguments(self, parser):
        parser.add_argument('--base-url', default='http://127.0.0.1:8000')
        parser.add_argument('--pdf-dir', default=str(Path(settings.BASE_DIR) / 'demo' / 'sample_pdfs'))

    def handle(self, base_url, pdf_dir, **options):
        pdf_dir = Path(pdf_dir)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        base_url = base_url.rstrip('/')
        summary = []
        for spec in DEMO_ORGS:
            remove_demo_org(spec['email'])
            user = register_organization(spec['name'], spec['email'], DEMO_PASSWORD)
            org = user.orgmembership.organization
            raw = user.new_assistant_token
            self.stdout.write(f'{org.name} ({org.slug})')
            for filename, (title, pages) in spec['docs'].items():
                path = pdf_dir / filename
                build_pdf(path, title, pages)
                upload = SimpleUploadedFile(filename, path.read_bytes(), content_type='application/pdf')
                doc = store_upload(org, upload)
                chunks = doc.chunks.filter(organization=org).count()
                note = f' — {doc.error_message}' if doc.error_message else ''
                self.stdout.write(f'  {filename}: {doc.page_count} pages, {doc.status}, {chunks} chunks{note}')
            summary.append((org, spec['email'], raw))

        self.stdout.write('')
        for org, email, raw in summary:
            self.stdout.write(self.style.SUCCESS(org.name))
            self.stdout.write(f'  dashboard: {base_url}/  login {email} / {DEMO_PASSWORD}')
            self.stdout.write(f'  assistant: {base_url}/chat/{raw}/')
            self.stdout.write(f'  embed demo: http://localhost:8080/external_site.html?t={raw}&host={base_url}')
        self.stdout.write('(embed demo: run `python -m http.server 8080` inside demo/)')
