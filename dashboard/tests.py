import io
import shutil
import tempfile
from pathlib import Path
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from accounts.services import register_organization
from knowledge import services
from knowledge.models import Chunk, Document

PASSWORD = 'correct-horse-battery-9'


def make_pdf(pages=1, text=True):
    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({
        NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'),
        NameObject('/BaseFont'): NameObject('/Helvetica'),
    }))
    for i in range(pages):
        page = writer.add_blank_page(612, 792)
        if text:
            stream = DecodedStreamObject()
            stream.set_data(f'BT /F1 12 Tf 72 720 Td (Policy text page {i + 1}) Tj ET'.encode())
            page[NameObject('/Resources')] = DictionaryObject(
                {NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})}
            )
            page[NameObject('/Contents')] = writer._add_object(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class StorageMixin:
    def setUp(self):
        super().setUp()
        self.storage = tempfile.mkdtemp()
        override = override_settings(PRIVATE_STORAGE_DIR=Path(self.storage))
        override.enable()
        self.addCleanup(override.disable)
        self.addCleanup(shutil.rmtree, self.storage, ignore_errors=True)
        self.user_a = register_organization('Org A', 'a@a.test', PASSWORD)
        self.org_a = self.user_a.orgmembership.organization
        self.client.force_login(self.user_a)

    def upload(self, content, name='policy.pdf'):
        return self.client.post(
            reverse('dashboard:upload_document'), {'file': SimpleUploadedFile(name, content, 'application/pdf')}
        )

    def assertUploadError(self, response, message, status=400):
        self.assertEqual(response.status_code, status)
        self.assertEqual(response.json()['error'], message)
        self.assertFalse(Document.objects.filter(organization=self.org_a).exists())


class UploadValidationTests(StorageMixin, TestCase):
    def test_wrong_extension(self):
        self.assertUploadError(self.upload(make_pdf(), name='policy.txt'), services.MSG_NOT_PDF)

    def test_wrong_magic_bytes(self):
        self.assertUploadError(self.upload(b'<html>not a pdf</html>'), services.MSG_NOT_PDF)

    def test_too_large(self):
        content = b'%PDF' + b'0' * services.MAX_BYTES
        self.assertUploadError(self.upload(content), services.MSG_TOO_LARGE, status=413)

    def test_document_limit(self):
        for i in range(services.MAX_DOCUMENTS):
            Document.objects.create(organization=self.org_a, original_name=f'{i}.pdf', stored_path='x', size_bytes=1)
        response = self.upload(make_pdf())
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()['error'], services.MSG_TOO_MANY_DOCS)
        self.assertEqual(Document.objects.filter(organization=self.org_a).count(), services.MAX_DOCUMENTS)

    def test_size_checked_before_document_limit(self):
        for i in range(services.MAX_DOCUMENTS):
            Document.objects.create(organization=self.org_a, original_name=f'{i}.pdf', stored_path='x', size_bytes=1)
        response = self.upload(b'%PDF' + b'0' * services.MAX_BYTES)
        self.assertEqual(response.json()['error'], services.MSG_TOO_LARGE)

    def test_corrupted_pdf(self):
        self.assertUploadError(self.upload(b'%PDF-1.4\ngarbage that is not a pdf'), services.MSG_CORRUPT)

    def test_too_many_pages(self):
        pages = services.MAX_PAGES + 1
        self.assertUploadError(self.upload(make_pdf(pages=pages)), services.MSG_TOO_MANY_PAGES.format(pages=pages))

    def test_no_extractable_text(self):
        self.assertUploadError(self.upload(make_pdf(text=False)), services.MSG_NO_TEXT)

    def test_valid_upload_saved_privately_and_ready(self):
        response = self.upload(make_pdf(pages=services.MAX_PAGES))
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body['count'], 1)
        doc = Document.objects.get(organization=self.org_a)
        self.assertEqual((doc.status, doc.page_count), (Document.Status.READY, services.MAX_PAGES))
        self.assertEqual(body['document']['id'], str(doc.pk))
        self.assertTrue(doc.stored_path.startswith(f'{self.org_a.pk}/'))
        self.assertTrue((Path(self.storage) / doc.stored_path).is_file())
        self.assertContains(self.client.get(reverse('dashboard:home')), '1</span> / 10 documents')

    def test_processing_failure_marks_document_failed(self):
        with mock.patch('knowledge.services.process_document', side_effect=RuntimeError('boom')), \
                self.assertLogs('knowledge.services', 'ERROR'):
            response = self.upload(make_pdf())
        self.assertEqual(response.status_code, 201)
        doc = Document.objects.get(organization=self.org_a)
        self.assertEqual(doc.status, Document.Status.FAILED)
        self.assertNotIn('boom', doc.error_message)

    def test_upload_requires_login(self):
        self.client.logout()
        self.assertEqual(self.upload(make_pdf()).status_code, 302)
        self.assertFalse(Document.objects.exists())


class DeleteDocumentTests(StorageMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.upload(make_pdf())
        self.doc_a = Document.objects.get(organization=self.org_a)
        self.path_a = Path(self.storage) / self.doc_a.stored_path

    def delete(self, pk):
        return self.client.post(reverse('dashboard:delete_document', args=[str(pk)]))

    def test_delete_removes_file_chunks_and_row(self):
        Chunk.objects.create(
            organization=self.org_a, document=self.doc_a, page_number=1, chunk_index=0, text='t', embedding=[0.1]
        )
        response = self.delete(self.doc_a.pk)
        self.assertEqual(response.json()['count'], 0)
        self.assertFalse(Document.objects.filter(pk=self.doc_a.pk).exists())
        self.assertFalse(Chunk.objects.filter(document=self.doc_a.pk).exists())
        self.assertFalse(self.path_a.exists())

    def test_org_a_cannot_delete_org_b_document(self):
        user_b = register_organization('Org B', 'b@b.test', PASSWORD)
        self.client.force_login(user_b)
        self.assertEqual(self.delete(self.doc_a.pk).status_code, 404)
        self.assertTrue(Document.objects.filter(pk=self.doc_a.pk).exists())
        self.assertTrue(self.path_a.exists())

    def test_invalid_object_id_is_404(self):
        self.assertEqual(self.delete('not-an-object-id').status_code, 404)

    def test_delete_requires_post(self):
        response = self.client.get(reverse('dashboard:delete_document', args=[str(self.doc_a.pk)]))
        self.assertEqual(response.status_code, 405)
        self.assertTrue(Document.objects.filter(pk=self.doc_a.pk).exists())
