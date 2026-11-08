import shutil
import tempfile
from pathlib import Path
from unittest import mock

from django.test import TestCase, override_settings

from accounts.services import register_organization
from ai.gemini_client import GeminiUnavailable
from dashboard.tests import make_pdf
from knowledge import pdf_processing
from knowledge.models import Chunk, Document
from knowledge.retrieval import retrieve, search

PASSWORD = 'S3cure-pass-123'


def fake_embed(texts, **kwargs):
    return [[1.0, 0.0, 0.0]] * len(texts)


class ChunkingTests(TestCase):
    def test_short_page_is_one_chunk(self):
        self.assertEqual(pdf_processing.chunk_text(pdf_processing.normalize('  hello   world ')), ['hello world'])

    def test_long_page_windows_respect_size_and_overlap(self):
        text = ' '.join(f'word{i}' for i in range(600))
        chunks = pdf_processing.chunk_text(text)
        self.assertGreater(len(chunks), 1)
        for prev, cur in zip(chunks, chunks[1:]):
            self.assertLessEqual(len(prev), pdf_processing.CHUNK_SIZE)
            self.assertIn(cur.split()[0], prev.split())  # overlap carries words over
        self.assertTrue(chunks[-1].endswith('word599'))

    def test_chunks_never_cross_pages(self):
        pages = [
            (1, ' '.join(f'alpha{i}' for i in range(400))),
            (2, 'beta short page'),
            (3, ' '.join(f'gamma{i}' for i in range(300))),
        ]
        prefix = {1: 'alpha', 2: 'beta', 3: 'gamma'}
        chunks = pdf_processing.chunk_pages(pages)
        self.assertEqual({n for n, _ in chunks}, {1, 2, 3})
        page_text = dict(pages)
        for number, text in chunks:
            self.assertIn(text, page_text[number])
            others = set(prefix.values()) - {prefix[number]}
            self.assertFalse(any(o in text for o in others), f'page {number} chunk leaked another page')


class ProcessDocumentTests(TestCase):
    def setUp(self):
        self.storage = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.storage, ignore_errors=True)
        override = override_settings(PRIVATE_STORAGE_DIR=self.storage)
        override.enable()
        self.addCleanup(override.disable)
        self.org = register_organization('Org A', 'a@a.test', PASSWORD).orgmembership.organization
        (self.storage / 'doc.pdf').write_bytes(make_pdf(pages=3))
        self.doc = Document.objects.create(
            organization=self.org, original_name='doc.pdf', stored_path='doc.pdf', size_bytes=1, page_count=3
        )

    @mock.patch('ai.gemini_client.embed_texts', side_effect=fake_embed)
    def test_chunks_stored_per_page_and_ready(self, embed):
        pdf_processing.process_document(self.doc)
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.status, Document.Status.READY)
        chunks = list(Chunk.objects.filter(organization=self.org, document=self.doc).order_by('chunk_index'))
        self.assertEqual([c.page_number for c in chunks], [1, 2, 3])
        self.assertEqual([c.text for c in chunks], [f'Policy text page {i}' for i in (1, 2, 3)])
        self.assertEqual(embed.call_args.kwargs['task_type'], 'RETRIEVAL_DOCUMENT')

    @mock.patch('ai.gemini_client.embed_texts', side_effect=GeminiUnavailable('down'))
    def test_embedding_outage_marks_failed_without_chunks(self, embed):
        pdf_processing.process_document(self.doc)
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.status, Document.Status.FAILED)
        self.assertEqual(self.doc.error_message, pdf_processing.MSG_EMBED_UNAVAILABLE)
        self.assertFalse(Chunk.objects.filter(organization=self.org).exists())


@override_settings(RETRIEVAL_MIN_SIMILARITY=0.55, VECTOR_BACKEND='numpy')
class RetrievalTests(TestCase):
    def setUp(self):
        self.org_a = register_organization('Org A', 'a@a.test', PASSWORD).orgmembership.organization
        self.org_b = register_organization('Org B', 'b@b.test', PASSWORD).orgmembership.organization
        self.a_chunks = self.add_doc(self.org_a, 'Scholarship_Rules.pdf', [
            ('Merit scholarship requires GPA of at least 3.5', [1.0, 0.0, 0.0]),
            ('Scholarship deadlines are in March', [0.9, 0.1, 0.0]),
        ])
        self.b_chunks = self.add_doc(self.org_b, 'Leave_Policy.pdf', [
            ('Annual leave is 20 days after probation', [0.0, 1.0, 0.0]),
            ('Sick leave requires a doctor note', [0.0, 0.8, 0.2]),
        ])

    def add_doc(self, org, name, items):
        doc = Document.objects.create(organization=org, original_name=name, stored_path='x.pdf',
                                      size_bytes=1, page_count=1, status=Document.Status.READY)
        return [
            Chunk.objects.create(organization=org, document=doc, page_number=1, chunk_index=i,
                                 text=text, embedding=vec)
            for i, (text, vec) in enumerate(items)
        ]

    @mock.patch('ai.gemini_client.embed_query', return_value=[1.0, 0.0, 0.0])
    def test_org_b_never_gets_org_a_chunks(self, embed):
        # Query vector and keywords match Org A's chunks exactly; Org B must still only see its own.
        hits = search(self.org_b, 'merit scholarship GPA', k=10).hits
        a_ids = {str(c.pk) for c in self.a_chunks}
        self.assertTrue(hits)
        self.assertFalse(a_ids & {h.chunk_id for h in hits})
        self.assertEqual({h.doc_name for h in hits}, {'Leave_Policy.pdf'})
        self.assertEqual(retrieve(self.org_b, 'merit scholarship GPA'), [])  # gated: nothing relevant in B

    @mock.patch('ai.gemini_client.embed_query', return_value=[1.0, 0.0, 0.0])
    def test_relevant_query_returns_ranked_hits(self, embed):
        hits = retrieve(self.org_a, 'merit scholarship GPA')
        self.assertEqual(hits[0].chunk_id, str(self.a_chunks[0].pk))
        self.assertEqual((hits[0].doc_name, hits[0].page), ('Scholarship_Rules.pdf', 1))
        embed.assert_called_once_with('merit scholarship GPA')

    @mock.patch('ai.gemini_client.embed_query', return_value=[0.0, 0.0, 1.0])
    def test_gate_returns_empty_below_threshold(self, embed):
        result = search(self.org_a, 'scholarship parking permit')
        self.assertTrue(result.hits)  # BM25 still matches "scholarship"
        self.assertLess(result.best_similarity, 0.55)
        self.assertEqual(retrieve(self.org_a, 'scholarship parking permit'), [])

    @mock.patch('ai.gemini_client.embed_query')
    def test_empty_query_skips_embedding(self, embed):
        self.assertEqual(retrieve(self.org_a, '   '), [])
        embed.assert_not_called()

    @mock.patch('ai.gemini_client.embed_query', return_value=[1.0, 0.0, 0.0])
    def test_chunks_of_unready_documents_are_ignored(self, embed):
        Document.objects.filter(organization=self.org_a).update(status=Document.Status.PROCESSING)
        self.assertEqual(search(self.org_a, 'scholarship').hits, [])
