from types import SimpleNamespace
from unittest import mock

import httpx2 as httpx
from django.test import SimpleTestCase, override_settings
from google.genai import errors
from pydantic import BaseModel

from ai import gemini_client as gc


class Answer(BaseModel):
    answer: str


def embed_response(n):
    return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2]) for _ in range(n)])


@override_settings(GEMINI_API_KEY='test-key')
@mock.patch('ai.gemini_client.time.sleep')
class GeminiClientTests(SimpleTestCase):
    def patch_client(self):
        client = mock.MagicMock()
        patcher = mock.patch('ai.gemini_client._client', return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)
        return client.models

    def test_embed_batches_and_preserves_order(self, sleep):
        models = self.patch_client()
        models.embed_content.side_effect = lambda model, contents, config: embed_response(len(contents))
        vectors = gc.embed_texts([f't{i}' for i in range(gc.EMBED_BATCH_SIZE + 5)])
        self.assertEqual(len(vectors), gc.EMBED_BATCH_SIZE + 5)
        self.assertEqual(models.embed_content.call_count, 2)

    def test_retries_then_succeeds(self, sleep):
        models = self.patch_client()
        models.embed_content.side_effect = [httpx.ReadTimeout('slow'), errors.ServerError(503, {}), embed_response(1)]
        self.assertEqual(gc.embed_query('q'), [0.1, 0.2])
        self.assertEqual(sleep.call_count, 2)

    def test_timeout_after_retries_is_unavailable(self, sleep):
        models = self.patch_client()
        models.embed_content.side_effect = httpx.ReadTimeout('slow')
        with self.assertRaises(gc.GeminiUnavailable):
            gc.embed_query('q')
        self.assertEqual(models.embed_content.call_count, gc.MAX_RETRIES + 1)

    def test_bad_request_is_not_retried(self, sleep):
        models = self.patch_client()
        models.generate_content.side_effect = errors.ClientError(400, {})
        with self.assertRaises(gc.GeminiBadResponse):
            gc.generate('hi')
        self.assertEqual(models.generate_content.call_count, 1)

    def test_invalid_json_is_bad_response(self, sleep):
        models = self.patch_client()
        models.generate_content.return_value = SimpleNamespace(text='{"wrong": 1}')
        with self.assertRaises(gc.GeminiBadResponse):
            gc.generate('hi', schema=Answer)

    def test_valid_json_is_parsed(self, sleep):
        models = self.patch_client()
        models.generate_content.return_value = SimpleNamespace(text='{"answer": "yes"}')
        self.assertEqual(gc.generate('hi', schema=Answer).answer, 'yes')

    @override_settings(GEMINI_API_KEY='')
    def test_missing_key_is_unavailable(self, sleep):
        with self.assertRaises(gc.GeminiUnavailable):
            gc.embed_query('q')


# ---- Phase 5: System 1 decisions, combiner, classifier ----
from ai import classifier, decisions as dec  # noqa: E402
from ai.combiner import combine  # noqa: E402
from ai.schemas import Decision, DecisionResult, Decomposition, CombineSpec, UserFact  # noqa: E402
from knowledge.retrieval import Hit  # noqa: E402


def rule(op, threshold, fact='gpa', id='d1'):
    return Decision(id=id, kind='boolean', question='?', fact=fact, operator=op, threshold=threshold,
                    options=[], source_page=1, source_document='Rules.pdf')


def facts(*items):
    return {dec._norm(f.name): f for f in (UserFact(name=n, value=v, type=t) for n, v, t in items)}


def res(id, value, kind='boolean', error=None):
    return DecisionResult(id=id, kind=kind, value=value, error=error)


class RuleEvaluatorTests(SimpleTestCase):
    def check(self, op, threshold, fact_value, expected, kind='number'):
        out = dec.evaluate_rule(rule(op, threshold), facts(('GPA', fact_value, kind)))
        self.assertEqual(out.value, expected, f'{fact_value} {op} {threshold}')

    def test_numeric_operators(self):
        self.check('>=', '3.5', '3.6', True)
        self.check('>=', '3.5', '3.5', True)
        self.check('>=', '3.5', '3.4', False)
        self.check('<=', '5', '7 days', False)
        self.check('<=', '5', '5', True)
        self.check('==', '2', '2.0', True)
        self.check('!=', '2', '3', True)

    def test_bool_and_string_operators(self):
        self.check('==', 'false', 'no', True, kind='bool')
        self.check('!=', 'true', 'yes', False, kind='bool')
        self.check('in', 'Full-time, Part-time', 'part-time', True, kind='string')
        self.check('not_in', 'full, partial', 'full', False, kind='string')
        self.check('not_in', 'full, partial', 'none', True, kind='string')

    def test_missing_or_unreadable_fact_is_unknown(self):
        self.assertEqual(dec.evaluate_rule(rule('>=', '3.5'), {}).value, 'unknown')
        self.check('>=', '3.5', None, 'unknown')
        self.check('>=', '3.5', 'excellent', 'unknown')
        self.check('>=', 'n/a', '3.6', 'unknown')
        self.check('>=', 'x', 'abc', 'unknown', kind='string')  # ordering needs numbers

    def test_fact_name_matching_is_normalized(self):
        out = dec.evaluate_rule(rule('>=', '2', fact='year_of_study'), facts(('Year of study', '2', 'number')))
        self.assertIs(out.value, True)
        self.assertEqual(out.confidence, 1.0)


class CombinerTests(SimpleTestCase):
    def test_all_required_truth_table(self):
        cases = [
            ((True, True), 'eligible', 'answered'),
            ((True, False), 'not_eligible', 'answered'),
            ((False, False), 'not_eligible', 'answered'),
            ((True, 'unknown'), 'needs_review', 'needs_review'),
            ((False, 'unknown'), 'not_eligible', 'answered'),  # a failed requirement is decisive
        ]
        for values, verdict, outcome in cases:
            results = [res(f'd{i}', v) for i, v in enumerate(values)]
            out = combine(results, 'all_required', [r.id for r in results], ['income'])
            self.assertEqual((out.verdict, out.outcome), (verdict, outcome), values)
            self.assertEqual(out.missing_facts, ['income'])

    def test_any_and_weighted(self):
        self.assertEqual(combine([res('a', False), res('b', True)], 'any').verdict, 'eligible')
        self.assertEqual(combine([res('a', False), res('b', 'unknown')], 'any').verdict, 'needs_review')
        self.assertEqual(combine([res('a', False), res('b', False)], 'any').verdict, 'not_eligible')
        self.assertEqual(combine([res('a', True), res('b', False)], 'weighted').verdict, 'eligible')
        weighted = [res('a', False), res('b', False), res('s', 0.9, kind='score')]
        self.assertEqual(combine(weighted, 'weighted').verdict, 'not_eligible')

    def test_error_forces_review_and_non_required_are_ignored(self):
        out = combine([res('a', True), res('c', None, kind='choice', error='evaluation failed')], 'all_required', ['a'])
        self.assertEqual(out.outcome, 'needs_review')
        self.assertIn('c: could not be evaluated', out.summary)
        out = combine([res('a', True), res('b', False)], 'all_required', ['a'])
        self.assertEqual(out.verdict, 'eligible')

    def test_no_booleans_gives_no_verdict(self):
        out = combine([res('s', 0.7, kind='score'), res('c', 'merit', kind='choice')])
        self.assertEqual((out.verdict, out.outcome), (None, 'answered'))
        self.assertEqual(out.summary, ['s (support score): 0.7', 'c (choice): merit'])


class RunDecisionsTests(SimpleTestCase):
    hits = [Hit('c1', 'Rules.pdf', 1, 'GPA at least 3.5. Choose merit or need.', 0.1, 0.8)]

    def plan(self):
        return Decomposition(
            user_facts=[UserFact(name='gpa', value='3.7', type='number')],
            decisions=[
                rule('>=', '3.5'),
                Decision(id='d2', kind='choice', question='Which?', fact=None, operator=None, threshold=None,
                         options=['merit', 'need'], source_page=1, source_document='Rules.pdf'),
                Decision(id='d3', kind='score', question='Supported?', fact=None, operator=None, threshold=None,
                         options=[], source_page=1, source_document='Rules.pdf'),
            ],
            combine=CombineSpec(rule='all_required', required_ids=['d1']), missing_facts=[])

    def test_parallel_results_and_one_error_is_isolated(self):
        def fake(prompt, *, schema=None, **kw):
            if schema.__name__ == 'ChoiceOut':
                raise gc.GeminiBadResponse('bad json')
            return schema(rating=4)
        with mock.patch.object(gc, 'generate', side_effect=fake):
            out = {r.id: r for r in dec.run_decisions(self.plan(), self.hits)}
        self.assertIs(out['d1'].value, True)
        self.assertEqual(out['d1'].evidence_ref, '[Rules.pdf p.1]')
        self.assertEqual(out['d2'].error, 'evaluation failed')
        self.assertAlmostEqual(out['d3'].value, 0.9)  # 0.5*0.8 + 0.5*4/4
        self.assertEqual(combine(list(out.values()), 'all_required', ['d1']).outcome, 'needs_review')


class ClassifierTests(SimpleTestCase):
    def test_confident_system1_does_not_call_llm(self):
        with mock.patch.object(gc, 'generate') as gen:
            out = classifier.classify('What is the difference between sick leave and annual leave?')
        self.assertEqual((out.label, out.source), ('comparison', 'system1'))
        gen.assert_not_called()

    def test_low_confidence_falls_back_to_llm_then_system1_on_error(self):
        with mock.patch.object(classifier, 'predict', return_value=('direct', 0.3)):
            with mock.patch.object(gc, 'generate', return_value=classifier.QuestionTypeOut(label='policy')):
                self.assertEqual(classifier.classify('hmm').source, 'llm')
            with mock.patch.object(gc, 'generate', side_effect=gc.GeminiUnavailable('down')):
                out = classifier.classify('hmm')
        self.assertEqual((out.label, out.source), ('direct', 'system1_low_confidence'))
