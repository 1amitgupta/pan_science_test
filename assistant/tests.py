import json
from unittest import mock

from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.services import register_organization
from ai import gemini_client
from ai.classifier import Classification
from ai.gemini_client import GeminiUnavailable
from ai.schemas import CombineSpec, Decision, Decomposition, UserFact
from knowledge.retrieval import Hit

from . import views
from .models import AssistantToken, QueryLog

HITS = [
    Hit('c1', 'Policy.pdf', 4, 'Refunds within 30 days.', 0.03, 0.8),
    Hit('c2', 'Policy.pdf', 4, 'Keep the receipt.', 0.02, 0.7),
    Hit('c3', 'FAQ.pdf', 1, 'Refunds are paid as cash or store credit.', 0.01, 0.6),
]

PASSWORD = 'correct-horse-battery-9'


class AssistantTestBase(TestCase):
    def setUp(self):
        cache.clear()
        self.user = register_organization('Org A', 'a@a.test', PASSWORD)
        self.org = self.user.orgmembership.organization
        _, self.raw = AssistantToken.issue(self.org)

    def ask(self, question='What is the refund policy?', token=None, raw_body=None):
        body = raw_body if raw_body is not None else json.dumps({'token': token or self.raw, 'question': question})
        return self.client.post(reverse('assistant:api_ask'), body, content_type='application/json')


class ChatPageTests(AssistantTestBase):
    def test_valid_token_renders_without_login(self):
        response = self.client.get(reverse('assistant:chat', args=[self.raw]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Org A')

    def test_invalid_token_is_friendly_404(self):
        response = self.client.get(reverse('assistant:chat', args=['not-a-real-token']))
        self.assertEqual(response.status_code, 404)
        self.assertContains(response, 'invalid or has been replaced', status_code=404)

    def test_regenerated_token_stops_working(self):
        old = self.raw
        _, new = AssistantToken.issue(self.org)
        self.assertEqual(self.client.get(reverse('assistant:chat', args=[old])).status_code, 404)
        self.assertEqual(self.client.get(reverse('assistant:chat', args=[new])).status_code, 200)
        self.assertEqual(self.ask(token=old).status_code, 401)

    def test_inactive_token_is_404(self):
        AssistantToken.objects.filter(organization=self.org).update(is_active=False)
        self.assertEqual(self.client.get(reverse('assistant:chat', args=[self.raw])).status_code, 404)

    def test_x_frame_options_absent_on_chat_page_present_on_dashboard(self):
        chat = self.client.get(reverse('assistant:chat', args=[self.raw]))
        self.assertNotIn('X-Frame-Options', chat.headers)
        self.client.force_login(self.user)
        for name in ('dashboard:home', 'dashboard:assistant'):
            self.assertEqual(self.client.get(reverse(name)).headers.get('X-Frame-Options'), 'DENY')

    def test_embed_js(self):
        response = self.client.get(reverse('assistant:embed_js'), {'t': self.raw})
        self.assertEqual(response.status_code, 200)
        self.assertIn('javascript', response['Content-Type'])
        self.assertIn(f'/a/{self.raw}/', response.content.decode().replace('\\u002D', '-'))  # escapejs
        self.assertEqual(self.client.get(reverse('assistant:embed_js'), {'t': 'bogus'}).status_code, 404)


class TokenScopeTests(AssistantTestBase):
    def test_token_cannot_access_dashboard_urls(self):
        regenerate = reverse('dashboard:regenerate_token')
        for url in (reverse('dashboard:home'), reverse('dashboard:assistant'), regenerate):
            for kwargs in ({'data': {'token': self.raw, 't': self.raw}}, {'HTTP_X_ASSISTANT_TOKEN': self.raw}):
                method = self.client.post if url == regenerate else self.client.get
                response = method(url, **kwargs)
                self.assertEqual(response.status_code, 302, url)
                self.assertIn(reverse('accounts:login'), response['Location'])
        self.assertIsNotNone(AssistantToken.lookup(self.raw))  # regenerate was not triggered

    def test_chat_page_does_not_log_in(self):
        self.client.get(reverse('assistant:chat', args=[self.raw]))
        self.assertEqual(self.client.get(reverse('dashboard:home')).status_code, 302)


class AskApiTests(AssistantTestBase):
    def test_invalid_token_401(self):
        response = self.ask(token='nope')
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()['error'], views.MSG_BAD_TOKEN)

    def test_missing_token_401(self):
        self.assertEqual(self.ask(raw_body=json.dumps({'question': 'hi'})).status_code, 401)

    def test_bad_json_400(self):
        self.assertEqual(self.ask(raw_body='{not json').status_code, 400)
        self.assertEqual(self.ask(raw_body='[1, 2]').status_code, 400)

    def test_empty_and_long_question_400(self):
        self.assertEqual(self.ask(question='   ').json()['error'], views.MSG_EMPTY_QUESTION)
        response = self.ask(question='x' * (views.MAX_QUESTION_CHARS + 1))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['error'], views.MSG_LONG_QUESTION)

    def test_get_not_allowed(self):
        self.assertEqual(self.client.get(reverse('assistant:api_ask')).status_code, 405)

    def test_no_hits_is_no_answer_without_generate(self):
        # Org has no documents → real retrieval returns nothing; Gemini must not be called.
        with mock.patch.object(gemini_client, 'generate') as gen, \
                mock.patch.object(gemini_client, 'embed_query') as emb:
            data = self.ask('What is the capital of France?').json()
        self.assertEqual(data['outcome'], 'no_answer')
        self.assertEqual(data['references'], [])
        self.assertTrue(data['request_id'])
        gen.assert_not_called()
        emb.assert_not_called()
        self.assertEqual(QueryLog.objects.get(organization=self.org).outcome, 'no_answer')

    @mock.patch('knowledge.retrieval.retrieve', return_value=HITS)
    def test_direct_answer_with_cited_references(self, retrieve):
        answer = 'Refunds are possible within 30 days [Policy.pdf p.4]. See also [Unknown.pdf p.9].'
        with mock.patch.object(gemini_client, 'generate', return_value=answer):
            data = self.ask('What is the deadline for scholarship applications?').json()
        self.assertEqual(data['outcome'], 'answered')
        self.assertEqual(data['question_type'], 'direct')
        self.assertEqual(data['references'], [{'document': 'Policy.pdf', 'page': 4}])
        self.assertEqual(data['decisions'], [])
        self.assertEqual(retrieve.call_args.args[0], self.org)  # org comes from the token
        log = QueryLog.objects.get(organization=self.org)
        self.assertEqual((log.outcome, log.question_type), ('answered', 'direct'))

    @mock.patch('knowledge.retrieval.retrieve', return_value=HITS)
    def test_uncited_answer_falls_back_to_top_hits(self, _):
        with mock.patch.object(gemini_client, 'generate', return_value='Within 30 days.'):
            data = self.ask('What is the deadline for scholarship applications?').json()
        self.assertEqual(data['references'], [{'document': 'Policy.pdf', 'page': 4}, {'document': 'FAQ.pdf', 'page': 1}])

    @mock.patch('knowledge.retrieval.retrieve', return_value=HITS)
    def test_decision_with_failing_choice_is_needs_review(self, _):
        plan = Decomposition(
            user_facts=[UserFact(name='days', value='20', type='number')],
            decisions=[
                Decision(id='within_30_days', kind='boolean', question='Within 30 days?', fact='days',
                         operator='<=', threshold='30', options=[], source_page=4, source_document='Policy.pdf'),
                Decision(id='method', kind='choice', question='Refund method?', fact=None, operator=None,
                         threshold=None, options=['cash', 'credit'], source_page=1, source_document='FAQ.pdf'),
                Decision(id='ghost', kind='boolean', question='Ungrounded', fact='days', operator='>=',
                         threshold='1', options=[], source_page=99, source_document='Nope.pdf'),
            ],
            combine=CombineSpec(rule='all_required', required_ids=['within_30_days', 'ghost']),
            missing_facts=[],
        )

        def fake(prompt, *, schema=None, **kw):
            if schema is Decomposition:
                return plan
            if schema is not None:  # the choice decision
                raise GeminiUnavailable('quota')
            self.assertIn('Verdict: needs_review', prompt)
            return 'Needs a human check [Policy.pdf p.4].'

        with mock.patch.object(gemini_client, 'generate', side_effect=fake):
            data = self.ask('I bought it 20 days ago, am I eligible for a refund?').json()
        self.assertEqual(data['question_type'], 'decision')
        self.assertEqual(data['outcome'], 'needs_review')
        rows = {d['id']: d for d in data['decisions']}
        self.assertEqual(set(rows), {'within_30_days', 'method'})  # ungrounded decision dropped
        self.assertIs(rows['within_30_days']['value'], True)
        self.assertEqual(rows['method']['error'], 'evaluation failed')
        self.assertNotIn('quota', json.dumps(data))
        self.assertEqual(QueryLog.objects.get(organization=self.org).outcome, 'needs_review')

    @mock.patch('knowledge.retrieval.retrieve', return_value=HITS)
    def test_decision_all_met_is_eligible(self, _):
        plan = Decomposition(
            user_facts=[UserFact(name='days', value='10', type='number')],
            decisions=[Decision(id='d1', kind='boolean', question='Within 30 days?', fact='days', operator='<=',
                                threshold='30', options=[], source_page=4, source_document='policy')],
            combine=CombineSpec(rule='all_required', required_ids=[]), missing_facts=[],
        )
        with mock.patch.object(gemini_client, 'generate', side_effect=lambda p, schema=None, **kw: plan if schema else 'Yes.'):
            data = self.ask('I bought it 10 days ago, am I eligible for a refund?').json()
        self.assertEqual((data['outcome'], data['verdict']), ('answered', 'eligible'))

    @mock.patch('knowledge.retrieval.retrieve', return_value=HITS)
    def test_gemini_failure_is_friendly_503(self, _):
        with mock.patch.object(gemini_client, 'generate', side_effect=GeminiUnavailable('HTTP 503 secret detail')):
            response = self.ask('What is the deadline for scholarship applications?')
        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertEqual(body['error'], views.MSG_UNAVAILABLE)
        self.assertNotIn('secret', json.dumps(body))
        self.assertTrue(body['request_id'])
        self.assertEqual(QueryLog.objects.get(organization=self.org).outcome, 'error')

    @mock.patch('knowledge.retrieval.retrieve', side_effect=GeminiUnavailable('embed timeout'))
    def test_embedding_failure_is_503(self, _):
        self.assertEqual(self.ask('What is the deadline for scholarship applications?').status_code, 503)

    @override_settings(ASK_RATE_LIMIT_PER_MINUTE=2)
    def test_rate_limit_429(self):
        codes = [self.ask('').status_code for _ in range(3)]
        self.assertEqual(codes, [400, 400, 429])


class AssistantDashboardTests(AssistantTestBase):
    def setUp(self):
        super().setUp()
        self.client.force_login(self.user)

    def test_regenerate_returns_new_url_and_invalidates_old(self):
        response = self.client.post(reverse('dashboard:regenerate_token'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        new = data['url'].rstrip('/').rsplit('/', 1)[-1]
        self.assertNotEqual(new, self.raw)
        self.assertIsNone(AssistantToken.lookup(self.raw))
        self.assertEqual(AssistantToken.lookup(new).organization, self.org)
        self.assertIn(data['url'], data['iframe_snippet'])
        self.assertIn(f'/embed.js?t={new}', data['script_snippet'])

    def test_regenerate_requires_post(self):
        self.assertEqual(self.client.get(reverse('dashboard:regenerate_token')).status_code, 405)

    def test_new_token_shown_once_from_session(self):
        session = self.client.session
        session[views.NEW_TOKEN_SESSION_KEY] = self.raw
        session.save()
        self.assertContains(self.client.get(reverse('dashboard:assistant')), f'/a/{self.raw}/')
        response = self.client.get(reverse('dashboard:assistant'))
        self.assertNotContains(response, self.raw)
        self.assertContains(response, self.raw[:8])

    def test_registration_puts_token_in_session(self):
        self.client.logout()
        self.client.post(reverse('accounts:register'), {
            'org_name': 'Org B', 'email': 'b@b.test', 'password': PASSWORD, 'password_confirm': PASSWORD,
        })
        raw = self.client.session.get(views.NEW_TOKEN_SESSION_KEY)
        self.assertIsNotNone(raw)
        self.assertEqual(AssistantToken.lookup(raw).organization.name, 'Org B')


class ErrorMatrixGapTests(AssistantTestBase):
    def test_rotated_token_is_401_on_api(self):
        old = self.raw
        AssistantToken.issue(self.org)
        self.assertEqual(self.ask(token=old).status_code, 401)

    @mock.patch('knowledge.retrieval.retrieve', return_value=HITS)
    def test_invalid_decomposition_json_is_needs_review_with_answer(self, _):
        def fake(prompt, *, schema=None, **kw):
            if schema is Decomposition:
                raise gemini_client.GeminiBadResponse('invalid JSON')
            return 'A person will need to check this [Policy.pdf p.4].'

        with mock.patch.object(gemini_client, 'generate', side_effect=fake):
            response = self.ask('I bought it 20 days ago, am I eligible for a refund?')
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual((data['outcome'], data['verdict']), ('needs_review', 'needs_review'))
        self.assertIn('check this', data['answer'])
        self.assertEqual(QueryLog.objects.get(organization=self.org).outcome, 'needs_review')


@mock.patch('ai.classifier.classify', return_value=Classification('direct', 0.9, 'system1'))
class CrossTenantAskTests(AssistantTestBase):
    """S8: Org B's token never retrieves Org A's chunks through /api/ask (real retrieval, mocked Gemini)."""

    def setUp(self):
        super().setUp()
        from knowledge.models import Chunk, Document
        doc = Document.objects.create(organization=self.org, original_name='Scholarship_Rules.pdf', stored_path='x.pdf',
                                      size_bytes=1, page_count=1, status=Document.Status.READY)
        Chunk.objects.create(organization=self.org, document=doc, page_number=1, chunk_index=0,
                             text='Merit scholarship requires GPA of at least 3.5', embedding=[1.0, 0.0])
        self.org_b = register_organization('Org B', 'b@b.test', PASSWORD).orgmembership.organization
        _, self.raw_b = AssistantToken.issue(self.org_b)

    @mock.patch('ai.gemini_client.embed_query', return_value=[1.0, 0.0])
    def test_org_b_token_gets_no_answer_for_org_a_content(self, embed, cls):
        with mock.patch.object(gemini_client, 'generate', return_value='LEAK [Scholarship_Rules.pdf p.1]') as gen:
            data = self.ask('What GPA does the merit scholarship require?', token=self.raw_b).json()
        self.assertEqual(data['outcome'], 'no_answer')
        self.assertEqual(data['references'], [])
        gen.assert_not_called()
        self.assertFalse(QueryLog.objects.filter(organization=self.org).exists())
        self.assertEqual(QueryLog.objects.get(organization=self.org_b).outcome, 'no_answer')

    @mock.patch('ai.gemini_client.embed_query', return_value=[1.0, 0.0])
    def test_org_a_token_still_gets_its_own_content(self, embed, cls):
        with mock.patch.object(gemini_client, 'generate', return_value='GPA 3.5 [Scholarship_Rules.pdf p.1]'):
            data = self.ask('What GPA does the merit scholarship require?').json()
        self.assertEqual(data['references'], [{'document': 'Scholarship_Rules.pdf', 'page': 1}])
