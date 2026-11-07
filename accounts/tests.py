import hashlib
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from assistant.models import AssistantToken, hash_token

from .models import Organization, OrgMembership
from .services import register_organization

PASSWORD = 'correct-horse-battery-9'
RAW_TOKEN = 'known-raw-token-value-for-the-test-suite'


class RegistrationTests(TestCase):
    def register(self, org_name='Acme Inc', email='Owner@Acme.test', password=PASSWORD):
        return self.client.post(
            reverse('accounts:register'), {'org_name': org_name, 'email': email, 'password': password}
        )

    def test_registration_creates_user_org_membership_and_token(self):
        response = self.register()
        self.assertRedirects(response, reverse('dashboard:home'))
        user = get_user_model().objects.get(username='owner@acme.test')
        org = Organization.objects.get(slug='acme-inc')
        self.assertEqual(OrgMembership.objects.get(user=user).organization, org)
        self.assertTrue(org.assistant_token.is_active)

    def test_token_stored_only_as_hash(self):
        with mock.patch('assistant.models.secrets.token_urlsafe', return_value=RAW_TOKEN):
            response = self.client.post(
                reverse('accounts:register'),
                {'org_name': 'Acme', 'email': 'a@acme.test', 'password': PASSWORD},
                follow=True,
            )
        token = AssistantToken.objects.get()
        self.assertEqual(token.token_hash, hashlib.sha256(RAW_TOKEN.encode()).hexdigest())
        self.assertEqual(token.token_prefix, RAW_TOKEN[:8])
        stored = [str(getattr(token, f.attname)) for f in token._meta.fields]
        self.assertNotIn(RAW_TOKEN, stored)
        self.assertNotContains(response, RAW_TOKEN)

    def test_issue_replaces_token_and_invalidates_old_hash(self):
        self.register()
        org = Organization.objects.get()
        old_hash = org.assistant_token.token_hash
        token, raw = AssistantToken.issue(org)
        self.assertEqual(AssistantToken.objects.filter(organization=org).count(), 1)
        self.assertEqual(token.token_hash, hash_token(raw))
        self.assertNotEqual(token.token_hash, old_hash)

    def test_duplicate_email_rejected_case_insensitively(self):
        self.register()
        self.client.logout()
        response = self.register(org_name='Other', email='OWNER@acme.test')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'already exists')
        self.assertEqual(Organization.objects.count(), 1)

    def test_same_org_name_gets_unique_slug(self):
        self.register()
        self.client.logout()
        self.register(email='two@acme.test')
        self.assertEqual(
            set(Organization.objects.values_list('slug', flat=True)), {'acme-inc', 'acme-inc-2'}
        )

    def test_failed_registration_leaves_nothing_behind(self):
        with mock.patch('accounts.services.AssistantToken.issue', side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                register_organization('Acme', 'a@acme.test', PASSWORD)
        self.assertFalse(get_user_model().objects.exists())
        self.assertFalse(Organization.objects.exists())
        self.assertFalse(OrgMembership.objects.exists())

    def test_weak_password_rejected(self):
        response = self.register(password='123')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(get_user_model().objects.exists())


class LoginLogoutTests(TestCase):
    def setUp(self):
        self.client.post(
            reverse('accounts:register'),
            {'org_name': 'Acme', 'email': 'a@acme.test', 'password': PASSWORD},
        )
        self.client.logout()

    def test_dashboard_requires_login(self):
        response = self.client.get(reverse('dashboard:home'))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('accounts:login'), response.url)

    def test_login_shows_org_name_and_logout_ends_session(self):
        response = self.client.post(
            reverse('accounts:login'), {'username': 'A@Acme.test', 'password': PASSWORD}, follow=True
        )
        self.assertContains(response, 'Acme')
        self.client.post(reverse('accounts:logout'))
        self.assertEqual(self.client.get(reverse('dashboard:home')).status_code, 302)
