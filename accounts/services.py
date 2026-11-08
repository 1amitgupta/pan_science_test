from django.contrib.auth import get_user_model
from django.utils.text import slugify

from assistant.models import AssistantToken

from .models import Organization, OrgMembership


def unique_org_slug(name):
    base = slugify(name) or 'org'
    slug, n = base, 1
    while Organization.objects.filter(slug=slug).exists():
        n += 1
        slug = f'{base}-{n}'
    return slug


def register_organization(org_name, email, password):
    """Create user, org, membership and token; undo on failure (standalone Mongo has no transactions).
    The raw assistant token is exposed once as `user.new_assistant_token`."""
    user = get_user_model().objects.create_user(username=email, email=email, password=password)
    org = None
    try:
        org = Organization.objects.create(name=org_name, slug=unique_org_slug(org_name))
        OrgMembership.objects.create(user=user, organization=org)
        _, raw_token = AssistantToken.issue(org)
    except Exception:
        if org is not None:
            org.delete()
        user.delete()
        raise
    user.new_assistant_token = raw_token
    return user
