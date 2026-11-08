import json
import time

from django.conf import settings
from django.core.cache import cache

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from django.urls import reverse
from django.views.decorators.cache import never_cache
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from ai import pipeline

from .models import AssistantToken

NEW_TOKEN_SESSION_KEY = 'assistant_new_token'
MAX_QUESTION_CHARS = 1000

MSG_BAD_TOKEN = 'This assistant link is invalid or has been replaced.'
MSG_BAD_REQUEST = 'Invalid request.'
MSG_EMPTY_QUESTION = 'Please enter a question.'
MSG_LONG_QUESTION = f'Questions are limited to {MAX_QUESTION_CHARS} characters.'
MSG_UNAVAILABLE = 'The assistant is temporarily unavailable. Please try again shortly.'
MSG_RATE_LIMITED = 'Too many questions in a short time. Please wait a minute and try again.'


def embed_payload(request, raw_token):
    """Absolute assistant URL plus iframe/script embed snippets for a raw token."""
    chat_url = request.build_absolute_uri(reverse('assistant:chat', args=[raw_token]))
    script_url = request.build_absolute_uri(reverse('assistant:embed_js')) + f'?t={raw_token}'
    return {
        'url': chat_url,
        'iframe_snippet': (
            f'<iframe src="{chat_url}" title="Assistant" width="400" height="600" '
            'style="border:1px solid #ddd;border-radius:8px"></iframe>'
        ),
        'script_snippet': f'<script src="{script_url}" async></script>',
    }


@never_cache
@xframe_options_exempt
@require_GET
def chat_page(request, token):
    record = AssistantToken.lookup(token)
    if record is None:
        return render(request, 'assistant/not_found.html', status=404)
    return render(request, 'assistant/chat.html', {'org': record.organization, 'token': token})


@never_cache
@require_GET
def embed_js(request):
    raw = request.GET.get('t', '')
    if AssistantToken.lookup(raw) is None:
        body = 'console.warn("Knowledge Assistant: invalid or replaced token.");'
        return HttpResponse(body, status=404, content_type='text/javascript; charset=utf-8')
    chat_url = request.build_absolute_uri(reverse('assistant:chat', args=[raw]))
    body = render_to_string('assistant/embed.js', {'chat_url': chat_url})
    return HttpResponse(body, content_type='text/javascript; charset=utf-8')


def _error(message, status):
    return JsonResponse({'error': message}, status=status)


def _rate_limited(request):
    """Fixed one-minute window per client IP, counted in the Django cache."""
    key = f"ask-rate:{request.META.get('REMOTE_ADDR', '')}:{int(time.time() // 60)}"
    if cache.add(key, 1, timeout=60):
        return False
    try:
        return cache.incr(key) > settings.ASK_RATE_LIMIT_PER_MINUTE
    except ValueError:  # expired between add and incr
        cache.add(key, 1, timeout=60)
        return False


@csrf_exempt  # authenticated by the assistant token in the body, never by cookies
@require_POST
def api_ask(request):
    if _rate_limited(request):
        return _error(MSG_RATE_LIMITED, 429)
    try:
        data = json.loads(request.body or b'{}')
    except (ValueError, UnicodeDecodeError):
        return _error(MSG_BAD_REQUEST, 400)
    if not isinstance(data, dict):
        return _error(MSG_BAD_REQUEST, 400)
    token, question = data.get('token'), data.get('question')
    record = AssistantToken.lookup(token) if isinstance(token, str) else None
    if record is None:
        return _error(MSG_BAD_TOKEN, 401)
    question = question.strip() if isinstance(question, str) else ''
    if not question:
        return _error(MSG_EMPTY_QUESTION, 400)
    if len(question) > MAX_QUESTION_CHARS:
        return _error(MSG_LONG_QUESTION, 400)

    try:
        payload = pipeline.ask(record.organization, question)
    except pipeline.PipelineUnavailable as exc:
        return JsonResponse({'error': MSG_UNAVAILABLE, 'request_id': exc.request_id}, status=503)
    return JsonResponse(payload)
