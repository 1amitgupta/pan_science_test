"""Single gateway for every Gemini call: timeout, 2 retries with backoff, typed errors."""
import logging
import time
from functools import lru_cache

from django.conf import settings
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def _transport_errors():
    # google-genai uses httpx2 (a drop-in httpx fork) and plain httpx when installed.
    found = []
    for name in ('httpx', 'httpx2'):
        try:
            found.append(__import__(name).TransportError)  # covers timeouts + connection errors
        except ImportError:
            pass
    return tuple(found)


TRANSPORT_ERRORS = _transport_errors()

MAX_RETRIES = 2
BACKOFF_SECONDS = 1.0
EMBED_BATCH_SIZE = 100  # batchEmbedContents accepts at most 100 inputs
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
UNAVAILABLE_CODES = RETRYABLE_CODES | {401, 403}


class GeminiError(Exception):
    """Base class; `str(exc)` is safe to log, not to show to end users."""


class GeminiUnavailable(GeminiError):
    """Timeout, quota, network, auth or 5xx: the service can't be used right now."""


class GeminiBadResponse(GeminiError):
    """The request was rejected or the response was empty/malformed/invalid."""


@lru_cache(maxsize=4)
def _build_client(api_key, timeout_ms):
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms, retry_options=types.HttpRetryOptions(attempts=1)),
    )


def _client():
    if not settings.GEMINI_API_KEY:
        raise GeminiUnavailable('GEMINI_API_KEY is not set')
    return _build_client(settings.GEMINI_API_KEY, int(settings.GEMINI_TIMEOUT_SECONDS * 1000))


def _call(label, fn):
    """Run fn() with retries; translate SDK/transport errors into typed errors."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except errors.APIError as exc:
            code = getattr(exc, 'code', None)
            if code not in RETRYABLE_CODES or attempt == MAX_RETRIES:
                logger.warning('Gemini %s failed (HTTP %s): %s', label, code, exc)
                cls = GeminiUnavailable if code in UNAVAILABLE_CODES else GeminiBadResponse
                raise cls(f'Gemini {label} failed with HTTP {code}') from exc
        except TRANSPORT_ERRORS as exc:
            if attempt == MAX_RETRIES:
                logger.warning('Gemini %s transport error: %r', label, exc)
                raise GeminiUnavailable(f'Gemini {label} timed out or could not connect') from exc
        time.sleep(BACKOFF_SECONDS * (2 ** attempt))


def embed_texts(texts, task_type='RETRIEVAL_DOCUMENT'):
    """Embed texts in batches; returns one float list per input, in order."""
    texts = list(texts)
    client = _client()
    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=settings.GEMINI_EMBED_DIM)
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        resp = _call('embed', lambda: client.models.embed_content(
            model=settings.GEMINI_EMBED_MODEL, contents=batch, config=config))
        embeddings = getattr(resp, 'embeddings', None) or []
        if len(embeddings) != len(batch) or any(not e.values for e in embeddings):
            raise GeminiBadResponse(f'expected {len(batch)} embeddings, got {len(embeddings)}')
        vectors.extend([float(v) for v in e.values] for e in embeddings)
    return vectors


def embed_query(text):
    return embed_texts([text], task_type='RETRIEVAL_QUERY')[0]


def generate(prompt, *, model=None, system=None, schema=None, temperature=0.2):
    """Return text, or a validated `schema` (pydantic model class) instance for JSON output."""
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    if schema is not None:
        config.response_mime_type = 'application/json'
        config.response_schema = schema
    client = _client()
    resp = _call('generate', lambda: client.models.generate_content(
        model=model or settings.GEMINI_LLM_MODEL, contents=prompt, config=config))
    text = getattr(resp, 'text', None)
    if not text:
        raise GeminiBadResponse('empty response')
    if schema is None:
        return text
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError('schema must be a pydantic BaseModel subclass')
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        logger.warning('Gemini JSON failed validation against %s: %s', schema.__name__, exc)
        raise GeminiBadResponse(f'response did not match {schema.__name__}') from exc
