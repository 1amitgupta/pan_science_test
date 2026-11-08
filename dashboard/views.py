from bson import ObjectId
from django.contrib.auth.decorators import login_required
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.template.defaultfilters import filesizeformat
from django.utils import timezone
from django.views.decorators.http import require_POST

from assistant.models import AssistantToken, QueryLog
from assistant.views import NEW_TOKEN_SESSION_KEY, embed_payload
from knowledge.models import Document
from knowledge.services import MAX_DOCUMENTS, UploadError, delete_document, store_upload


def current_org(request):
    return request.user.orgmembership.organization


def document_json(doc):
    return {
        'id': str(doc.pk),
        'name': doc.original_name,
        'pages': doc.page_count,
        'size': filesizeformat(doc.size_bytes),
        'status': doc.status,
        'status_label': doc.get_status_display(),
        'error': doc.error_message,
        'uploaded': timezone.localtime(doc.uploaded_at).strftime('%Y-%m-%d %H:%M'),
    }


@login_required
def home(request):
    docs = Document.objects.filter(organization=current_org(request)).order_by('-uploaded_at')
    return render(request, 'dashboard/knowledge.html', {'documents': docs, 'max_documents': MAX_DOCUMENTS})


@login_required
@require_POST
def upload_document(request):
    org = current_org(request)
    try:
        doc = store_upload(org, request.FILES.get('file'))
    except UploadError as exc:
        return JsonResponse({'error': exc.message}, status=exc.status)
    count = Document.objects.filter(organization=org).count()
    return JsonResponse({'document': document_json(doc), 'count': count}, status=201)


@login_required
@require_POST
def delete_document_view(request, pk):
    if not ObjectId.is_valid(pk):
        raise Http404
    org = current_org(request)
    doc = get_object_or_404(Document, pk=ObjectId(pk), organization=org)
    delete_document(doc)
    return JsonResponse({'deleted': pk, 'count': Document.objects.filter(organization=org).count()})


@login_required
def assistant(request):
    org = current_org(request)
    token = AssistantToken.objects.filter(organization=org).first()
    raw = request.session.pop(NEW_TOKEN_SESSION_KEY, None)  # shown once, then only the hash remains
    if raw and (token is None or AssistantToken.lookup(raw) != token):
        raw = None
    context = {'token': token, 'embed': embed_payload(request, raw) if raw else None}
    return render(request, 'dashboard/assistant.html', context)


@login_required
@require_POST
def regenerate_token(request):
    _, raw = AssistantToken.issue(current_org(request))
    request.session.pop(NEW_TOKEN_SESSION_KEY, None)
    return JsonResponse(embed_payload(request, raw))



USAGE_COLORS = {'answered': 'success', 'no_answer': 'secondary', 'needs_review': 'warning', 'error': 'danger'}


@login_required
def usage(request):
    logs = QueryLog.objects.filter(organization=current_org(request))
    stats = [('Total questions', logs.count(), 'dark')] + [
        (label, logs.filter(outcome=value).count(), USAGE_COLORS[value]) for value, label in QueryLog.Outcome.choices
    ]
    recent = list(logs.order_by('-created_at')[:20])
    for log in recent:
        log.color = USAGE_COLORS.get(log.outcome, 'secondary')
    return render(request, 'dashboard/usage.html', {'stats': stats, 'recent': recent})
