from bson import ObjectId
from django.contrib.auth.decorators import login_required
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.template.defaultfilters import filesizeformat
from django.utils import timezone
from django.views.decorators.http import require_POST

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
