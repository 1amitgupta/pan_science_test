from django.urls import path

from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.home, name='home'),
    path('documents/upload/', views.upload_document, name='upload_document'),
    path('documents/<str:pk>/delete/', views.delete_document_view, name='delete_document'),
    path('assistant/', views.assistant, name='assistant'),
    path('assistant/regenerate/', views.regenerate_token, name='regenerate_token'),
]
