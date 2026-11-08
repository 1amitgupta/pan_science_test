from django.urls import path

from . import views

app_name = 'assistant'

urlpatterns = [
    path('a/<str:token>/', views.chat_page, name='chat'),
    path('embed.js', views.embed_js, name='embed_js'),
    path('api/ask', views.api_ask, name='api_ask'),
]
