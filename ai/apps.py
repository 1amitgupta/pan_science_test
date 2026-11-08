from django.apps import AppConfig


class AiConfig(AppConfig):
    name = 'ai'

    def ready(self):
        from .classifier import get_model
        get_model()  # train the System 1 classifier once at startup (cached)
