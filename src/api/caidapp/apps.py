from django.apps import AppConfig


class CaidappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "caidapp"

    def ready(self):
        import caidapp.signals
