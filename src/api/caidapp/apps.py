from django.apps import AppConfig


class CaidappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "caidapp"

    def ready(self):
        """The code run on application init is here."""
        import caidapp.duplicate_hash_signals  # noqa: F401
        import caidapp.observation_embedding_signals  # noqa: F401
        import caidapp.signals  # noqa: F401

        pass
