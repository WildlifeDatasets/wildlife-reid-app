from django.db.models.signals import post_migrate
from django.dispatch import receiver
from caidapp.models import IdentificationModel


@receiver(post_migrate)
def create_default_models(sender, **kwargs):
    defaults = [
        {"name": "Model1", "description": "Default description for Model1", "public": True,
         "model_path": "/path/to/model1"},
        {"name": "Model2", "description": "Default description for Model2", "public": False,
         "model_path": "/path/to/model2"},
    ]

    for default in defaults:
        IdentificationModel.objects.update_or_create(
            name=default["name"],
            defaults=default,
        )
