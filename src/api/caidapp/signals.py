from django.db.models.signals import post_migrate
from django.dispatch import receiver
from caidapp.models import IdentificationModel


@receiver(post_migrate)
def create_default_models(sender, **kwargs):
    defaults = [
        {"name": "LynxV4-MegaDescriptor-v2-T-256", "description": "Default description for Model1", "public": True,
         "model_path": "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256"},
        {"name": "LynxV3-MegaDescriptor-T-224", "description": "strakajk/LynxV3-MegaDescriptor-T-224", "public": True,
         "model_path": "hf-hub:strakajk/LynxV3-MegaDescriptor-T-224"},
    ]

    for default in defaults:
        IdentificationModel.objects.update_or_create(
            name=default["name"],
            defaults=default,
        )
