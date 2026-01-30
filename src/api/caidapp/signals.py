from caidapp.models import IdentificationModel
from django.db.models.signals import post_migrate, post_save
from django.dispatch import receiver
from . import models


@receiver(post_migrate)
def create_default_models(sender, **kwargs):
    """Create default IdentificationModel entries after migrations."""
    defaults = [
        {
            "name": "LynxV4-MegaDescriptor-v2-T-256",
            "description": "Default description for Model1",
            "public": True,
            "model_path": "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
        },
        {
            "name": "LynxV3-MegaDescriptor-T-224",
            "description": "strakajk/LynxV3-MegaDescriptor-T-224",
            "public": True,
            "model_path": "hf-hub:strakajk/LynxV3-MegaDescriptor-T-224",
        },
    ]

    for default in defaults:
        IdentificationModel.objects.update_or_create(
            name=default["name"],
            defaults=default,
        )


@receiver(post_save, sender=models.CaIDUser)
def create_personal_workgroup(sender, instance, created, **kwargs):
    if not created:
        return

    user = instance.user

    wg = models.WorkGroup.objects.create(
        name=f"{user.username}",
        default_taxon_for_identification=models.get_taxon("Animalia"),
    )

    instance.workgroup = wg
    instance.workgroup_admin = True
    instance.save(update_fields=["workgroup", "workgroup_admin"])