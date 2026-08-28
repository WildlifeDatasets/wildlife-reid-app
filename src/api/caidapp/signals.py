from caidapp.models import IdentificationModel
from django.db import transaction
from django.db.models.signals import post_delete, post_migrate, post_save
from django.dispatch import receiver

from . import models


def _ensure_mediafile_has_observation(mediafile_id, using="default"):
    """Restore the no-detection placeholder if the media file still exists and is empty."""
    observations = models.AnimalObservation.objects.using(using)
    if not models.MediaFile.objects.using(using).filter(pk=mediafile_id).exists():
        return
    if observations.filter(mediafile_id=mediafile_id).exists():
        return
    placeholder, _ = observations.get_or_create(
        mediafile_id=mediafile_id,
        is_no_detection_placeholder=True,
    )
    if observations.filter(mediafile_id=mediafile_id).exclude(pk=placeholder.pk).exists():
        placeholder.delete(using=using)


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
    """Automatically create a personal workgroup for each new user."""
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


@receiver(post_save, sender=models.MediaFile)
def create_no_detection_placeholder(sender, instance, created, raw=False, using="default", **kwargs):
    """Give every newly created media file one observation row immediately."""
    if created and not raw:
        _ensure_mediafile_has_observation(instance.pk, using=using)


@receiver(post_save, sender=models.AnimalObservation)
def remove_superseded_placeholder(sender, instance, raw=False, using="default", **kwargs):
    """Remove a placeholder once a real observation is stored for the same media file."""
    if raw or instance.is_no_detection_placeholder:
        return
    models.AnimalObservation.objects.using(using).filter(
        mediafile_id=instance.mediafile_id,
        is_no_detection_placeholder=True,
    ).exclude(pk=instance.pk).delete()


@receiver(post_delete, sender=models.AnimalObservation)
def restore_placeholder_after_last_observation_delete(sender, instance, using="default", **kwargs):
    """Restore the invariant after deletion, but only once the transaction commits.

    Deferring the check avoids recreating a child while its MediaFile is being
    cascade-deleted.
    """
    mediafile_id = instance.mediafile_id
    transaction.on_commit(
        lambda: _ensure_mediafile_has_observation(mediafile_id, using=using),
        using=using,
    )
