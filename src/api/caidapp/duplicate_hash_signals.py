"""Schedule exact hashing from committed source/import changes, independently of ReID."""

from functools import partial

from django.db import transaction
from django.db.models.signals import post_delete, post_save, pre_save
from django.dispatch import receiver

from . import models
from .duplicate_hashes import HASH_FIELDS, invalidate_hashes, schedule_hashes


def _group_for_media(mediafile_id):
    return (
        models.MediaFile.objects.filter(pk=mediafile_id, parent__import_finished=True)
        .values_list(
            "parent__owner__workgroup_id",
            flat=True,
        )
        .first()
    )


def _schedule(groups, using="default"):
    for group_id in set(groups) - {None}:
        transaction.on_commit(partial(schedule_hashes, group_id), using=using, robust=True)


@receiver(pre_save, sender=models.MediaFile)
def remember_hash_source(sender, instance, raw=False, using="default", update_fields=None, **kwargs):
    """Remember the source and preserve hash fields against unrelated saves of stale model instances."""
    if raw:
        return
    old = (
        sender.objects.using(using)
        .filter(pk=instance.pk)
        .values(
            "mediafile",
            "image_file",
            "media_type",
            "parent__owner__workgroup_id",
            *HASH_FIELDS,
        )
        .first()
        if instance.pk
        else None
    )
    instance._duplicate_old_group = old["parent__owner__workgroup_id"] if old else None
    written = {"mediafile", "image_file", "media_type"} if update_fields is None else set(update_fields)
    source = tuple(
        (
            (getattr(instance, field).name or "" if field != "media_type" else instance.media_type)
            if old is None or field in written
            else (old[field] or "")
        )
        for field in ("mediafile", "image_file", "media_type")
    )
    instance._duplicate_source_changed = old is None or source != (
        old["mediafile"] or "",
        old["image_file"] or "",
        old["media_type"],
    )
    instance._duplicate_source_changed |= any(
        field in written and getattr(instance, field) and not getattr(instance, field)._committed
        for field in ("mediafile", "image_file")
    )
    if old:
        for field in HASH_FIELDS:
            setattr(instance, field, old[field])


@receiver(post_save, sender=models.MediaFile)
def schedule_source_hash(sender, instance, raw=False, using="default", **kwargs):
    """Invalidate replaced sources immediately and batch the group's finished media after commit."""
    if raw:
        return
    changed = getattr(instance, "_duplicate_source_changed", False)
    if changed:
        invalidate_hashes(sender.objects.using(using).filter(pk=instance.pk))
        instance.hash_revision += 1
    group = _group_for_media(instance.pk)
    old_group = getattr(instance, "_duplicate_old_group", None)
    if changed or old_group != group:
        if models.UploadedArchive.objects.filter(pk=instance.parent_id, import_finished=True).exists():
            _schedule([group, old_group], using)


@receiver(pre_save, sender=models.UploadedArchive)
def remember_import_completion(sender, instance, raw=False, **kwargs):
    """Avoid postponing hashing for unrelated import status and timestamp saves."""
    if not raw:
        instance._duplicate_import_before = (
            sender.objects.filter(pk=instance.pk)
            .values(
                "import_finished",
                "owner__workgroup_id",
            )
            .first()
            if instance.pk
            else None
        )


@receiver(post_save, sender=models.UploadedArchive)
def schedule_completed_import(sender, instance, raw=False, using="default", **kwargs):
    """A completed import is the common event shared with identification preparation."""
    if raw:
        return
    old = getattr(instance, "_duplicate_import_before", None)
    group_id = instance.owner.workgroup_id
    if instance.import_finished and (not old or not old["import_finished"] or old["owner__workgroup_id"] != group_id):
        _schedule([group_id, old["owner__workgroup_id"] if old else None], using)


@receiver(post_save, sender=models.AnimalObservation)
@receiver(post_delete, sender=models.AnimalObservation)
def schedule_identity_evidence(sender, instance, raw=False, using="default", **kwargs):
    """Annotation changes rebuild identity candidates using existing hashes, never rehash pixels."""
    if not raw:
        _schedule([_group_for_media(instance.mediafile_id)], using)


@receiver(post_delete, sender=models.MediaFile)
def schedule_after_media_delete(sender, instance, using="default", **kwargs):
    """Refresh exact groups and identity evidence after permanent deletion anywhere in the app."""
    group_id = (
        models.UploadedArchive.objects.filter(pk=instance.parent_id)
        .values_list("owner__workgroup_id", flat=True)
        .first()
    )
    _schedule([group_id], using)


@receiver(post_delete, sender=models.IndividualIdentity)
def schedule_after_identity_merge(sender, instance, using="default", **kwargs):
    """Identity merging can bulk-update observations; deleting its source still triggers a refresh."""
    _schedule([instance.owner_workgroup_id], using)


@receiver(pre_save, sender=models.CaIDUser)
def remember_group_membership(sender, instance, raw=False, **kwargs):
    """Remember both groups so moving an owner cannot leave cross-group candidates behind."""
    if not raw:
        instance._duplicate_previous_group = (
            sender.objects.filter(pk=instance.pk).values_list("workgroup_id", flat=True).first()
        )


@receiver(post_save, sender=models.CaIDUser)
def schedule_group_membership(sender, instance, raw=False, using="default", **kwargs):
    """Reuse source hashes while rebuilding both catalogues after owner membership changes."""
    if not raw and getattr(instance, "_duplicate_previous_group", None) != instance.workgroup_id:
        _schedule([getattr(instance, "_duplicate_previous_group", None), instance.workgroup_id], using)
