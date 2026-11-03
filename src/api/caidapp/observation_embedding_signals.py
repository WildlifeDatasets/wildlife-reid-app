"""Schedule embedding backfills only when their actual inputs change."""

from functools import partial

from django.db import transaction
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver

from . import models
from .observation_embeddings import schedule_embeddings

INPUT_FIELDS = {
    models.AnimalObservation: (
        "mediafile_id",
        "bbox_x_center",
        "bbox_y_center",
        "bbox_width",
        "bbox_height",
        "orientation",
        "is_no_detection_placeholder",
    ),
    models.MediaFile: ("image_file", "mediafile", "media_type", "parent_id"),
    models.WorkGroup: ("identification_model_id",),
    models.IdentificationModel: ("model_path", "base_model_path", "source_identification_model_id"),
    models.UploadedArchive: ("import_finished",),
}


@receiver(pre_save)
def remember_embedding_input(sender, instance, raw=False, update_fields=None, using="default", **kwargs):
    """Remember persisted input fields, ignoring saves that only edit labels or metadata."""
    fields = INPUT_FIELDS.get(sender)
    if raw or fields is None:
        return
    instance._embedding_input_changed = False
    if update_fields is not None and not set(update_fields).intersection(
        {field for field in fields} | {field.removesuffix("_id") for field in fields}
    ):
        return
    instance._embedding_input_changed = True
    instance._embedding_previous_input = (
        sender.objects.using(using).filter(pk=instance.pk).values(*fields).first() if instance.pk else None
    )
    instance._embedding_file_replaced = sender is models.MediaFile and any(
        getattr(instance, field) and not getattr(instance, field)._committed for field in ("image_file", "mediafile")
    )


@receiver(post_save)
def schedule_changed_embedding_input(sender, instance, raw=False, using="default", **kwargs):
    """Register after-commit timers for observation, source, import and encoder changes."""
    if raw or not getattr(instance, "_embedding_input_changed", False):
        return
    instance._embedding_input_changed = False
    current = sender.objects.using(using).filter(pk=instance.pk).values(*INPUT_FIELDS[sender]).first()
    if current == instance._embedding_previous_input and not instance._embedding_file_replaced:
        return
    if sender is models.AnimalObservation:
        if current["is_no_detection_placeholder"]:
            return
        workgroups = models.WorkGroup.objects.filter(caiduser__uploadedarchive__mediafile__observations=instance)
    elif sender is models.MediaFile:
        workgroups = models.WorkGroup.objects.filter(caiduser__uploadedarchive__mediafile=instance)
    elif sender is models.WorkGroup:
        workgroups = models.WorkGroup.objects.filter(pk=instance.pk)
    elif sender is models.IdentificationModel:
        affected = {instance.pk}
        frontier = affected
        while frontier:
            frontier = set(
                models.IdentificationModel.objects.filter(source_identification_model_id__in=frontier)
                .exclude(pk__in=affected)
                .values_list("pk", flat=True)
            )
            affected.update(frontier)
        workgroups = models.WorkGroup.objects.filter(identification_model_id__in=affected)
    else:
        if not current["import_finished"]:
            return
        workgroups = models.WorkGroup.objects.filter(caiduser__uploadedarchive=instance)
    for workgroup_id in workgroups.values_list("pk", flat=True).distinct():
        transaction.on_commit(partial(schedule_embeddings, workgroup_id), using=using, robust=True)
