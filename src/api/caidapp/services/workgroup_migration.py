import logging

from django.core.exceptions import ValidationError
from django.db import transaction

from .. import models

logger = logging.getLogger(__name__)


@transaction.atomic
def migrate_user_to_workgroup(
    *,
    user: models.CaIDUser,
    target_workgroup: models.WorkGroup,
    approved_by: models.CaIDUser | None = None,
):
    """Permanently migrate a user and their personal workgroup data."""
    # Lock nullable relations separately. PostgreSQL rejects FOR UPDATE when
    # select_related() produces a LEFT OUTER JOIN to the nullable workgroup FK.
    user = models.CaIDUser.objects.select_for_update().get(pk=user.pk)
    target_workgroup = models.WorkGroup.objects.select_for_update().get(pk=target_workgroup.pk)
    source_workgroup_id = user.workgroup_id

    if source_workgroup_id == target_workgroup.pk:
        raise ValidationError("The user is already a member of the target workgroup.")
    if source_workgroup_id is None:
        raise ValidationError("The user must belong to a personal workgroup before migration.")

    source_workgroup = models.WorkGroup.objects.select_for_update().get(pk=source_workgroup_id)
    source_members = models.CaIDUser.objects.select_for_update().filter(workgroup=source_workgroup)
    if source_members.count() != 1 or source_members.first().pk != user.pk:
        raise ValidationError("Only users in a one-member personal workgroup can accept an invitation.")

    logger.info(
        "Migrating user %s from %s to %s (approved by %s)",
        user,
        source_workgroup,
        target_workgroup,
        approved_by,
    )

    # Identification suggestions are transient and may refer to the old model.
    unfinished_identifications = models.MediafilesForIdentification.objects.filter(
        mediafile__parent__owner=user
    )
    affected_archive_ids = list(
        unfinished_identifications.values_list("mediafile__parent_id", flat=True).distinct()
    )
    unfinished_identifications.delete()
    models.UploadedArchive.objects.filter(pk__in=affected_archive_ids).update(
        identification_status="C",
        identification_started_at=None,
        identification_finished_at=None,
    )

    # Archives, localities, albums, collections, and devices remain owned by the
    # user. Identities are the data that are owned directly by a workgroup.
    models.IndividualIdentity.objects.filter(owner_workgroup=source_workgroup).update(
        owner_workgroup=target_workgroup
    )

    user.workgroup = target_workgroup
    user.workgroup_admin = False
    user.save(update_fields=["workgroup", "workgroup_admin"])

    # The source is now empty. Cascades remove its obsolete model and result data.
    source_workgroup.delete()
    logger.info("Migration completed: user %s is now in workgroup %s", user, target_workgroup)
    return user
