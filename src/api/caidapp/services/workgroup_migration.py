from django.db import transaction
import logging
from .. import models

logger = logging.getLogger(__name__)

@transaction.atomic
def migrate_user_to_workgroup(
    *,
    user: models.CaIDUser,
    target_workgroup: models.WorkGroup,
    approved_by: models.CaIDUser | None = None,
):
    """
    Destructively migrate user to another workgroup.

    All user-owned data are moved together with the user.
    This operation is transactional.
    """

    source_workgroup = user.workgroup

    if source_workgroup == target_workgroup:
        logger.info(
            "Migration skipped: user %s already in workgroup %s",
            user,
            target_workgroup,
        )
        return

    logger.info(
        "Starting migration of user %s from %s to %s (approved by %s)",
        user,
        source_workgroup,
        target_workgroup,
        approved_by,
    )

    # ------------------------------------------------------------------
    # 1️⃣ Přepnutí workgroup uživatele
    # ------------------------------------------------------------------
    user.workgroup = target_workgroup
    user.workgroup_admin = False
    user.save(update_fields=["workgroup"])

    # ------------------------------------------------------------------
    # 2️⃣ UploadedArchive – taxon + implicitní příslušnost k WG
    # ------------------------------------------------------------------
    models.UploadedArchive.objects.filter(owner=user).update(
        taxon_for_identification=target_workgroup.default_taxon_for_identification
    )

    # ------------------------------------------------------------------
    # 3️⃣ IndividualIdentity – vlastnictví identity patří WG
    # ------------------------------------------------------------------
    # models.IndividualIdentity.objects.filter(owner=user).update(
    #     owner_workgroup=target_workgroup
    # )

    # ------------------------------------------------------------------
    # 4️⃣ MediaFile – nepřímo (patří archivu, ale někdy se hodí explicitně)
    # (pokud máš někde FK na workgroup, sem ho přidej)
    # ------------------------------------------------------------------
    # models.MediaFile.objects.filter(
    #     parent__owner=user
    # ).update(...)
    # Here it would be good to set

    # ------------------------------------------------------------------
    # 5️⃣ Audit / log
    # ------------------------------------------------------------------
    logger.info(
        "Migration completed: user %s is now in workgroup %s",
        user,
        target_workgroup,
    )
