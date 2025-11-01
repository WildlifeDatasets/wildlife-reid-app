import logging

import Levenshtein
import pandas as pd
from tqdm import tqdm

from . import models
from .model_tools import order_identity_by_mediafile_count, remove_diacritics
from .models import CaIDUser, IndividualIdentity, Locality, MediaFile, MergeIdentitySuggestionResult, UploadedArchive

logger = logging.getLogger(__name__)


def user_has_access_to_uploadedarchives_filter_params(caiduser: CaIDUser):
    return models.user_has_access_filter_params(caiduser=caiduser, prefix="owner")


def user_has_access_to_mediafiles_filter_params(caiduser: CaIDUser):
    return models.user_has_access_filter_params(caiduser=caiduser, prefix="parent__owner")


def user_has_rw_access_to_mediafile(ciduser: CaIDUser, mediafile: MediaFile, accept_none: bool) -> bool:
    """Check if user has access to mediafile."""
    if mediafile is None:
        if accept_none:
            return True
        return False
    if mediafile.parent is None:
        if accept_none:
            return True
        return False
    return (mediafile.parent.owner.id == ciduser.id) or (mediafile.parent.owner.workgroup == ciduser.workgroup)


def user_has_rw_acces_to_uploadedarchive(
    ciduser: CaIDUser, uploadedarchive: UploadedArchive, accept_none: bool = False
) -> bool:
    """Check if user has access to uploadedarchive."""
    if uploadedarchive is None:
        if accept_none:
            return True
        return False
    if uploadedarchive.owner is None:
        if accept_none:
            return True
        return False

    return (uploadedarchive.owner.id == ciduser.id) or (uploadedarchive.owner.workgroup == ciduser.workgroup)


def prepare_dataframe_for_uploads_in_one_locality(locality_id: int) -> pd.DataFrame:
    """Prepare dataframe for uploads in one locality."""
    locality = Locality.objects.get(id=locality_id)

    locality_uploads = UploadedArchive.objects.filter(locality_at_upload_object=locality).order_by("uploaded_at")

    df = pd.DataFrame.from_records(locality_uploads.values())

    # remove unnecessary columns
    df = df.drop(
        columns=[
            "status_message",
            "preview",
            "outputdir",
            "thumbnail",
            "zip_file",
            "csv_file",
            "hash",
            "locality_at_upload_object_id",
            "owner_id",
        ]
    )

    # add columns for locality
    for upload in locality_uploads:
        df.loc[df["id"] == upload.id, "count_of_mediafiles"] = upload.count_of_mediafiles()
        df.loc[df["id"] == upload.id, "count_of_representative_mediafiles"] = (
            upload.count_of_representative_mediafiles()
        )
        df.loc[df["id"] == upload.id, "count_of_mediafiles_with_taxon_for_identification"] = (
            upload.count_of_mediafiles_with_taxon_for_identification()
        )
        df.loc[df["id"] == upload.id, "earliest_captured_taxon"] = upload.earliest_captured_taxon()
        df.loc[df["id"] == upload.id, "latest_captured_taxon"] = upload.latest_captured_taxon()
    #

    # df["uploaded_at"] = df["uploaded_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # df["uploaded_at"] = pd.to_datetime(df["uploaded_at"])
    return df


def compute_identity_suggestions(workgroup_id: int, limit: int = 100) -> int:
    # user = get_user_model().objects.get(id=user_id)
    suggestions = []
    workgroup = models.WorkGroup.objects.get(id=workgroup_id)
    print(f"Computing identity suggestions for workgroup {workgroup.name} ({workgroup.id})")
    logger.debug(f"Computing identity suggestions for workgroup {workgroup.name} ({workgroup.id})")

    all_identities = IndividualIdentity.objects.filter(
        owner_workgroup=workgroup,
    )
    total = all_identities.count()
    print(f"Total identities: {total}")

    for i, identity1 in tqdm(enumerate(all_identities), total=total, desc="Computing identity suggestions"):
        for j in range(i + 1, len(all_identities)):
            identity2 = all_identities[j]
            if identity1 == identity2:
                continue

            if identity1.code == identity2.code:
                identity_a, identity_b = order_identity_by_mediafile_count(identity1, identity2)
                suggestions.append((identity_a.id, identity_b.id, 0))
                continue

            if abs(len(identity1.name) - len(identity2.name)) > 8:
                continue

            identity1_name = remove_diacritics(identity1.name)
            identity2_name = remove_diacritics(identity2.name)
            distance = Levenshtein.distance(identity1_name, identity2_name)

            if distance < (len(identity1_name) / 4.0 + len(identity2_name) / 4.0):
                identity_a, identity_b = order_identity_by_mediafile_count(identity1, identity2)
                suggestions.append((identity_a.id, identity_b.id, distance))

    # seřadit
    suggestions.sort(key=lambda x: (x[2], -len(IndividualIdentity.objects.get(id=x[1]).name)))

    # uložit výsledek do DB
    result = MergeIdentitySuggestionResult.objects.create(
        workgroup=workgroup,
        suggestions=suggestions,
    )
    return result.id
