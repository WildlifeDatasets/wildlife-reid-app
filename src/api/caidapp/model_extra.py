from .models import CaIDUser, MediaFile, UploadedArchive, Location
import pandas as pd


def _user_has_rw_access_to_mediafile(ciduser: CaIDUser, mediafile: MediaFile, accept_none: bool) -> bool:
    """Check if user has access to mediafile."""
    if mediafile is None:
        if accept_none:
            return True
        return False
    if mediafile.parent is None:
        if accept_none:
            return True
        return False
    return (mediafile.parent.owner.id == ciduser.id) or (
        mediafile.parent.owner.workgroup == ciduser.workgroup
    )


def _user_has_rw_acces_to_uploadedarchive(
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

    return (uploadedarchive.owner.id == ciduser.id) or (
        uploadedarchive.owner.workgroup == ciduser.workgroup
    )


def prepare_dataframe_for_uploads_in_one_location(location_id: int) -> pd.DataFrame:
    """Prepare dataframe for uploads in one location."""
    location = Location.objects.get(id=location_id)

    location_uploads = UploadedArchive.objects.filter(location_at_upload_object=location).order_by("uploaded_at")

    df = pd.DataFrame.from_records(location_uploads.values())




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
            "location_at_upload_object_id",
            "owner_id",
        ])

    # add columns for location
    for upload in location_uploads:
        df.loc[df["id"] == upload.id, "count_of_mediafiles"] = upload.count_of_mediafiles()
        df.loc[df["id"] == upload.id, "count_of_representative_mediafiles"] = upload.count_of_representative_mediafiles()
        df.loc[df["id"] == upload.id, "count_of_mediafiles_with_taxon_for_identification"] = upload.count_of_mediafiles_with_taxon_for_identification()
        df.loc[df["id"] == upload.id, "earliest_captured_taxon"] = upload.earliest_captured_taxon()
        df.loc[df["id"] == upload.id, "latest_captured_taxon"] = upload.latest_captured_taxon()
    #

    # df["uploaded_at"] = df["uploaded_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # df["uploaded_at"] = pd.to_datetime(df["uploaded_at"])
    return df