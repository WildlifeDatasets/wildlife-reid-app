from .models import CaIDUser, MediaFile, UploadedArchive


def _user_has_rw_access_to_mediafile(ciduser: CaIDUser, mediafile: MediaFile) -> bool:
    """Check if user has access to mediafile."""
    return (mediafile.parent.owner.id == ciduser.id) or (
        mediafile.parent.owner.workgroup == ciduser.workgroup
    )


def _user_has_rw_acces_to_uploadedarchive(
    ciduser: CaIDUser, uploadedarchive: UploadedArchive
) -> bool:
    """Check if user has access to uploadedarchive."""
    return (uploadedarchive.owner.id == ciduser.id) or (
        uploadedarchive.owner.workgroup == ciduser.workgroup
    )
