import logging
import os.path
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import codenamize
from django.contrib.auth import get_user_model
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from location_field.models.plain import PlainLocationField

from .model_tools import (
    _get_zip_path_in_unique_folder,
    generate_sha1,
    get_output_dir,
    random_string,
    random_string12,
)

# Create your models here.
logger = logging.getLogger("database")


class WorkGroup(models.Model):
    name = models.CharField(max_length=50)
    hash = models.CharField(max_length=50, default=random_string12)
    identification_init_at = models.DateTimeField("Identification init at", blank=True, null=True)
    identification_init_status = models.CharField(
        max_length=255, blank=True, default="Not initiated"
    )
    identification_init_message = models.CharField(max_length=255, blank=True, default="")

    def __str__(self):
        return str(self.name)


class CaIDUser(models.Model):
    DjangoUser = get_user_model()
    id = models.AutoField(primary_key=True)
    user = models.OneToOneField(DjangoUser, on_delete=models.CASCADE)
    hash = models.CharField(max_length=50, default=random_string12)
    workgroup = models.ForeignKey(WorkGroup, on_delete=models.CASCADE, null=True, blank=True)
    workgroup_admin = models.BooleanField(default=False)
    import_dir = models.CharField(max_length=255, blank=True, default="")
    dir_import_status = models.CharField(max_length=255, blank=True, default="")
    dir_import_message = models.CharField(max_length=255, blank=True, default="")

    @receiver(post_save, sender=DjangoUser)
    def create_user_profile(sender, instance, created, **kwargs):  # NOSONAR
        """Create object when django user is created."""
        if created:
            CaIDUser.objects.create(user=instance)

    @receiver(post_save, sender=DjangoUser)
    def save_user_profile(sender, instance, **kwargs):  # NOSONAR
        """Save object when django user is saved."""
        logger.debug(sender)
        logger.debug(instance)
        logger.debug(kwargs)
        # pdb.set_trace()

        if not hasattr(instance, "caiduser"):
            profile, _ = CaIDUser.objects.get_or_create(user=instance)
            instance.caiduser = profile
        # UserProfile.objects.get_or_create(user=request.user)
        instance.caiduser.save()

    def __str__(self):
        return str(self.user)


def _hash():
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    hash_str = generate_sha1(dt, salt=random_string())
    return hash_str


def human_readable_hash():
    """Return a human readable hash composed from words."""
    number_of_words = 3
    return codenamize.codenamize(_hash(), number_of_words - 1, 0, " ", True)


class Taxon(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self):
        return str(self.name)


class Location(models.Model):
    name = models.CharField(max_length=50)
    visible_name = models.CharField(max_length=255, blank=True, default=human_readable_hash)
    location = PlainLocationField(
        based_fields=["city"],
        zoom=7,
        null=True,
        blank=True,
    )
    # If the user is deleted, then we will keep the location but it does not
    # belong to any user which is not good.
    owner = models.ForeignKey(CaIDUser, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return str(self.name)


class UploadedArchive(models.Model):
    uploaded_at = models.DateTimeField("Uploaded at", default=datetime.now)
    archivefile = models.FileField(
        "Archive File",
        upload_to=_get_zip_path_in_unique_folder,
        max_length=500,
    )
    preview = models.ImageField(blank=True, null=True)
    outputdir = models.CharField(max_length=255, blank=True, default=get_output_dir)
    thumbnail = models.ImageField(upload_to=outputdir, blank=True)
    zip_file = models.FileField(upload_to=outputdir, blank=True, null=True)
    csv_file = models.FileField(upload_to=outputdir, blank=True, null=True)
    output_updated_at = models.DateTimeField("Output updated at", blank=True, null=True)
    hash = models.CharField(max_length=255, blank=True, default=_hash)
    status = models.CharField(max_length=255, blank=True, default="Created")
    # status_message = models.CharField(max_length=2047, blank=True, default="")
    status_message = models.TextField(blank=True)
    started_at = models.DateTimeField("Started at", blank=True, null=True)
    finished_at = models.DateTimeField("Finished at", blank=True, null=True)
    identification_status = models.CharField(max_length=255, blank=True, default="Created")
    identification_started_at = models.DateTimeField("Started at", blank=True, null=True)
    identification_finished_at = models.DateTimeField("Finished at", blank=True, null=True)
    location_at_upload = models.CharField(max_length=255, blank=True, default="")
    location_at_upload_object = models.ForeignKey(
        Location, on_delete=models.SET_NULL, null=True, blank=True
    )
    owner = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    contains_identities = models.BooleanField(default=False)
    contains_single_taxon = models.BooleanField(default=False)
    taxon_for_identification_at_upload = models.CharField(max_length=255, blank=True, default="")
    taxon_for_identification = models.ForeignKey(
        Taxon, on_delete=models.SET_NULL, null=True, blank=True
    )
    mediafiles_imported = models.BooleanField("Media Files Imported Correctly", default=False)
    earliest_captured_at = models.DateTimeField("Earliest Captured at", blank=True, null=True)
    latest_captured_at = models.DateTimeField("Latest Captured at", blank=True, null=True)

    def update_location_in_mediafiles(self, location: Union[str, Location]):
        """Update location in mediafiles."""
        if isinstance(location, str):
            location = get_location(self.owner, location)
        mediafiles = MediaFile.objects.filter(parent=self)
        for mediafile in mediafiles:
            mediafile.location = location
        self.location_at_upload_object = location
        self.location = location.name

    def update_earliest_and_latest_captured_at(self):
        """Update the earliest and latest captured at in the archive based on mediafiles."""
        mediafiles = MediaFile.objects.filter(parent=self)
        earliest_captured_at = None
        latest_captured_at = None
        for mediafile in mediafiles:
            if mediafile.captured_at is not None:
                if earliest_captured_at is None or mediafile.captured_at < earliest_captured_at:
                    earliest_captured_at = mediafile.captured_at
                if latest_captured_at is None or mediafile.captured_at > latest_captured_at:
                    latest_captured_at = mediafile.captured_at
                logger.debug(f"{mediafile=}")
        logger.debug(f"{mediafiles.count()=}")
        logger.debug(f"{earliest_captured_at=}, {latest_captured_at=}")
        self.earliest_captured_at = earliest_captured_at
        self.latest_captured_at = latest_captured_at
        self.save()

    # def update_earliest_and_latest_captured_at(self):
    #     mediafiles = MediaFile.objects.filter(parent=self)
    #     earliest_captured_at = None
    #     latest_captured_at = None
    #     for mediafile in mediafiles:
    #         if earliest_captured_at is None:
    #             earliest_captured_at = mediafile.captured_at
    #         if latest_captured_at is None:
    #             latest_captured_at = mediafile.captured_at
    #         # convert to datetime
    #         if mediafile.captured_at < earliest_captured_at:
    #             earliest_captured_at = mediafile.captured_at
    #         if mediafile.captured_at > latest_captured_at:
    #             latest_captured_at = mediafile.captured_at
    #     self.earliest_captured_at = earliest_captured_at
    #     self.latest_captured_at = latest_captured_at

    def __str__(self):
        return str(Path(self.archivefile.name).name)


@receiver(models.signals.post_delete, sender=UploadedArchive)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    """Deletes file from filesystem when corresponding `MediaFile` object is deleted."""
    if instance.archivefile and os.path.isfile(instance.archivefile.path):
        os.remove(instance.archivefile.path)
    if instance.outputdir and os.path.isdir(instance.outputdir):
        shutil.rmtree(instance.outputdir)


# class Notification(models.Model):
#     title = models.CharField(max_length=255)
#     message = models.CharField(max_length=255)
#     owner = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
#     created_at = models.DateTimeField("Created at", default=datetime.now)
#     read_at = models.DateTimeField("Read at", blank=True, null=True)
#
#     def __str__(self):
#         return str(self.title)


class IndividualIdentity(models.Model):
    SEX_CHOICES = (
        ('M', "Male"),
        ('F', 'Female'),
        ('U', 'Unknown'),
    )
    COAT_TYPE_CHOICES = (
        ('S', "Spots"),
        ("M", "Marble"),
        ("U", "Unknown"),
    )
    name = models.CharField(max_length=50)
    id_worker = models.IntegerField(null=True, blank=True)
    owner_workgroup = models.ForeignKey(WorkGroup, on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    sex = models.CharField(max_length=2, choices=SEX_CHOICES, default='U')
    coat_type = models.CharField(max_length=2, choices=COAT_TYPE_CHOICES, default='U')
    note = models.TextField(blank=True)

    def __str__(self):
        return str(self.name)


class MediaFile(models.Model):
    ORIENTATION_CHOICES = (
        ("L", "Left"),
        ("R", "Right"),
        ("N", "None"),
    )
    parent = models.ForeignKey(UploadedArchive, on_delete=models.CASCADE, null=True)
    category = models.ForeignKey(Taxon, blank=True, null=True, on_delete=models.CASCADE)
    location = models.ForeignKey(Location, blank=True, null=True, on_delete=models.CASCADE)
    captured_at = models.DateTimeField("Captured at", blank=True, null=True)
    mediafile = models.FileField(
        "Media File",
        blank=True,
        null=True,
        max_length=500,
    )
    # image representation of mediafile (orig file for images, single frame for videos
    image_file = models.FileField(
        "Image File",
        blank=True,
        null=True,
        max_length=500,
    )
    thumbnail = models.ImageField(blank=True, null=True, max_length=500)
    identity = models.ForeignKey(
        IndividualIdentity, blank=True, null=True, on_delete=models.SET_NULL
    )
    identity_is_representative = models.BooleanField(default=False)
    updated_by = models.ForeignKey(CaIDUser, on_delete=models.SET_NULL, null=True, blank=True)
    updated_at = models.DateTimeField("Updated at", blank=True, null=True)
    metadata_json = models.JSONField(blank=True, null=True)
    animal_number = models.IntegerField(null=True, blank=True)
    media_type = models.CharField(max_length=255, blank=True, default="image")
    orientation = models.CharField(max_length=2, choices=ORIENTATION_CHOICES, default="N")
    original_filename = models.CharField(max_length=512, blank=True, default="")

    class Meta:
        ordering = ["-identity_is_representative", "captured_at"]

    def __str__(self):
        return str(Path(self.mediafile.name).name)


class AnimalObservation(models.Model):
    mediafile = models.ForeignKey(MediaFile, on_delete=models.CASCADE, null=True, blank=True)
    taxon = models.ForeignKey(Taxon, on_delete=models.SET_NULL, null=True, blank=True)
    metadata_json = models.JSONField(blank=True, null=True)
    updated_by = models.ForeignKey(CaIDUser, on_delete=models.SET_NULL, null=True, blank=True)
    updated_at = models.DateTimeField("Updated at", blank=True, null=True)


class MediafilesForIdentification(models.Model):
    mediafile = models.ForeignKey(MediaFile, on_delete=models.SET_NULL, null=True, blank=True)

    top1mediafile = models.ForeignKey(
        MediaFile, related_name="top1", on_delete=models.SET_NULL, null=True, blank=True
    )
    top2mediafile = models.ForeignKey(
        MediaFile, related_name="top2", on_delete=models.SET_NULL, null=True, blank=True
    )
    top3mediafile = models.ForeignKey(
        MediaFile, related_name="top3", on_delete=models.SET_NULL, null=True, blank=True
    )
    top1score = models.FloatField(null=True, blank=True)
    top2score = models.FloatField(null=True, blank=True)
    top3score = models.FloatField(null=True, blank=True)
    top1name = models.CharField(max_length=255, blank=True, default="")
    top2name = models.CharField(max_length=255, blank=True, default="")
    top3name = models.CharField(max_length=255, blank=True, default="")
    paired_points = models.JSONField(blank=True, null=True)


class Album(models.Model):
    name = models.CharField(max_length=50)
    description = models.CharField(max_length=255, blank=True, default="")
    owner = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    mediafiles = models.ManyToManyField(MediaFile, blank=True)
    created_at = models.DateTimeField("Created at", default=datetime.now)
    hash = models.CharField(max_length=255, blank=True, default=_hash)
    public_hash = models.CharField(max_length=255, blank=True, default=_hash)
    cover = models.ForeignKey(
        MediaFile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="cover",
    )

    def __str__(self):
        return str(self.name)

    def get_absolute_url(self):
        """Return absolute url."""
        return f"/album/{str(self.hash)}/"


class ArchiveCollection(models.Model):
    name = models.CharField(max_length=50)
    owner = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    archives = models.ManyToManyField(UploadedArchive, blank=True)
    created_at = models.DateTimeField("Created at", default=datetime.now)
    hash = models.CharField(max_length=255, blank=True, default=_hash)
    starts_at = models.DateTimeField("Starts at", blank=True, null=True)
    ends_at = models.DateTimeField("Ends at", blank=True, null=True)

    def __str__(self):
        return str(self.name)

    # def get_absolute_url(self):
    #     """Return absolute url."""
    #     return f"/album/{str(self.hash)}/"


class AlbumShareRoleType(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self):
        return str(self.name)


class AlbumShareRole(models.Model):
    album = models.ForeignKey(Album, on_delete=models.CASCADE, null=True, blank=True)
    user = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    role = models.ForeignKey(AlbumShareRoleType, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return str(self.album.name) + " " + str(self.user.user.username)


def get_unique_name(name: str, workgroup: WorkGroup) -> Optional[IndividualIdentity]:
    """Return taxon according to the name, create it if necessary."""
    if (name is None) or (name == ""):
        return None
    objs = IndividualIdentity.objects.filter(name=name, owner_workgroup=workgroup)
    if len(objs) == 0:
        identity = IndividualIdentity(name=name, owner_workgroup=workgroup)
        identity.save()
    else:
        identity = objs[0]
    return identity


def get_taxon(name: str) -> Optional[Taxon]:
    """Return taxon according to the name, create it if necessary."""
    if (name is None) or (name == ""):
        return None
    objs = Taxon.objects.filter(name=name)
    if len(objs) == 0:
        taxon = Taxon(name=name)
        taxon.save()
    else:
        taxon = objs[0]
    return taxon


def get_location(caiduser: CaIDUser, name: str) -> Location:
    """Return location according to the name, create it if necessary.

    Parameters
    ----------
    request
    """
    objs = Location.objects.filter(name=name, owner__workgroup=caiduser.workgroup)
    if len(objs) == 0:
        location = Location(name=name, owner=caiduser)
        location.save()
    else:
        location = objs[0]
    return location


def get_content_owner_filter_params(ciduser: CaIDUser, prefix: str) -> dict:
    """Parameters for filtering user content based on existence of workgroup.

    Parameters
    ----------
    request : HttpRequest
        Request object.
    prefix : str
        Prefix for filtering with ciduser.
        If the filter will be used in MediaFile, the prefix should be "parent__owner".
        If the filter will be used in Location, the prefix should be "owner".
    """
    if ciduser.workgroup:
        # filter_params = dict(parent__owner__workgroup=request.user.caiduser.workgroup)
        filter_params = {f"{prefix}__workgroup": ciduser.workgroup}
    else:
        filter_params = {f"{prefix}": ciduser}
    return filter_params
