import logging
import os.path
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from django.contrib.auth import get_user_model
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from location_field.models.plain import PlainLocationField

from .model_tools import (
    generate_sha1,
    get_output_dir,
    random_string,
    random_string12,
    upload_to_unqiue_folder,
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

class Taxon(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self):
        return str(self.name)

class UploadedArchive(models.Model):
    uploaded_at = models.DateTimeField("Uploaded at", default=datetime.now)
    archivefile = models.FileField(
        "Archive File",
        upload_to=upload_to_unqiue_folder,
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
    started_at = models.DateTimeField("Started at", blank=True, null=True)
    finished_at = models.DateTimeField("Finished at", blank=True, null=True)
    identification_status = models.CharField(max_length=255, blank=True, default="Created")
    identification_started_at = models.DateTimeField("Started at", blank=True, null=True)
    identification_finished_at = models.DateTimeField("Finished at", blank=True, null=True)
    location_at_upload = models.CharField(max_length=255, blank=True, default="")
    owner = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    contains_identities = models.BooleanField(default=False)
    contains_single_taxon = models.BooleanField(default=False)
    taxon_for_identification = models.ForeignKey(Taxon, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return str(Path(self.archivefile.name).name)


@receiver(models.signals.post_delete, sender=UploadedArchive)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    """Deletes file from filesystem when corresponding `MediaFile` object is deleted."""
    if instance.archivefile and os.path.isfile(instance.archivefile.path):
        os.remove(instance.archivefile.path)
    if instance.outputdir and os.path.isdir(instance.outputdir):
        shutil.rmtree(instance.outputdir)




class Location(models.Model):
    name = models.CharField(max_length=50)
    location = PlainLocationField(
        based_fields=["city"],
        zoom=7,
        null=True,
        blank=True,
    )
    owner = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return str(self.name)


class IndividualIdentity(models.Model):
    name = models.CharField(max_length=50)
    id_worker = models.IntegerField(null=True, blank=True)
    owner_workgroup = models.ForeignKey(WorkGroup, on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return str(self.name)


class MediaFile(models.Model):
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
    updated_by = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    updated_at = models.DateTimeField("Updated at", blank=True, null=True)
    metadata_json = models.JSONField(blank=True, null=True)

    class Meta:
        ordering = ["-identity_is_representative", "captured_at"]

    def __str__(self):
        return str(Path(self.mediafile.name).name)

class AnimalObservation(models.Model):
    mediafile = models.ForeignKey(MediaFile, on_delete=models.CASCADE, null=True, blank=True)
    taxon = models.ForeignKey(Taxon, on_delete=models.CASCADE, null=True, blank=True)
    metadata_json = models.JSONField(blank=True, null=True)


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
