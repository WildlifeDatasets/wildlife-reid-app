import logging
from datetime import datetime
from pathlib import Path

from django.contrib.auth import get_user_model
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver

from .model_tools import (
    generate_sha1,
    get_output_dir,
    randomString,
    randomString12,
    upload_to_unqiue_folder,
)

# Create your models here.
logger = logging.getLogger("database")

User = get_user_model()


class CIDUser(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # bio = models.TextField(max_length=500, blank=True)
    # location = models.CharField(max_length=30, blank=True)
    # birth_date = models.DateField(null=True, blank=True)
    hash = models.CharField(max_length=50, default=randomString12)

    @receiver(post_save, sender=User)
    def create_user_profile(sender, instance, created, **kwargs):
        """TODO add docstring."""
        if created:
            CIDUser.objects.create(user=instance)

    @receiver(post_save, sender=User)
    def save_user_profile(sender, instance, **kwargs):
        """TODO add docstring."""
        logger.debug(sender)
        logger.debug(instance)
        logger.debug(kwargs)
        # pdb.set_trace()
        from django.core.exceptions import ObjectDoesNotExist

        try:
            instance.ciduser
        except ObjectDoesNotExist:
            profile, created = CIDUser.objects.get_or_create(user=instance)
            instance.ciduser = profile
        # UserProfile.objects.get_or_create(user=request.user)
        instance.ciduser.save()


def _hash():
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    hash = generate_sha1(dt, salt=randomString())
    return hash


class UploadedArchive(models.Model):
    # email = models.EmailField(max_length=200)
    # hash = scaffanweb_tools.randomString(12)
    uploaded_at = models.DateTimeField("Uploaded at", default=datetime.now)
    archivefile = models.FileField(
        "Archive File",
        upload_to=upload_to_unqiue_folder,
        # blank=True,
        # null=True,
        max_length=500,
    )
    preview = models.ImageField(blank=True, null=True)
    outputdir = models.CharField(max_length=255, blank=True, default=get_output_dir)
    thumbnail = models.ImageField(upload_to=outputdir, blank=True)
    zip_file = models.FileField(upload_to=outputdir, blank=True, null=True)
    csv_file = models.FileField(upload_to=outputdir, blank=True, null=True)
    hash = models.CharField(max_length=255, blank=True, default=_hash)
    started_at = models.DateTimeField("Started at", blank=True, null=True)
    finished_at = models.DateTimeField("Finished at", blank=True, null=True)
    owner = models.ForeignKey(CIDUser, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return str(Path(self.archivefile.name).name)


class Taxon(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self):
        return str(self.name)


class MediaFile(models.Model):
    parent = models.ForeignKey(UploadedArchive, on_delete=models.CASCADE)
    species = models.ForeignKey(Taxon, blank=True, null=True, on_delete=models.CASCADE)
    mediafile = models.FileField(
        "Media File",
        # upload_to=upload_to_unqiue_folder,
        blank=True,
        null=True,
        max_length=500,
    )

    def __str__(self):
        return str(Path(self.mediafile.name).name)
