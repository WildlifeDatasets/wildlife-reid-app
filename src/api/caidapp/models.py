import logging
import os.path
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import codenamize
import numpy as np
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.db import models
from django.db.models import Count, Q
from django.db.models.query import QuerySet
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.urls import reverse_lazy
from location_field.models.plain import PlainLocationField
from tqdm import tqdm

from . import fs_data
from .fs_data import convert_to_mp4
from .model_tools import (
    generate_sha1,
    get_output_dir,
    get_zip_path_in_unique_folder,
    random_string,
    random_string12,
)

# Create your models here.
logger = logging.getLogger("database")

UA_STATUS_CHOICES = (
    ("C", "Created"),
    ("F", "Failed"),
    ("TAIP", "Importing"),  # former Taxon Processing
    ("TAID", "Missing taxa"),
    ("TKN", "Taxa known"),
    ("TV", "Taxa verified"),
    ("IR", "Ready for ID"),
    ("IAIP", "ID processing"),
    ("IAID", "ID AI done"),
    ("ID", "Identified"),
    ("U", "Unknown"),
)
UA_STATUS_CHOICES_DICT = dict(UA_STATUS_CHOICES)

ORIENTATION_CHOICES = (
    ("L", "Left"),
    ("R", "Right"),
    ("F", "Front"),
    ("B", "Back"),
    ("N", "None"),
    ("U", "Unknown"),
)
TAXON_NOT_CLASSIFIED = "Not Classified"


def get_hash():
    """Return a hash composed from date and random string."""
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    hash_str = generate_sha1(dt, salt=random_string())
    return hash_str


def get_hash8():
    """Return a hash composed from date and random string."""
    return get_hash()[:8]


def human_readable_hash():
    """Return a human readable hash composed from words."""
    number_of_words = 3
    return codenamize.codenamize(get_hash(), number_of_words - 1, 0, " ", True)


class Taxon(models.Model):
    name = models.CharField(max_length=50)
    parent = models.ForeignKey("self", on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return str(self.name)


class IdentificationModel(models.Model):
    name = models.CharField(max_length=50)
    description = models.CharField(max_length=255, blank=True, default="")
    public = models.BooleanField(default=False)
    model_path = models.CharField(max_length=255, blank=True, default="")
    workgroup = models.ForeignKey(
        "WorkGroup",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="identification_models",
    )

    def __str__(self):
        return str(self.name)


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


class WorkGroup(models.Model):
    name = models.CharField(max_length=50)
    hash = models.CharField(max_length=50, default=random_string12)
    identification_init_at = models.DateTimeField("Identification init at", blank=True, null=True)
    identification_init_status = models.CharField(max_length=255, blank=True, default="Not initiated")
    identification_init_model_path = models.CharField(max_length=512, blank=True, default="")
    identification_init_message = models.TextField(blank=True, default="")
    identification_reid_at = models.DateTimeField("Identification reid at", blank=True, null=True)
    identification_reid_status = models.CharField(max_length=255, blank=True, default="Not initiated")
    identification_reid_message = models.TextField(blank=True, default="")
    identification_train_status = models.CharField(max_length=255, blank=True, default="")
    sequence_time_limit = models.IntegerField("Sequence time limit [s]", default=120)
    identification_scheduled_init_task_id = models.CharField(max_length=255, null=True, blank=True)
    identification_scheduled_init_eta = models.DateTimeField(null=True, blank=True)
    identification_scheduled_run_task_id = models.CharField(max_length=255, null=True, blank=True)
    identification_scheduled_run_eta = models.DateTimeField(null=True, blank=True)
    check_taxon_before_identification = models.BooleanField(
        "Check taxon before identification",
        default=True,
        help_text="Do the identification only for media files and observations with the correct taxon. "
        + "Ignore the other observations and media files.",
    )
    default_taxon_for_identification = models.ForeignKey(
        Taxon,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    identification_model = models.ForeignKey(
        IdentificationModel,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="actual_workgroup_identification_model",
    )

    def save(self, *args, **kwargs):
        """Save workgroup and set default taxon and identification model if not set."""
        # if not self.default_taxon_for_identification:
        #     self.default_taxon_for_identification = get_taxon("Lynx lynx")

        if not self.identification_model:
            model = IdentificationModel.objects.filter(public=True).first()
            self.identification_model = model
        super().save(*args, **kwargs)

    def __str__(self):
        return str(self.name)

    def number_of_uploaded_archives(self) -> int:
        """Return number of uploaded archives."""
        return UploadedArchive.objects.filter(owner__workgroup=self).count()

    def number_of_uploaded_media_files(self) -> int:
        """Return number of uploaded files."""
        return MediaFile.objects.filter(parent__owner__workgroup=self).count()

    def number_of_media_files_with_taxon_for_identification(self) -> int:
        """Return number of uploaded files with taxon for identification."""
        return MediaFile.objects.filter(
            parent__owner__workgroup=self, parent__taxon_for_identification__isnull=False
        ).count()

    def number_of_uploaded_archives_with_taxon_for_identification(self) -> int:
        """Return number of uploaded archives with taxon for identification."""
        return UploadedArchive.objects.filter(owner__workgroup=self, taxon_for_identification__isnull=False).count()

    def number_of_media_files_with_missing_identity(self) -> int:
        """Return number of uploaded files with missing identity."""
        return MediaFile.objects.filter(
            parent__owner__workgroup=self,
            identity__isnull=True,
            parent__taxon_for_identification__isnull=False,
        ).count()

    def number_of_representative_media_files(self):
        """Return number of representative media files."""
        return MediaFile.objects.filter(
            parent__owner__workgroup=self,
            identity_is_representative=True,
            parent__taxon_for_identification__isnull=False,
        ).count()

    def number_of_uploaded_archives_ready_for_identification(self) -> int:
        """Return number of uploaded archives ready for identification."""
        return UploadedArchive.objects.filter(
            owner__workgroup=self,
            taxon_for_identification__isnull=False,
            identification_status="IR",
        ).count()

    def number_of_media_files_in_uploaded_archives_ready_for_identification(self) -> int:
        """Return number of media files in uploaded archives ready for identification."""
        return MediaFile.objects.filter(
            parent__owner__workgroup=self,
            parent__taxon_for_identification__isnull=False,
            parent__identification_status="IR",
        ).count()


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
    identification_model = models.ForeignKey(IdentificationModel, on_delete=models.SET_NULL, null=True, blank=True)
    show_taxon_classification = models.BooleanField(default=True)
    show_wellcome_message_on_next_login = models.BooleanField(default=False)
    show_base_dataset = models.BooleanField(default=False)
    default_taxon_for_identification = models.ForeignKey(
        Taxon,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    timezone = models.CharField(max_length=50, blank=True, default=settings.TIME_ZONE)
    ml_consent_given = models.BooleanField(default=False)
    ml_consent_given_date = models.DateTimeField("ML consent given date", blank=True, null=True)

    @receiver(post_save, sender=DjangoUser)
    def create_user_profile(sender, instance, created, **kwargs):  # NOSONAR
        """Create object when django user is created."""
        if created:
            CaIDUser.objects.create(user=instance)

    @receiver(post_save, sender=DjangoUser)
    def save_user_profile(sender, instance, **kwargs):  # NOSONAR
        """Save object when django user is saved."""
        # logger.debug(sender)
        # logger.debug(instance)
        # logger.debug(kwargs)
        # pdb.set_trace()

        if not hasattr(instance, "caiduser"):
            profile, _ = CaIDUser.objects.get_or_create(user=instance)
            instance.caiduser = profile
        # UserProfile.objects.get_or_create(user=request.user)
        instance.caiduser.save()

    def save(self, *args, **kwargs):
        """Save user and set default taxon if not set."""
        if not self.default_taxon_for_identification:
            self.default_taxon_for_identification = get_taxon("Lynx lynx")
        super().save(*args, **kwargs)

    def __str__(self):
        user_str = str(self.user)
        if self.user.first_name:
            user_str += " " + self.user.first_name
        if self.user.last_name:
            user_str += " " + self.user.last_name
        return user_str

    def number_of_uploaded_archives(self) -> int:
        """Return number of uploaded archives."""
        return UploadedArchive.objects.filter(owner=self).count()

    def number_of_uploaded_media_files(self) -> int:
        """Return number of uploaded files."""
        return MediaFile.objects.filter(parent__owner=self).count()

    def number_of_uploaded_media_files_with_known_taxon(self) -> int:
        """Return number of uploaded files with known taxon."""
        return MediaFile.objects.filter(parent__owner=self, taxon__isnull=False).count()

    def number_of_media_files_with_missing_taxa_general(self) -> int:
        """Return number of uploaded files with missing taxon.

        If possible, we use data for whole workgroup.
        """
        return len(get_mediafiles_with_missing_taxon(self))

    def number_of_media_files_general(self) -> int:
        """Return number of uploaded files.

        If possible, we use data for whole workgroup.
        """
        return MediaFile.objects.filter(**user_has_access_filter_params(self, "parent__owner")).count()

    def number_of_media_files_with_missing_taxa_verification_general(self) -> int:
        """Return number of uploaded files with missing verification.

        If possible, we use data for whole workgroup.
        """
        return (
            MediaFile.objects.filter(**user_has_access_filter_params(self, "parent__owner"))
            .filter(taxon_verified=False)
            .count()
        )

    def number_of_identities(self) -> int:
        """Return number of identities."""
        return IndividualIdentity.objects.filter(owner_workgroup=self.workgroup).count()

    def number_of_media_files_with_missing_identity(self) -> int:
        """Return number of uploaded files with missing identity."""
        return MediaFile.objects.filter(
            parent__owner__workgroup=self.workgroup,
            identity__isnull=True,
            parent__taxon_for_identification__isnull=False,
        ).count()

    def number_of_media_files_with_known_identity(self) -> int:
        """Return number of uploaded files with missing identity."""
        return MediaFile.objects.filter(
            parent__owner__workgroup=self.workgroup,
            identity__isnull=False,
            parent__taxon_for_identification__isnull=False,
        ).count()

    def number_of_media_files_for_identification(self) -> int:
        """Return number of media files for identification."""
        return MediaFile.objects.filter(
            parent__owner__workgroup=self.workgroup,
            parent__taxon_for_identification__isnull=False,
        ).count()


class Locality(models.Model):
    name = models.CharField(max_length=50)
    visible_name = models.CharField(max_length=255, blank=True, default=human_readable_hash)
    location = PlainLocationField(
        based_fields=["city"],
        zoom=7,
        null=True,
        blank=True,
    )
    hash = models.CharField(max_length=50, default=get_hash8)
    # If the user is deleted, then we will keep the locality but it does not
    # belong to any user which is not good.
    owner = models.ForeignKey(CaIDUser, on_delete=models.SET_NULL, null=True, blank=True)
    note = models.TextField(blank=True, default="")
    area = models.ForeignKey("Area", on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return str(self.name)

    # def set_location_from_str(self, order: int = 3):
    #     """Round location for anonymization."""
    #     if (self.location is None) or (self.location == ""):
    #         return self.location
    #     location_str = str(self.location)
    #     if "," in location_str:
    #         lat, lon = location_str.split(",")
    #         self.set_location(lat, lon, order=order)
    #     else:
    #         logger.debug(f"Location {self.location} is not in format 'lat,lon'.")
    def set_location_from_str(self, location_str: str, order: int = 3):
        """Round location for anonymization."""
        if "," in location_str:
            lat, lon = location_str.split(",")
            self.set_location(lat, lon, order=order)
        else:
            logger.debug(f"Location {location_str} is not in format 'lat,lon'.")

    def set_location(self, lat: float, lon: float, order: int = 3):
        """Round location for anonymization."""
        lat = round(float(lat), order)
        lon = round(float(lon), order)
        self.location = f"{lat},{lon}"
        self.save()
        self.set_closest_area()

    # remove and use mediafile_set
    def mediafiles(self):
        """Return mediafiles."""
        return MediaFile.objects.filter(locality=self).all()

    def identities(self):
        """Return identities."""
        return IndividualIdentity.objects.filter(mediafile__locality=self).all()

    def set_closest_area(self):
        """Find area for the locality."""
        areas = Area.objects.all()

        if "," in str(self.location):
            lat, lon = str(self.location).split(",")
            lat = float(lat)
            lon = float(lon)

            closest_area = None
            closest_distance = 1000000
            for area in areas:
                if "," in str(area.location):
                    alat, alon = str(area.location).split(",")
                    alat = float(alat)
                    alon = float(alon)
                    distance = np.sqrt((lat - alat) ** 2 + (lon - alon) ** 2)

                    if distance < closest_distance:
                        closest_distance = distance
                        closest_area = area

            if closest_area is not None:
                self.area = closest_area
                self.save()

    def closest_localities(self, distance_threshold=0.1):
        """Find closest localities."""
        localities = Locality.objects.all()

        if "," in str(self.location):
            lat, lon = str(self.location).split(",")
            lat = float(lat)
            lon = float(lon)

            closest_localities = []
            for locality in localities:
                if "," in str(locality.location):
                    alat, alon = str(locality.location).split(",")
                    alat = float(alat)
                    alon = float(alon)
                    distance = np.sqrt((lat - alat) ** 2 + (lon - alon) ** 2)

                    if distance < distance_threshold:
                        closest_localities.append(locality)

            return closest_localities
        return []


class Area(models.Model):
    name = models.CharField(max_length=50)
    visible_name = models.CharField(max_length=255, blank=True, default=human_readable_hash)
    location = PlainLocationField(
        based_fields=["city"],
        zoom=5,
        null=True,
        blank=True,
    )
    hash = models.CharField(max_length=50, default=get_hash8)
    # If the user is deleted, then we will keep the locality but it does not
    # belong to any user which is not good.
    # owner = models.ForeignKey(CaIDUser, on_delete=models.SET_NULL, null=True, blank=True)
    note = models.TextField(blank=True, default="")

    def __str__(self):
        return str(self.name)

    # on location change, update all localities
    def save(self, *args, **kwargs):
        """Save area and update localities."""
        super(Area, self).save(*args, **kwargs)
        localities = Locality.objects.all()
        for locality in localities:
            locality.set_closest_area()

    # def mediafiles(self):
    #     """Return mediafiles."""
    #     return MediaFile.objects.filter(locality=self).all()
    #
    # def identities(self):
    #     """Return identities."""
    #     return IndividualIdentity.objects.filter(mediafile__locality=self).all()


class UploadedArchive(models.Model):
    name = models.CharField(max_length=255, blank=True, default="")
    uploaded_at = models.DateTimeField("Uploaded at", default=datetime.now)
    archivefile = models.FileField(
        "Archive File",
        upload_to=get_zip_path_in_unique_folder,
        max_length=500,
    )
    preview = models.ImageField(blank=True, null=True)
    outputdir = models.CharField(max_length=255, blank=True, default=get_output_dir)
    thumbnail = models.ImageField(upload_to=outputdir, blank=True)
    zip_file = models.FileField(upload_to=outputdir, blank=True, null=True)
    csv_file = models.FileField(upload_to=outputdir, blank=True, null=True)
    output_updated_at = models.DateTimeField("Output updated at", blank=True, null=True)
    hash = models.CharField(max_length=255, blank=True, default=get_hash)
    taxon_status = models.CharField(
        max_length=255,
        blank=True,
        choices=UA_STATUS_CHOICES,
        default="C",
    )
    status_message = models.TextField(blank=True)
    started_at = models.DateTimeField("Started at", blank=True, null=True)
    finished_at = models.DateTimeField("Finished at", blank=True, null=True)
    identification_status = models.CharField(
        max_length=255,
        blank=True,
        choices=UA_STATUS_CHOICES,
        default="C",
    )
    identification_started_at = models.DateTimeField("Started at", blank=True, null=True)
    identification_finished_at = models.DateTimeField("Finished at", blank=True, null=True)
    locality_at_upload = models.CharField(max_length=255, blank=True, default="")
    locality_at_upload_object = models.ForeignKey(Locality, on_delete=models.SET_NULL, null=True, blank=True)
    owner = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    contains_identities = models.BooleanField(default=False)
    contains_single_taxon = models.BooleanField(default=False)
    taxon_for_identification_at_upload = models.CharField(max_length=255, blank=True, default="")
    taxon_for_identification = models.ForeignKey(Taxon, on_delete=models.SET_NULL, null=True, blank=True)
    mediafiles_imported = models.BooleanField("Media Files Imported Correctly", default=False)
    earliest_captured_at = models.DateTimeField("Earliest Captured at", blank=True, null=True)
    latest_captured_at = models.DateTimeField("Latest Captured at", blank=True, null=True)
    locality_check_at = models.DateTimeField("Locality Check at", blank=True, null=True)
    mediafiles_at_upload = models.IntegerField("Media Files at Upload", default=0)
    images_at_upload = models.IntegerField("Images at Upload", default=0)
    videos_at_upload = models.IntegerField("Videos at Upload", default=0)
    files_at_upload = models.IntegerField("Files at Upload", default=0)
    import_error_spreadsheet = models.FileField(upload_to=outputdir, blank=True, null=True)

    def refresh_status_after_migration(self, request: Optional[object] = None):
        """Refresh possible old setup of object to 'migrated' one."""
        # couples [[old_status, new_status], ...]

        new_old_status = [
            ["Created", "C"],
            ["Failed", "F"],
            ["Taxon processing", "TAIP"],
            ["Taxon AI done", "TAID"],
            ["Taxon classification finished", "TAID"],
            ["TAAI", "TAID"],
            ["Taxons classified", "TAID"],
            ["Taxa known", "TKN"],
            ["Taxons done", "TKN"],
            ["Taxa verified", "TV"],
            ["Taxons verified", "TV"],
            ["ID processing", "IAIP"],
            ["ID AI done", "IAID"],
            ["Identification finished", "IAID"],
        ]

        applied = False
        for old_status, new_status in new_old_status:
            if self.taxon_status == old_status:
                self.taxon_status = new_status
                self.save()
                applied = True
        if not applied:
            logger.debug(f"Status {self.taxon_status} not found in refresh.")
            if request:
                messages.debug(request, f"Status {self.taxon_status} not found in refresh.")

    def next_processing_step_structure(self) -> Optional[tuple]:
        """Return suggestion for next processing step in structure."""
        if self.taxon_status == "C":
            return None
        elif self.taxon_status == "F":
            return None
        elif self.taxon_status == "TAIP":
            return None
        elif self.taxon_status == "TAID":
            return "Annotate taxa", reverse_lazy(
                "caidapp:missing_taxon_annotation", kwargs={"uploaded_archive_id": self.id}
            )
        elif self.taxon_status == "TKN":
            return "Verify taxa", reverse_lazy("caidapp:verify_taxa", kwargs={"uploaded_archive_id": self.id})
        elif (self.taxon_status == "TV") and (self.owner.workgroup is not None):
            return "Go to identification", reverse_lazy(
                "caidapp:select_taxon_for_identification", kwargs={"uploadedarchive_id": self.id}
            )
        else:
            return None

    def extract_locality_check_at_from_filename(self, commit=True):
        """Extract location check at from filename."""
        logger.debug(f"{self.locality_check_at=}")
        if self.locality_check_at is None:
            archive_name = Path(self.archivefile.name).stem
            logger.debug(f"{archive_name=}")
            # find date in archive_name {YYYY-MM-DD}
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", archive_name)
            logger.debug(f"{date_match=}")

            if date_match:
                date_match = date_match.group()
                logger.debug(f"{date_match=}")
                self.locality_check_at = datetime.strptime(date_match, "%Y-%m-%d")
                if commit:
                    self.save()

    def count_of_mediafiles(self):
        """Return number of mediafiles in the archive."""
        if self.taxon_status in ("C", "TAIP", "TAID"):
            return self.mediafiles_at_upload
        return MediaFile.objects.filter(parent=self).count()

    def count_of_representative_mediafiles(self):
        """Return number of representative mediafiles in the archive."""
        return MediaFile.objects.filter(parent=self, identity_is_representative=True).count()

    def count_of_mediafiles_with_taxon_for_identification(self):
        """Return number of mediafiles with taxon for identification in the archive."""
        if self.taxon_for_identification is None:
            return None
        else:
            return MediaFile.objects.filter(parent=self, taxon=self.taxon_for_identification).count()

    def update_location_in_mediafiles(self, location: Union[str, Locality]):
        """Update location in mediafiles."""
        if isinstance(location, str):
            location = get_locality(self.owner, location)
        mediafiles = MediaFile.objects.filter(parent=self)
        for mediafile in mediafiles:
            mediafile.locality = location
        self.locality_at_upload_object = location
        self.location = location.name

    def earliest_captured_taxon(self):
        """Return earliest captured taxon in the archive."""
        return MediaFile.objects.filter(parent=self).order_by("captured_at").first().taxon

    def latest_captured_taxon(self):
        """Return latest captured taxon in the archive."""
        return MediaFile.objects.filter(parent=self).order_by("-captured_at").first().taxon

    def update_earliest_and_latest_captured_at(self):
        """Update the earliest and latest captured at in the archive based on media files."""
        mediafiles = MediaFile.objects.filter(parent=self)
        earliest_captured_at = None
        latest_captured_at = None
        for mediafile in mediafiles:
            if mediafile.captured_at is not None:
                if earliest_captured_at is None or mediafile.captured_at < earliest_captured_at:
                    earliest_captured_at = mediafile.captured_at
                if latest_captured_at is None or mediafile.captured_at > latest_captured_at:
                    latest_captured_at = mediafile.captured_at
        # logger.debug(f"{earliest_captured_at=}, {latest_captured_at=}")
        self.earliest_captured_at = earliest_captured_at
        self.latest_captured_at = latest_captured_at
        self.save()

    def get_name(self):
        """Return name of the archive."""
        return str(Path(self.archivefile.name).name)

    def __str__(self):
        if self.name:
            return str(self.name)
        else:
            self.name = Path(self.archivefile.name).stem
            return str(self.name)

    def taxons_are_verified(self):
        """Return True if all taxons are verified."""
        return self.mediafile_set.filter(taxon_verified=False).count() == 0

    def mediafiles_with_missing_taxon(self, **kwargs):
        """Return media files with missing taxon."""
        return get_mediafiles_with_missing_taxon(self.owner, uploadedarchive=self, **kwargs)

    def count_of_mediafiles_with_missing_taxon(self):
        """Return number of media files with missing taxon."""
        return self.mediafiles_with_missing_taxon().count()

    def count_of_mediafiles_with_taxon(self):
        """Return number of media files with taxon."""
        return self.count_of_mediafiles() - self.count_of_mediafiles_with_missing_taxon()

    def percents_of_mediafiles_with_taxon(self) -> float:
        """Return percents of media files with taxon."""
        if self.count_of_mediafiles() == 0:
            return 0
        return (
            100
            * (self.count_of_mediafiles() - self.count_of_mediafiles_with_missing_taxon())
            / self.count_of_mediafiles()
        )

    def count_of_mediafiles_with_verified_taxon(self):
        """Return number of media files with verified taxon."""
        return self.mediafile_set.filter(taxon_verified=True).count()

    def count_of_mediafiles_with_unverified_taxon(self):
        """Return number of media files with unverified taxon."""
        return self.mediafile_set.filter(taxon_verified=False).count()

    def percents_of_mediafiles_with_verified_taxon(self) -> float:
        """Return percents of media files with verified taxon."""
        if self.count_of_mediafiles() == 0:
            return 0
        return 100 * (self.mediafile_set.filter(taxon_verified=True).count()) / self.count_of_mediafiles()

    def has_all_taxons(self):
        """Return True if all media files have taxon."""
        return self.count_of_mediafiles_with_missing_taxon() == 0

    def count_of_identities(self):
        """Return number of unique identities in the archive."""
        return self.mediafile_set.filter(Q(identity__isnull=False)).values("identity").distinct().count()

    def count_of_taxons(self):
        """Return number of unique taxons in the archive."""
        not_classified_taxon = Taxon.objects.get(name=TAXON_NOT_CLASSIFIED)
        return (
            self.mediafile_set.filter(Q(taxon=None) | Q(taxon=not_classified_taxon)).values("taxon").distinct().count()
        )

    def number_of_media_files_in_archive(self) -> dict:
        """Return number of media files in the archive and update UploadedArchive."""
        counts = fs_data.count_files_in_archive(self.archivefile.path)
        if self.files_at_upload != counts["file_count"]:
            self.images_at_upload = counts["image_count"]
            self.videos_at_upload = counts["video_count"]
            self.files_at_upload = counts["file_count"]
            self.mediafiles_at_upload = self.images_at_upload + self.videos_at_upload
            self.save()
        return counts

    def update_status(self):
        """Update status with respect to manual annotations."""
        status = self.taxon_status

        if status in ("TV", "TKN", "TAID"):
            n_missing_taxon = self.count_of_mediafiles_with_missing_taxon()
            n_unverified_taxon = self.count_of_mediafiles_with_unverified_taxon()
            if n_missing_taxon == 0:
                if n_unverified_taxon == 0:
                    status = "TV"
                else:
                    status = "TKN"
            else:
                status = "TAID"

        if status in UA_STATUS_CHOICES_DICT:
            if status != self.taxon_status:
                logger.debug(f"Status of {self} is changed: {self.taxon_status} -> {status}")
                self.taxon_status = status
                self.save()
        else:
            status_message = f"Unknown status '{status}'. Prev. message: " + str(self.status_message)
            logger.warning(f"Status of {self} is unknown: {status}, status_message: {self.status_message}")
            status = "U"
            self.status_message = status_message
            self.taxon_status = status
            self.save()

    def get_identification_status(self) -> dict:
        """Return short status message, long message and color-style for the status."""
        # find 'F' in self.STATUS_CHOICES[]
        status = self.identification_status
        status_message = self.status_message
        status_style = "dark"
        if self.taxon_status == "TAID":  # "Taxons classified":
            status_style = "dark"
        elif self.taxon_status == "F":
            status_style = "danger"
        elif self.taxon_status == "TKN":
            status_message = "All media files have taxon."
            status_style = "secondary"
        if self.taxon_status == "TV":
            status_message = "All taxons are verified."
            status_style = "secondary"

        if status == "F":
            status_style = "danger"

        elif status == "U":
            status_style = "warning"
        elif status == "IAIP":
            status_style = "secondary"
        elif status == "IAID":
            status_style = "primary"

        status = UA_STATUS_CHOICES_DICT.get(status, "Unknown")

        return dict(
            status=status,
            status_message=status_message,
            status_style=status_style,
        )

    def get_status(self) -> dict:
        """Return short status message, long message and color-style for the status."""
        # find 'F' in self.STATUS_CHOICES[]
        status = self.taxon_status
        status_message = self.status_message
        status_style = "dark"
        if self.taxon_status == "TAID":  # "Taxons classified":
            status_style = "secondary"
        elif self.taxon_status == "F":
            status_style = "danger"

        if status == "TKN":
            status_message = "All media files have taxon."
            status_style = "primary"
        if self.taxon_status == "TV":
            status_message = "All taxons are verified."
            status_style = "success"
        if self.taxon_status == "U":
            status_style = "warning"

        status = UA_STATUS_CHOICES_DICT.get(status, "Unknown")

        return dict(
            status=status,
            status_message=status_message,
            status_style=status_style,
        )

    def get_sequence_by_id(self, sequence_id: Union[int, None]) -> "Sequence":
        """Return sequence by id or create a new one."""
        if sequence_id is None:
            sequence = None
            sequence_id = self.sequence_set.count()
        else:
            sequence = Sequence.objects.filter(uploaded_archive=self, local_id=sequence_id).first()
        if sequence is None:
            sequence = Sequence.objects.create(uploaded_archive=self, local_id=sequence_id)
            sequence.save()
            # logger.debug("Sequence created.")

        return sequence

    def make_sequences(self):
        """Create sequences from media files."""
        mediafiles = MediaFile.objects.filter(parent=self)
        for mediafile in tqdm(mediafiles, desc="Creating sequences"):
            try:
                sequence_id = mediafile.metadata_json.get("sequence_number", None)
            except Exception as e:
                logger.debug(f"Problem with getting sequence_number from mediafile {mediafile}.")
                logger.debug(f"Error: {e}")
                sequence_id = None
            sequence = self.get_sequence_by_id(sequence_id)
            mediafile.sequence = sequence
            mediafile.save()

    @property
    def metadata_csv_url(self):
        """Return URL to metadata CSV file."""
        # relativní cesta vzhledem k MEDIA_ROOT
        rel_path = (Path(self.outputdir) / "metadata.csv").relative_to(settings.MEDIA_ROOT)
        return f"{settings.MEDIA_URL}{rel_path}".replace("\\", "/")


class IndividualIdentity(models.Model):
    SEX_CHOICES = (
        ("M", "Male"),
        ("F", "Female"),
        ("U", "Unknown"),
    )
    COAT_TYPE_CHOICES = (
        ("S", "Spotted"),
        ("M", "Marble"),
        ("N", "Unspotted"),
        ("U", "Unknown"),
    )
    name = models.CharField(max_length=100)
    id_worker = models.IntegerField(null=True, blank=True)
    owner_workgroup = models.ForeignKey(WorkGroup, on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    sex = models.CharField(max_length=2, choices=SEX_CHOICES, default="U")
    coat_type = models.CharField(max_length=2, choices=COAT_TYPE_CHOICES, default="U")
    note = models.TextField(blank=True)
    code = models.CharField(max_length=50, blank=True, null=True)
    juv_code = models.CharField("Juv. Code", max_length=50, null=True, blank=True)
    hash = models.CharField(max_length=50, blank=True)
    birth_date = models.DateField("Birth date", blank=True, null=True)
    death_date = models.DateField("Death date", blank=True, null=True)

    # remove mediefiles and use mediafile_set
    def mediafiles(self):
        """Return mediafiles."""
        return MediaFile.objects.filter(identity=self).all()

    def last_seen(self):
        """Return last seen date."""
        last = MediaFile.objects.filter(identity=self).order_by("-captured_at").first()
        return last.captured_at if last else None

    def count_of_representative_mediafiles(self):
        """Return number of representative media files."""
        return MediaFile.objects.filter(identity=self, identity_is_representative=True).count()

    def count_of_mediafiles(self):
        """Return number of media files."""
        return MediaFile.objects.filter(identity=self).count()

    def count_of_localities(self):
        """Return number of localities."""
        return MediaFile.objects.filter(identity=self).values("locality").distinct().count()

    def localities(self):
        """Return localities ordered by count of media files."""
        # return MediaFile.objects.filter(identity=self).values("locality").distinct().or
        return Locality.objects.filter(mediafiles__identity=self).annotate(count=Count("mediafiles")).order_by("-count")

    def __str__(self):
        return str(self.name)

    def save(self, *args, **kwargs):
        """Save object."""
        if not self.hash:
            self.hash = get_hash8()
        super().save(*args, **kwargs)

    def get_sex_display(self):
        """Return human readable sex."""
        return dict(self.SEX_CHOICES).get(self.sex, "Unknown")

    def get_coat_type_display(self):
        """Return human readable coat type."""
        return dict(self.COAT_TYPE_CHOICES).get(self.coat_type, "Unknown")

    def suggested_code_from_name(self):
        """Find code in identity name.

        If the name contains B character fallowed by a number, then the number is used as code.
        """
        # looking for B{number}
        code = None
        code_match = re.search(r"B\d+", self.name)
        if code_match:
            code = code_match.group()
            # new_name = self.name.replace(code, "")
        return code

    def suggested_name_without_code(self):
        """Return name without code."""
        code = self.suggested_code_from_name()
        name = None
        if code:
            name = self.name.replace(code, "")
        return name


class Sequence(models.Model):
    uploaded_archive = models.ForeignKey(UploadedArchive, on_delete=models.CASCADE, null=True)
    local_id = models.IntegerField(null=True, blank=True)


class MediaFile(models.Model):
    # ORIENTATION_CHOICES = (
    #     ("L", "Left"),
    #     ("R", "Right"),
    #     ("F", "Front"),
    #     ("B", "Back"),
    #     ("N", "None"),
    #     ("U", "Unknown"),
    # )
    MEDIA_TYPE_CHOICES = (
        ("image", "Image"),
        ("video", "Video"),
    )
    parent = models.ForeignKey(UploadedArchive, on_delete=models.CASCADE, null=True)
    taxon = models.ForeignKey(Taxon, blank=True, null=True, on_delete=models.CASCADE, verbose_name="Taxon")
    # legacy_taxon = taxon
    predicted_taxon = models.ForeignKey(
        Taxon, blank=True, null=True, on_delete=models.SET_NULL, related_name="predicted_taxon"
    )
    predicted_taxon_confidence = models.FloatField(null=True, blank=True)
    locality = models.ForeignKey(
        Locality,
        blank=True,
        null=True,
        on_delete=models.CASCADE,
        related_name="mediafiles",
    )
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
    static_thumbnail = models.ImageField(blank=True, null=True, max_length=500)
    preview = models.ImageField(blank=True, null=True, max_length=500)  # 1200 x 800 px preview
    identity = models.ForeignKey(IndividualIdentity, blank=True, null=True, on_delete=models.SET_NULL)
    identity_is_representative = models.BooleanField(default=False)
    updated_by = models.ForeignKey(CaIDUser, on_delete=models.SET_NULL, null=True, blank=True)
    updated_at = models.DateTimeField("Updated at", blank=True, null=True)
    metadata_json = models.JSONField(blank=True, null=True)
    animal_number = models.IntegerField(null=True, blank=True)
    media_type = models.CharField(
        max_length=255,
        blank=True,
        default="image",
        choices=MEDIA_TYPE_CHOICES,
    )
    orientation = models.CharField(max_length=2, choices=ORIENTATION_CHOICES, default="N")
    original_filename = models.CharField(max_length=512, blank=True, default="")

    taxon_verified = models.BooleanField("Taxon verified", default=False)
    taxon_verified_at = models.DateTimeField("Taxon verified at", blank=True, null=True)
    sequence = models.ForeignKey(Sequence, on_delete=models.SET_NULL, null=True, blank=True)
    note = models.TextField(blank=True, default="")
    media_file_corrupted = models.BooleanField("Media file corrupted", default=False)
    used_for_init_identification = models.BooleanField("Used for init identification", default=False)

    class Meta:
        ordering = ["-identity_is_representative", "captured_at"]

    def __str__(self):
        return str(self.original_filename)

    def extract_original_filename(self, commit=True):
        """Extract original filename from metadata_json or mediafile."""
        if self.metadata_json:
            if "original_path" in self.metadata_json:
                self.original_filename = Path(self.metadata_json["original_path"]).name
                if commit:
                    self.save()
                return self.original_filename
        self.original_filename = Path(self.mediafile.name).name
        if commit:
            self.save()
        return self.original_filename

    def get_static_thumbnail(self, force: bool = True, width: int = 400) -> models.ImageField:
        """Return static thumbnail for mediafile."""
        logger.debug(f"Getting static thumbnail for {self.mediafile.name=}")

        if self.static_thumbnail:
            return self.static_thumbnail
        else:
            logger.debug(f"static_thumbnail Does not exist for {self.mediafile.name=}")
            return None

    def is_preidentified(self):
        """Return True if mediafile is preidentified."""
        return MediafilesForIdentification.objects.filter(mediafile=self).exists()

    @property
    def taxon_from_observations(self):
        """Return taxon from observations."""
        taxons = set()
        for obs in self.observations.all():
            if obs.taxon is not None:
                taxons.add(obs.taxon)
        if len(taxons) == 1:
            return taxons.pop()
        elif len(taxons) > 1:
            return get_taxon("Mixed")
        return None

    @property
    def is_verified_from_observations(self):
        """Return True if taxon is verified from observations."""
        for obs in self.observations.all():
            if obs.taxon_verified is False:
                return False
        return True

    def is_for_suggestion(self):
        """Return True if mediafile is for suggestion."""
        is_for_suggestion = False
        predicted_taxons = set()
        if self.predicted_taxon is not None:
            is_for_suggestion = True
            for obs in self.observations.all():
                if (
                    (obs.taxon is None)
                    or (obs.taxon.name == TAXON_NOT_CLASSIFIED)
                    or ((obs.taxon.name == "Animalia") and (obs.taxon_verified is False))
                ):
                    predicted_taxons.add(obs.taxon)
                else:
                    is_for_suggestion = False

        if len(predicted_taxons) != 1:
            is_for_suggestion = False

        return is_for_suggestion

        # if (self.predicted_taxon is not None):
        #     if self.first_observation is not None:
        #         if self.first_observation.taxon == self.first_observation.predicted_taxon:
        #             return True
        # return (self.predicted_taxon is not None) and (
        #     (self.taxon is None)
        #     or (self.taxon.name == TAXON_NOT_CLASSIFIED)
        #     or ((self.taxon.name == "Animalia") and (self.taxon_verified is False))
        # )

    def is_consistent_with_uploaded_archive_taxon_for_identification(self) -> bool:
        """Return True if mediafile is consistent with uploaded archive taxo for identification."""
        return self.taxon == self.parent.taxon_for_identification

    def make_thumbnail_for_mediafile_if_necessary(
        self,
        # mediafile: MediaFile,
        thumbnail_width: int = 400,
        preview_width: int = 1200,
        force=False,
    ):
        """Make small image representing the upload."""
        # logger.debug("Making thumbnail for mediafile")
        # mediafile = image or video
        # image_file = static image created from mediafile

        # preview: image or video in mp4 format with a bit smaller size than the original
        # thumbnail: small image with fixed size, or GIF (stored in webP)
        # static_preview: image with fixed size
        mediafile_path = Path(settings.MEDIA_ROOT) / self.mediafile.name
        if self.parent is None:
            logger.error(f"Mediafile {self.id} has no parent.")
            return
        output_dir = Path(settings.MEDIA_ROOT) / self.parent.outputdir
        preview_abs_pth = output_dir / "previews" / Path(self.mediafile.name).name
        thumbnail_abs_pth = output_dir / "thumbnails" / Path(self.mediafile.name).name
        thumbnail_abs_pth = thumbnail_abs_pth.with_suffix(".webp")
        static_thumbnail_abs_pth = output_dir / "static_thumbnails" / Path(self.mediafile.name).name
        static_thumbnail_abs_pth = static_thumbnail_abs_pth.with_suffix(".webp")

        if self.media_type == "video":
            preview_abs_pth = preview_abs_pth.with_suffix(".mp4")
            # thumbnail_abs_pth = thumbnail_abs_pth.with_suffix(".gif")
        else:
            preview_abs_pth = preview_abs_pth.with_suffix(".webp")

        preview_rel_pth = os.path.relpath(preview_abs_pth, settings.MEDIA_ROOT)
        thumbnail_rel_pth = os.path.relpath(thumbnail_abs_pth, settings.MEDIA_ROOT)
        static_thumbnail_rel_pth = os.path.relpath(thumbnail_abs_pth, settings.MEDIA_ROOT)

        if (not self.preview) or (not self.preview.name) or (not preview_abs_pth.exists()) or force:
            if not force:
                logger.debug(f"preview does not exist for {self.mediafile.name=}")
            preview_abs_pth.parent.mkdir(exist_ok=True, parents=True)
            # logger.debug(f"Creating preview for {preview_rel_pth}")
            if self.media_type == "video":
                convert_to_mp4(mediafile_path, preview_abs_pth)
            else:
                fs_data.make_thumbnail_from_file(mediafile_path, preview_abs_pth, width=preview_width)
            self.preview = str(preview_rel_pth)
            self.save()

        if (not self.thumbnail) or (not self.thumbnail.name) or (not thumbnail_abs_pth.exists()) or force:
            if not force:
                logger.debug(f"thumbnail does not exist for {self.mediafile.name=}")
            thumbnail_abs_pth.parent.mkdir(exist_ok=True, parents=True)
            if self.media_type == "video":
                fs_data.make_gif_from_video_file(mediafile_path, thumbnail_abs_pth, width=thumbnail_width)
            else:
                fs_data.make_thumbnail_from_file(mediafile_path, thumbnail_abs_pth, width=thumbnail_width)
            self.thumbnail = str(thumbnail_rel_pth)
            self.save()

        if (
            (not self.static_thumbnail)
            or (not self.static_thumbnail.name)
            or (not static_thumbnail_abs_pth.exists())
            or force
        ):
            if not force:
                logger.debug(f"static_thumbnail does not exist for {self.mediafile.name=}")
            fs_data.make_thumbnail_from_file(mediafile_path, static_thumbnail_abs_pth, width=thumbnail_width)
            self.static_thumbnail = str(static_thumbnail_rel_pth)
            self.save()
            # self.get_static_thumbnail(force=force)

    def save(self, *args, **kwargs):
        """Save object."""
        if self.pk:
            old = MediaFile.objects.get(pk=self.pk)
            if old.identity_is_representative != self.identity_is_representative:
                from .tasks import schedule_init_identification_for_workgroup

                schedule_init_identification_for_workgroup(self.parent.owner.workgroup, delay_minutes=40)
                # and the reid will be started after the init

        super().save(*args, **kwargs)

    @property
    def first_observation(self) -> Optional["AnimalObservation"]:
        """Return the first AnimalObservation related to this MediaFile."""
        obs = self.observations.first()
        if obs:
            return obs
        return None

    @property
    def first_observation_get_or_create(self) -> "AnimalObservation":
        """Return the first AnimalObservation related to this MediaFile, or create one if none exists."""
        obs, created = AnimalObservation.objects.get_or_create(mediafile=self)
        return obs

    def mediafile_variant_url(self, variant: str = "images") -> str:
        """Get mediafile variant URL."""
        # relative path to MEDIA_ROOT
        rel_path = Path(self.parent.outputdir).relative_to(settings.MEDIA_ROOT)
        rel_path = rel_path / variant / Path(self.mediafile.name).name
        return f"{settings.MEDIA_URL}{rel_path}".replace("\\", "/")

    @property
    def detection_url(self):
        """Get detection image URL."""
        return self.mediafile_variant_url("detection_images")

    @property
    def thumbnail_url(self):
        """Get thumbnail URL."""
        return self.mediafile_variant_url("thumbnails")

    @property
    def static_thumbnail_url(self):
        """Get static thumbnail URL."""
        return self.mediafile_variant_url("static_thumbnails")

    @property
    def preview_url(self):
        """Get preview URL."""
        return self.mediafile_variant_url("previews")


class AnimalObservation(models.Model):
    mediafile = models.ForeignKey(
        MediaFile,
        on_delete=models.CASCADE,
        related_name="observations",
        # null=True, blank=True
    )
    taxon = models.ForeignKey(Taxon, on_delete=models.SET_NULL, null=True, blank=True)
    taxon_verified = models.BooleanField("Taxon verified", default=False)
    taxon_verified_at = models.DateTimeField("Taxon verified at", blank=True, null=True)
    predicted_taxon = models.ForeignKey(
        Taxon,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="predicted_observations",
    )
    predicted_taxon_confidence = models.FloatField(null=True, blank=True)

    identity = models.ForeignKey(IndividualIdentity, on_delete=models.SET_NULL, null=True, blank=True)
    identity_is_representative = models.BooleanField(default=False)

    bbox_x_center = models.FloatField(null=True, blank=True)
    bbox_y_center = models.FloatField(null=True, blank=True)
    bbox_width = models.FloatField(null=True, blank=True)
    bbox_height = models.FloatField(null=True, blank=True)

    orientation = models.CharField(max_length=2, choices=ORIENTATION_CHOICES, default="N")

    updated_by = models.ForeignKey(CaIDUser, on_delete=models.SET_NULL, null=True, blank=True)
    updated_at = models.DateTimeField("Updated at", blank=True, null=True)
    metadata_json = models.JSONField(blank=True, null=True)

    def set_bbox_from_xyxy(self, x_min, y_min, x_max, y_max, img_w, img_h):
        """YOLO-style relative bbox from absolute in px."""
        self.bbox_x_center = ((x_min + x_max) / 2) / img_w
        self.bbox_y_center = ((y_min + y_max) / 2) / img_h
        self.bbox_width = (x_max - x_min) / img_w
        self.bbox_height = (y_max - y_min) / img_h

    def get_bbox_xyxy(self, img_w, img_h):
        """Get absolute pixel coordinates (x_min, y_min, x_max, y_max)."""
        x_min = int((self.bbox_x_center - self.bbox_width / 2) * img_w)
        y_min = int((self.bbox_y_center - self.bbox_height / 2) * img_h)
        x_max = int((self.bbox_x_center + self.bbox_width / 2) * img_w)
        y_max = int((self.bbox_y_center + self.bbox_height / 2) * img_h)
        return x_min, y_min, x_max, y_max

    @property
    def bbox_xywh_percent_as_str(self) -> Optional[str]:
        """Get bbox according to W3C media-fragments specification.

        https://www.w3.org/TR/media-frags/
        """
        # print("Getting bbox as string")
        # logger.debug("Getting bbox as string")
        if self.bbox_x_center is None:
            # return "nic"
            return None
        x = self.bbox_x_center - self.bbox_width / 2
        y = self.bbox_y_center - self.bbox_height / 2
        logger.debug(f"{x=}, {y=}, {self.bbox_width=}, {self.bbox_height=}")

        # return f"xywh=percent:{x:.4f},{y:.4f},{self.bbox_width:.4f},{self.bbox_height:.4f}"
        return f"xywh=percent:{x*100:.4f},{y*100:.4f},{self.bbox_width*100:.4f},{self.bbox_height*100:.4f}"


class MediafilesForIdentification(models.Model):
    mediafile = models.ForeignKey(MediaFile, on_delete=models.SET_NULL, null=True, blank=True)

    top1mediafile = models.ForeignKey(MediaFile, related_name="top1", on_delete=models.SET_NULL, null=True, blank=True)
    top2mediafile = models.ForeignKey(MediaFile, related_name="top2", on_delete=models.SET_NULL, null=True, blank=True)
    top3mediafile = models.ForeignKey(MediaFile, related_name="top3", on_delete=models.SET_NULL, null=True, blank=True)
    top1score = models.FloatField(null=True, blank=True)
    top2score = models.FloatField(null=True, blank=True)
    top3score = models.FloatField(null=True, blank=True)
    top1name = models.CharField(max_length=255, blank=True, default="")
    top2name = models.CharField(max_length=255, blank=True, default="")
    top3name = models.CharField(max_length=255, blank=True, default="")
    paired_points = models.JSONField(blank=True, null=True)


class MediafileIdentificationSuggestion(models.Model):
    for_identification = models.ForeignKey(
        MediafilesForIdentification, related_name="top_mediafiles", on_delete=models.CASCADE
    )
    mediafile = models.ForeignKey(MediaFile, on_delete=models.SET_NULL, null=True, blank=True)
    video_sequence_frame_super_id = models.IntegerField(default=0)
    score = models.FloatField(null=True, blank=True)
    name = models.CharField(max_length=255, blank=True, default="")
    paired_points = models.JSONField(blank=True, null=True)
    identity = models.ForeignKey(IndividualIdentity, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ["-score"]  # Order by score descending by default


class Album(models.Model):
    name = models.CharField(max_length=50)
    description = models.CharField(max_length=255, blank=True, default="")
    owner = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    mediafiles = models.ManyToManyField(MediaFile, blank=True)
    created_at = models.DateTimeField("Created at", default=datetime.now)
    hash = models.CharField(max_length=255, blank=True, default=get_hash)
    public_hash = models.CharField(max_length=255, blank=True, default=get_hash)
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
    hash = models.CharField(max_length=255, blank=True, default=get_hash)
    starts_at = models.DateTimeField("Starts at", blank=True, null=True)
    ends_at = models.DateTimeField("Ends at", blank=True, null=True)

    def __str__(self):
        return str(self.name)


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


class CaptureDevice(models.Model):
    """Capture device model."""

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, default="")
    owner = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)

    # JSON pole pro rozpoznávací znaky
    exif_features = models.JSONField(
        blank=True,
        default=dict,
        help_text="Dictionary of EXIF keys and expected values for identifying this device.",
    )

    def __str__(self):
        return self.name


def get_unique_name(name: str, workgroup: WorkGroup) -> Optional[IndividualIdentity]:
    """Return taxon according to the name, create it if necessary."""
    if (name is None) or (name == ""):
        return None
    # make sure name is string
    name = str(name)
    if len(name) > 100:
        name = name[:100]
    objs = IndividualIdentity.objects.filter(name=name, owner_workgroup=workgroup)
    if len(objs) == 0:
        identity = IndividualIdentity(name=name, owner_workgroup=workgroup)
        identity.save()
    else:
        identity = objs[0]
    return identity


def get_unique_code(code: str, workgroup: WorkGroup) -> Optional[IndividualIdentity]:
    """Return taxon according to the code, create it if necessary."""
    if (code is None) or (code == ""):
        return None
    # make sure code is string
    code = str(code)
    if len(code) > 50:
        code = code[:50]
    objs = IndividualIdentity.objects.filter(code=code, owner_workgroup=workgroup)
    if len(objs) == 0:
        identity = IndividualIdentity(code=code, owner_workgroup=workgroup)
        identity.save()
    else:
        identity = objs[0]
    return identity


def get_locality(caiduser: CaIDUser, name: str) -> Union[Locality, None]:
    """Return location according to the name, create it if necessary."""
    if (name is None) or (name == ""):
        return None

    objs = Locality.objects.filter(name=name, **user_has_access_filter_params(caiduser, "owner"))
    if len(objs) == 0:
        location = Locality(name=name, owner=caiduser)
        location.save()
    else:
        location = objs[0]
    return location


@receiver(models.signals.post_delete, sender=UploadedArchive)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    """Deletes file from filesystem when corresponding `MediaFile` object is deleted."""
    if instance.archivefile and os.path.isfile(instance.archivefile.path):
        os.remove(instance.archivefile.path)
    if instance.outputdir and os.path.isdir(instance.outputdir):
        shutil.rmtree(instance.outputdir)


# def get_mediafiles_with_missing_taxon(
#     caiduser: CaIDUser, uploadedarchive: Optional[UploadedArchive] = None, **kwargs
# ) -> QuerySet:
#     """Return media files with missing taxon."""
#     not_classified_taxon = get_taxon(TAXON_NOT_CLASSIFIED)
#     animalia_taxon = get_taxon("Animalia")
#
#     kwargs_filter = user_has_access_filter_params(caiduser, "parent__owner")
#     if uploadedarchive is not None:
#         kwargs["parent"] = uploadedarchive
#
#     mediafiles = MediaFile.objects.filter(
#         Q(taxon=None)
#         | Q(taxon=not_classified_taxon)
#         | (Q(taxon=animalia_taxon) & Q(taxon_verified=False)),
#         **kwargs_filter,
#         parent__contains_single_taxon=False,
#         **kwargs,
#     ).select_related(
#         "parent",
#         "taxon",
#         "predicted_taxon",
#         "locality",
#         "identity",
#         "updated_by",
#         "sequence"
#     )
#     return mediafiles


from django.db.models import Exists, OuterRef


def get_mediafiles_with_missing_taxon(
    caiduser: CaIDUser, uploadedarchive: Optional[UploadedArchive] = None, **kwargs
) -> QuerySet:
    """Return MediaFiles whose observations have missing or unverified taxon."""
    not_classified_taxon = get_taxon(TAXON_NOT_CLASSIFIED)
    animalia_taxon = get_taxon("Animalia")

    # přístupové omezení
    kwargs_filter = user_has_access_filter_params(caiduser, "parent__owner")

    if uploadedarchive is not None:
        kwargs["parent"] = uploadedarchive

    # poddotaz: existuje nějaká observation s platným taxonem?
    valid_obs = AnimalObservation.objects.filter(
        mediafile=OuterRef("pk"),
    ).filter(
        ~Q(taxon=None),
        ~Q(taxon=not_classified_taxon),
        ~(Q(taxon=animalia_taxon) & Q(taxon_verified=False)),
    )

    # vyber mediafiles, které žádnou validní observation nemají
    mediafiles = (
        MediaFile.objects.annotate(has_valid_obs=Exists(valid_obs))
        .filter(
            has_valid_obs=False,
            parent__contains_single_taxon=False,
            **kwargs_filter,
            **kwargs,
        )
        .select_related(
            "parent",
            "predicted_taxon",
            "locality",
            "identity",
            "updated_by",
            "sequence",
        )
    )

    return mediafiles


# def get_mediafiles_with_missing_verification(
#     caiduser: CaIDUser, uploadedarchive: Optional[UploadedArchive] = None, **kwargs
# ) -> QuerySet:
#     """Return media files with missing taxon verification."""
#     kwargs_filter = user_has_access_filter_params(caiduser, "parent__owner")
#     if uploadedarchive is not None:
#         kwargs["parent"] = uploadedarchive
#
#     logger.debug(f"{caiduser=}, {uploadedarchive=}, {kwargs=}, {kwargs_filter=}")
#     mediafiles = MediaFile.objects.filter(
#         taxon_verified=False, **kwargs_filter, parent__contains_single_taxon=False, **kwargs
#     ).select_related(
#         "parent",
#         "taxon",
#         "predicted_taxon",
#         "locality",
#         "identity",
#         "updated_by",
#         "sequence"
#     )
#
#     logger.debug(f"{mediafiles.count()=}")
#     return mediafiles


def get_mediafiles_with_missing_verification(
    caiduser: CaIDUser, uploadedarchive: Optional[UploadedArchive] = None, **kwargs
) -> QuerySet:
    """Return MediaFiles that have at least one observation without verified taxon."""
    kwargs_filter = user_has_access_filter_params(caiduser, "parent__owner")
    if uploadedarchive is not None:
        kwargs["parent"] = uploadedarchive

    logger.debug(f"{caiduser=}, {uploadedarchive=}, {kwargs=}, {kwargs_filter=}")

    # Poddotaz: existuje nějaká observace, která není verifikovaná?
    unverified_obs = AnimalObservation.objects.filter(mediafile=OuterRef("pk")).filter(
        Q(taxon_verified=False) | Q(taxon_verified__isnull=True)
    )

    # Vybereme MediaFiles, které mají alespoň jednu takovou observaci
    mediafiles = (
        MediaFile.objects.annotate(has_unverified_obs=Exists(unverified_obs))
        .filter(
            has_unverified_obs=True,
            parent__contains_single_taxon=False,
            **kwargs_filter,
            **kwargs,
        )
        .select_related(
            "parent",
            "predicted_taxon",
            "locality",
            "identity",
            "updated_by",
            "sequence",
        )
    )

    logger.debug(f"{mediafiles.count()=}")
    return mediafiles


def get_all_relevant_localities(request):
    """Get all users localities."""
    params = user_has_access_filter_params(request.user.caiduser, "owner")
    # logger.debug(f"{params=}")
    localities = (
        Locality.objects.filter(**params)
        # .annotate(mediafile_count=Count('uploadedarchive__mediafile'))
        .annotate(mediafile_count=Count("mediafiles")).order_by("name")
    )
    return localities


def user_has_access_filter_params(caiduser: CaIDUser, prefix: str) -> dict:
    """Parameters for filtering user content based on existence of workgroup.

    Parameters
    ----------
    caiduser: CaIDUser
        User who want to access the data.
    prefix : str
        Prefix for filtering with ciduser.
        If the filter will be used in MediaFile, the prefix should be "parent__owner".
        If the filter will be used in Location or UploadedArchive, the prefix should be "owner".
    """
    if caiduser.workgroup:
        filter_params = {f"{prefix}__workgroup": caiduser.workgroup}
    else:
        filter_params = {f"{prefix}": caiduser}
    return filter_params


class Notification(models.Model):
    """Notification model."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    LEVEL_CHOICES = [
        (DEBUG, "Debug"),
        (INFO, "Info"),
        (WARNING, "Warning"),
        (ERROR, "Error"),
        (CRITICAL, "Critical"),
    ]
    BOOTSTRAP_CLASSES = {
        DEBUG: "secondary",  # šedá
        INFO: "info",  # modrá
        WARNING: "warning",  # žlutá
        ERROR: "danger",  # červená
        CRITICAL: "dark",  # tmavá (nebo taky danger, podle vkusu)
    }

    user = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    message = models.TextField(blank=True, default="")
    json_message = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField("Created at", auto_now_add=True)
    read = models.BooleanField("Read", default=False)
    level = models.PositiveSmallIntegerField(choices=LEVEL_CHOICES, default=INFO)

    def __str__(self):
        return f"Notification for {self.user.user.username} at {self.created_at}"

    def bootstrap_class(self):
        """Return bootstrap class for the notification level."""
        return self.BOOTSTRAP_CLASSES.get(self.level, "secondary")


class MergeIdentitySuggestionResult(models.Model):
    workgroup = models.ForeignKey(WorkGroup, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    suggestions = models.JSONField()  # uloží [(id1, id2, distance), ...]

    def __str__(self):
        return f"Suggestions for {self.workgroup} at {self.created_at}"
