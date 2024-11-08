import logging
import os.path
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import codenamize
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.db import models
from django.db.models import Q
from django.db.models.query import QuerySet
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.urls import reverse_lazy
from location_field.models.plain import PlainLocationField

from . import fs_data
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
    ("TAIP", "Taxon processing"),
    ("TAID", "Missing taxa"),
    ("TKN", "Taxa known"),
    ("TV", "Taxa verified"),
    ("IR", "Ready for ID"),
    ("IAIP", "ID processing"),
    ("IAID", "ID AI done"),
    ("U", "Unknown"),
)
UA_STATUS_CHOICES_DICT = dict(UA_STATUS_CHOICES)


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


class WorkGroup(models.Model):
    name = models.CharField(max_length=50)
    hash = models.CharField(max_length=50, default=random_string12)
    identification_init_at = models.DateTimeField("Identification init at", blank=True, null=True)
    identification_init_status = models.CharField(
        max_length=255, blank=True, default="Not initiated"
    )
    identification_init_message = models.TextField(blank=True, default="")

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


class Taxon(models.Model):
    name = models.CharField(max_length=50)
    parent = models.ForeignKey("self", on_delete=models.SET_NULL, null=True, blank=True)

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
    hash = models.CharField(max_length=50, default=get_hash8)
    # If the user is deleted, then we will keep the location but it does not
    # belong to any user which is not good.
    owner = models.ForeignKey(CaIDUser, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return str(self.name)


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
    # status_message = models.CharField(max_length=2047, blank=True, default="")
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
    location_check_at = models.DateTimeField("Location Check at", blank=True, null=True)

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
            return "Verify taxa", reverse_lazy(
                "caidapp:verify_taxa", kwargs={"uploaded_archive_id": self.id}
            )
        else:
            return None

    def extract_location_check_at_from_filename(self, commit=True):
        """Extract location check at from filename."""
        logger.debug(f"{self.location_check_at=}")
        if self.location_check_at is None:
            archive_name = Path(self.archivefile.name).stem
            logger.debug(f"{archive_name=}")
            # find date in archive_name {YYYY-MM-DD}
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", archive_name)
            logger.debug(f"{date_match=}")

            if date_match:
                date_match = date_match.group()
                logger.debug(f"{date_match=}")
                self.location_check_at = datetime.strptime(date_match, "%Y-%m-%d")
                if commit:
                    self.save()

    def count_of_mediafiles(self):
        """Return number of mediafiles in the archive."""
        return MediaFile.objects.filter(parent=self).count()

    def count_of_representative_mediafiles(self):
        """Return number of representative mediafiles in the archive."""
        return MediaFile.objects.filter(parent=self, identity_is_representative=True).count()

    def count_of_mediafiles_with_taxon_for_identification(self):
        """Return number of mediafiles with taxon for identification in the archive."""
        if self.taxon_for_identification is None:
            return None
        else:
            return MediaFile.objects.filter(
                parent=self, category=self.taxon_for_identification
            ).count()

    def update_location_in_mediafiles(self, location: Union[str, Location]):
        """Update location in mediafiles."""
        if isinstance(location, str):
            location = get_location(self.owner, location)
        mediafiles = MediaFile.objects.filter(parent=self)
        for mediafile in mediafiles:
            mediafile.location = location
        self.location_at_upload_object = location
        self.location = location.name

    def earliest_captured_taxon(self):
        """Return earliest captured taxon in the archive."""
        return MediaFile.objects.filter(parent=self).order_by("captured_at").first().category

    def latest_captured_taxon(self):
        """Return latest captured taxon in the archive."""
        return MediaFile.objects.filter(parent=self).order_by("-captured_at").first().category

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
                logger.debug(f"{mediafile=}")
        logger.debug(f"{mediafiles.count()=}")
        logger.debug(f"{earliest_captured_at=}, {latest_captured_at=}")
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
        # not_classified_taxon = Taxon.objects.get(name="Not Classified")
        # animalia_taxon = Taxon.objects.get(name="Animalia")
        # return self.mediafile_set.filter(
        #     Q(category=None) | Q(category=not_classified_taxon) |
        #     (Q(category=animalia_taxon) & Q(taxon_verified=False)),
        #     **kwargs
        # )

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
        return (
            100
            * (self.mediafile_set.filter(taxon_verified=True).count())
            / self.count_of_mediafiles()
        )

    def has_all_taxons(self):
        """Return True if all media files have taxon."""
        return self.count_of_mediafiles_with_missing_taxon() == 0

    def count_of_identities(self):
        """Return number of unique identities in the archive."""
        return (
            self.mediafile_set.filter(Q(identity__isnull=False))
            .values("identity")
            .distinct()
            .count()
        )

    def count_of_taxons(self):
        """Return number of unique taxons in the archive."""
        not_classified_taxon = Taxon.objects.get(name="Not Classified")
        return (
            self.mediafile_set.filter(Q(category=None) | Q(category=not_classified_taxon))
            .values("category")
            .distinct()
            .count()
        )

    def number_of_media_files_in_archive(self) -> dict:
        """Return number of media files in the archive."""
        counts = fs_data.count_files_in_archive(self.archivefile.path)
        return counts

    def update_status(self):
        """Update status with respect to manual annotations."""
        status = self.taxon_status

        if status in ("TV", "TKN", "TAID"):
            if self.count_of_mediafiles_with_unverified_taxon() == 0:
                status = "TV"
            elif status == "TV":
                # someone unverified a file
                status = "TKN"  # it will be rechecked on next lines

            if self.count_of_mediafiles_with_missing_taxon() == 0:
                status = "TKN"
            elif status == "TKN":
                # some taxons were removed
                status = "TAID"

        if status in UA_STATUS_CHOICES_DICT:
            if status != self.taxon_status:
                logger.debug(f"Status of {self} is changed: {self.taxon_status} -> {status}")
                self.taxon_status = status
                self.save()
        else:
            status_message = f"Unknown status '{status}'. Prev. message: " + str(
                self.status_message
            )
            logger.warning(
                f"Status of {self} is unknown: {status}, status_message: {self.status_message}"
            )
            status = "U"
            self.status_message = status_message
            self.taxon_status = status
            self.save()

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
            logger.debug("Sequence created.")

        return sequence

    def make_sequences(self):
        """Create sequences from media files."""
        mediafiles = MediaFile.objects.filter(parent=self)
        for mediafile in mediafiles:
            try:
                sequence_id = mediafile.metadata_json.get("sequence_number", None)
            except Exception as e:
                logger.debug(f"Problem with getting sequence_number from mediafile {mediafile}.")
                logger.debug(f"Error: {e}")
                sequence_id = None
            sequence = self.get_sequence_by_id(sequence_id)
            mediafile.sequence = sequence
            mediafile.save()


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
    name = models.CharField(max_length=50)
    id_worker = models.IntegerField(null=True, blank=True)
    owner_workgroup = models.ForeignKey(WorkGroup, on_delete=models.CASCADE, null=True, blank=True)
    updated_by = models.ForeignKey(CaIDUser, on_delete=models.CASCADE, null=True, blank=True)
    sex = models.CharField(max_length=2, choices=SEX_CHOICES, default="U")
    coat_type = models.CharField(max_length=2, choices=COAT_TYPE_CHOICES, default="U")
    note = models.TextField(blank=True)
    code = models.CharField(max_length=50, default=random_string12)
    juv_code = models.CharField("Juv. Code", max_length=50, default=random_string12)
    hash = models.CharField(max_length=50, blank=True)

    def count_of_representative_mediafiles(self):
        """Return number of representative media files."""
        return MediaFile.objects.filter(identity=self, identity_is_representative=True).count()

    def count_of_mediafiles(self):
        """Return number of media files."""
        return MediaFile.objects.filter(identity=self).count()

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


class Sequence(models.Model):
    uploaded_archive = models.ForeignKey(UploadedArchive, on_delete=models.CASCADE, null=True)
    local_id = models.IntegerField(null=True, blank=True)


class MediaFile(models.Model):
    ORIENTATION_CHOICES = (
        ("L", "Left"),
        ("R", "Right"),
        ("N", "None"),
    )
    parent = models.ForeignKey(UploadedArchive, on_delete=models.CASCADE, null=True)
    category = models.ForeignKey(Taxon, blank=True, null=True, on_delete=models.CASCADE)
    predicted_taxon = models.ForeignKey(
        Taxon, blank=True, null=True, on_delete=models.SET_NULL, related_name="predicted_taxon"
    )
    predicted_taxon_confidence = models.FloatField(null=True, blank=True)
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
    preview = models.ImageField(blank=True, null=True, max_length=500)  # 1200 x 800 px preview
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

    taxon_verified = models.BooleanField("Taxon verified", default=False)
    taxon_verified_at = models.DateTimeField("Taxon verified at", blank=True, null=True)
    sequence = models.ForeignKey(Sequence, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ["-identity_is_representative", "captured_at"]

    def __str__(self):
        return str(self.original_filename)
        # return str(Path(self.mediafile.name).name)

    def extract_original_filename(self, commit=True):
        """Extract original filename from metadata_json or mediafile."""
        if self.metadata_json:
            if "vanilla_path" in self.metadata_json:
                self.original_filename = Path(self.metadata_json["vanilla_path"]).name
                if commit:
                    self.save()
                return self.original_filename
        self.original_filename = Path(self.mediafile.name).name
        if commit:
            self.save()
        return self.original_filename

    def is_preidentified(self):
        """Return True if mediafile is preidentified."""
        return MediafilesForIdentification.objects.filter(mediafile=self).exists()

    def is_for_suggestion(self):
        """Return True if mediafile is for suggestion."""
        return (self.predicted_taxon is not None) and (
            (self.category.name == "Not Classified")
            or ((self.category.name == "Animalia") and (self.taxon_verified is False))
        )


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
        If the filter will be used in Location or UploadedArchive, the prefix should be "owner".
    """
    if ciduser.workgroup:
        # filter_params = dict(parent__owner__workgroup=request.user.caiduser.workgroup)
        filter_params = {f"{prefix}__workgroup": ciduser.workgroup}
    else:
        filter_params = {f"{prefix}": ciduser}
    return filter_params


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


def get_mediafiles_with_missing_taxon(
    caiduser: CaIDUser, uploadedarchive: Optional[UploadedArchive] = None, **kwargs
) -> QuerySet:
    """Return media files with missing taxon."""
    not_classified_taxon = get_taxon("Not Classified")
    animalia_taxon = get_taxon("Animalia")

    kwargs_filter = get_content_owner_filter_params(caiduser, "parent__owner")
    if uploadedarchive is not None:
        kwargs["parent"] = uploadedarchive

    mediafiles = MediaFile.objects.filter(
        Q(category=None)
        | Q(category=not_classified_taxon)
        | (Q(category=animalia_taxon) & Q(taxon_verified=False)),
        **kwargs_filter,
        parent__contains_single_taxon=False,
        **kwargs,
    )
    logger.debug(f"{mediafiles.count()=}")
    return mediafiles


def get_mediafiles_with_missing_verification(
    caiduser: CaIDUser, uploadedarchive: Optional[UploadedArchive] = None, **kwargs
) -> QuerySet:
    """Return media files with missing taxon verification."""
    kwargs_filter = get_content_owner_filter_params(caiduser, "parent__owner")
    if uploadedarchive is not None:
        kwargs["parent"] = uploadedarchive

    logger.debug(f"{caiduser=}, {uploadedarchive=}, {kwargs=}, {kwargs_filter=}")
    mediafiles = MediaFile.objects.filter(
        taxon_verified=False, **kwargs_filter, parent__contains_single_taxon=False, **kwargs
    )
    logger.debug(f"{mediafiles.count()=}")
    return mediafiles
