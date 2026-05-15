import json
import os
import time
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management.base import BaseCommand
from django.test import Client
from django.urls import reverse

from caidapp.models import CaIDUser, Locality, UploadedArchive, WorkGroup


class NewUploadHealthcheckError(Exception):
    """Custom exception raised when the new upload healthcheck fails."""

    pass


class Command(BaseCommand):
    help = "End-to-end healthcheck of the new upload flow with real ZIP input."

    def add_arguments(self, parser):
        parser.add_argument("--username", default="system_healthcheck_new_upload")
        parser.add_argument("--zip-name", default="2021-05-06_Tri_lokality_XYZ.zip")
        parser.add_argument("--expected-locality", default="Xandovice")
        parser.add_argument("--timeout-seconds", type=int, default=900)
        parser.add_argument("--cleanup-after", action="store_true")

    def handle(self, *args, **options):
        dataset_dir = os.getenv("WRAP_TEST_DATA_DIR")
        if not dataset_dir:
            raise NewUploadHealthcheckError("WRAP_TEST_DATA_DIR is not configured inside the container.")

        zip_path = Path(dataset_dir) / options["zip_name"]
        if not zip_path.exists():
            raise NewUploadHealthcheckError(f"Test ZIP does not exist: {zip_path}")

        user = self._get_or_create_healthcheck_user(options["username"])
        caiduser = user.caiduser
        expected_locality = options["expected_locality"]

        self.stdout.write(f"Using healthcheck user: {user.username}")
        self.stdout.write(f"Using ZIP: {zip_path}")
        self._cleanup_user_artifacts(caiduser)

        uploaded_archive = self._upload_zip_via_api(user, zip_path)
        self.stdout.write(f"Created UploadedArchive ID: {uploaded_archive.id}")

        created_locality = self._poll_for_completion(
            uploaded_archive_id=uploaded_archive.id,
            caiduser=caiduser,
            expected_locality=expected_locality,
            timeout_seconds=options["timeout_seconds"],
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"New upload healthcheck passed. Created locality '{created_locality.name}' "
                f"for UploadedArchive {uploaded_archive.id}."
            )
        )

        if options["cleanup_after"]:
            self._cleanup_user_artifacts(caiduser)
            self.stdout.write("Cleaned up healthcheck artifacts.")

    def _get_or_create_healthcheck_user(self, username):
        User = get_user_model()
        user, _ = User.objects.get_or_create(
            username=username,
            defaults={
                "email": f"{username}@example.com",
                "is_staff": True,
            },
        )
        if not user.is_staff:
            user.is_staff = True
            user.save(update_fields=["is_staff"])

        caiduser: CaIDUser = user.caiduser
        if caiduser.workgroup is None:
            workgroup, _ = WorkGroup.objects.get_or_create(name=f"{username}_workgroup")
            caiduser.workgroup = workgroup
        caiduser.workgroup_admin = True
        caiduser.show_taxon_classification = True
        caiduser.show_reid = True
        caiduser.save()
        return user

    def _cleanup_user_artifacts(self, caiduser: CaIDUser):
        UploadedArchive.objects.filter(owner=caiduser).delete()
        Locality.objects.filter(owner=caiduser).delete()

    def _upload_zip_via_api(self, user, zip_path: Path) -> UploadedArchive:
        client = Client()
        client.force_login(user)

        response = client.post(
            reverse("caidapp:new_upload"),
            data={
                "locality_at_upload": "",
                "upload_target": "taxon_processing",
                "taxon_mode": "recognize_taxa",
                "directory_structure": "*/{locality}",
                "directory_mapping": json.dumps({"locality": 1}),
                "ml_consent": "on",
                "upload_files": [
                    SimpleUploadedFile(
                        zip_path.name,
                        zip_path.read_bytes(),
                        content_type="application/zip",
                    )
                ],
            },
        )

        if response.status_code != 200:
            raise NewUploadHealthcheckError(
                f"Upload endpoint returned {response.status_code}: {response.content.decode(errors='ignore')}"
            )

        payload = response.json()
        if not payload.get("ok"):
            raise NewUploadHealthcheckError(f"Upload endpoint returned non-ok payload: {payload}")

        uploaded_archive_id = payload.get("uploaded_archive_id")
        if not uploaded_archive_id:
            raise NewUploadHealthcheckError(f"Upload endpoint did not return uploaded_archive_id: {payload}")

        return UploadedArchive.objects.get(id=uploaded_archive_id)

    def _poll_for_completion(
        self,
        uploaded_archive_id: int,
        caiduser: CaIDUser,
        expected_locality: str,
        timeout_seconds: int,
    ) -> Locality:
        poll_interval = 10
        started_at = time.time()

        while time.time() - started_at < timeout_seconds:
            uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
            locality = Locality.objects.filter(owner=caiduser, name=expected_locality).first()

            if locality and uploaded_archive.import_finished:
                if uploaded_archive.mediafile_set.filter(locality=locality).exists():
                    return locality

            if uploaded_archive.taxon_status == "F":
                raise NewUploadHealthcheckError(
                    f"Taxon processing failed for UploadedArchive {uploaded_archive_id}: "
                    f"{uploaded_archive.status_message}"
                )

            elapsed = int(time.time() - started_at)
            self.stdout.write(
                f"Waiting for import... status={uploaded_archive.taxon_status} "
                f"import_finished={uploaded_archive.import_finished} elapsed={elapsed}s"
            )
            time.sleep(poll_interval)

        raise NewUploadHealthcheckError(
            f"Timeout waiting for UploadedArchive {uploaded_archive_id} to create locality '{expected_locality}'."
        )
