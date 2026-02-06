import time
import sys
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from caidapp.models import UploadedArchive
from caidapp.tasks import run_species_prediction_async

class Command(BaseCommand):
    help = 'End-to-end production healthcheck of the inference pipeline (API -> broker -> worker -> GPU -> result).'

    def handle(self, *args, **options):
        username = "system_healthcheck"
        User = get_user_model()
        
        self.stdout.write(f"Starting inference healthcheck for user: {username}")

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            self.stderr.write(f"Error: User '{username}' not found.")
            sys.exit(1)

        caiduser = user.caiduser
        # Get the latest uploaded archive for this user
        ua = UploadedArchive.objects.filter(owner=caiduser, contains_single_taxon=False).order_by('-uploaded_at').first()
        
        if not ua:
            self.stderr.write(f"Error: No UploadedArchive found for user '{username}'. User must have test data.")
            sys.exit(1)

        self.stdout.write(f"Using UploadedArchive ID: {ua.id} (Current Status: {ua.taxon_status})")

        # Trigger inference
        self.stdout.write("Triggering inference task (force_init=True)...")
        try:
            # force_init=True ensures a fresh run, deleting old mediafiles and reprocessing
            run_species_prediction_async(uploaded_archive=ua, force_init=True)
            self.stdout.write("Inference task triggered successfully.")
        except Exception as e:
            self.stderr.write(f"Error triggering inference: {e}")
            sys.exit(1)

        # Polling for completion
        # Timeout: 10 minutes (600 seconds)
        timeout_seconds = 600
        poll_interval = 10
        start_time = time.time()
        
        self.stdout.write(f"Waiting for completion (Timeout: {timeout_seconds}s)...")
        
        while time.time() - start_time < timeout_seconds:
            # Refresh from DB to get latest status
            ua.refresh_from_db()
            status = ua.taxon_status
            
            # Check for success status
            # TAID = Taxon AI Done (Success)
            # TKN = Taxa Known (Also implies success/done)
            # TV = Taxa Verified
            # ID = Identified
            # Essentially anything that means "AI finished"
            success_statuses = ["TAID", "TKN", "TV", "IR", "IAIP", "IAID", "ID"]
            
            if status in success_statuses:
                self.stdout.write(self.style.SUCCESS(f"Healthcheck passed! Final Status: {status}"))
                sys.exit(0)
            
            # Check for failure status
            if status == "F":
                 self.stderr.write(self.style.ERROR(f"Healthcheck failed. Status is Failed (F). Message: {ua.status_message}"))
                 sys.exit(1)
            
            if int(time.time() - start_time) % 30 == 0:
                 self.stdout.write(f"Still waiting... Current Status: {status}")

            time.sleep(poll_interval)
            
        self.stderr.write(self.style.ERROR("Healthcheck timed out."))
        sys.exit(1)
