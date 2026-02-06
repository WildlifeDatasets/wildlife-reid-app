import time
import sys
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from caidapp.models import UploadedArchive
from caidapp.tasks import run_species_prediction_async

class InferenceHealthcheckError(Exception):
    """Custom exception to trigger Sentry alert on healthcheck failure."""
    pass

class Command(BaseCommand):
    help = 'End-to-end production healthcheck of the inference pipeline (Taxon + Identification).'

    def handle(self, *args, **options):
        username = "system_healthcheck"
        User = get_user_model()
        
        self.stdout.write(f"Starting inference healthcheck for user: {username}")

        try:
            user = User.objects.get(username=username)
            caiduser = user.caiduser
        except User.DoesNotExist:
            raise InferenceHealthcheckError(f"Error: User '{username}' not found.")
        except Exception as e:
            raise InferenceHealthcheckError(f"Error getting CaIDUser profile for '{username}': {e}")

        # --- PART 1: Taxon Classification Healthcheck ---
        self.stdout.write("\n--- [PART 1] Taxon Classification Healthcheck ---")
        ua_taxon = UploadedArchive.objects.filter(owner=caiduser, contains_single_taxon=False).order_by('-uploaded_at').first()
        if not ua_taxon:
            self.stdout.write(self.style.WARNING("Warning: No archive for Taxon Classification found (contains_single_taxon=False). Skipping Part 1."))
        else:
            self.run_taxon_check(ua_taxon)

        # --- PART 2: Identification Healthcheck ---
        self.stdout.write("\n--- [PART 2] Identification Healthcheck ---")
        ua_id = UploadedArchive.objects.filter(
            owner=caiduser, 
            contains_single_taxon=True, 
            contains_identities=False
        ).order_by('-uploaded_at').first()
        
        if not ua_id:
            raise InferenceHealthcheckError("Error: No UploadedArchive found for identification healthcheck (contains_single_taxon=True, contains_identities=False).")
        
        if not caiduser.workgroup or not caiduser.workgroup.identification_model:
            raise InferenceHealthcheckError(f"Error: Workgroup '{caiduser.workgroup}' has no identification_model assigned. Cannot run identification.")

        self.run_identification_check(ua_id, caiduser.workgroup)

        self.stdout.write(self.style.SUCCESS("\nAll inference healthchecks passed!"))

    def run_taxon_check(self, ua):
        self.stdout.write(f"Using UploadedArchive ID: {ua.id} for Taxon Classification.")
        self.stdout.write("Triggering taxon inference task (force_init=True)...")
        
        # Let exceptions (e.g. Redis connection error) propagate to Sentry
        run_species_prediction_async(uploaded_archive=ua, force_init=True)
        
        self.poll_status(ua, 'taxon_status', ["TAID", "TKN", "TV", "IR"])
        self.stdout.write(self.style.SUCCESS("Taxon Classification passed."))

    def run_identification_check(self, ua, workgroup):
        from caidapp.views import run_identification
        
        self.stdout.write(f"Using UploadedArchive ID: {ua.id} for Identification.")
        self.stdout.write("Triggering identification task...")
        
        # Reset identification status to IR (Ready for ID) to ensure task can be triggered
        ua.identification_status = "IR"
        ua.save()

        # Let exceptions propagate to Sentry
        success = run_identification(ua, workgroup)
        if not success:
            raise InferenceHealthcheckError("Failed to trigger identification (run_identification returned False - possibly no observations found).")

        self.poll_status(ua, 'identification_status', ["IAID"])
        self.stdout.write(self.style.SUCCESS("Identification passed."))

    def poll_status(self, ua, field_name, success_statuses, timeout_seconds=900):
        poll_interval = 10
        start_time = time.time()
        
        self.stdout.write(f"Polling {field_name} (Timeout: {timeout_seconds}s)...")
        
        while time.time() - start_time < timeout_seconds:
            ua.refresh_from_db()
            status = getattr(ua, field_name)
            
            if status in success_statuses:
                self.stdout.write(f"Status reached: {status}")
                return
            
            if status == "F":
                 raise InferenceHealthcheckError(f"Task failed. {field_name} is 'F'. Message: {ua.status_message}")
            
            elapsed = int(time.time() - start_time)
            if elapsed > 0 and (elapsed // poll_interval) % 3 == 0:
                 if elapsed % 30 < poll_interval:
                    self.stdout.write(f"Still waiting... {field_name}: {status} ({elapsed}s elapsed)")

            time.sleep(poll_interval)
            
        raise InferenceHealthcheckError(f"Timeout waiting for {field_name}. Current status: {getattr(ua, field_name)}")
