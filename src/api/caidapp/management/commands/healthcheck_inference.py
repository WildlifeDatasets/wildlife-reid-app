import time
import sys
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from caidapp.models import UploadedArchive
from caidapp.tasks import run_species_prediction_async

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
            self.stderr.write(self.style.ERROR(f"Error: User '{username}' not found."))
            sys.exit(1)
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error getting CaIDUser profile for '{username}': {e}"))
            sys.exit(1)

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
            self.stderr.write(self.style.ERROR("Error: No UploadedArchive found for identification healthcheck (contains_single_taxon=True, contains_identities=False)."))
            sys.exit(1)
        
        if not caiduser.workgroup or not caiduser.workgroup.identification_model:
            self.stderr.write(self.style.ERROR(f"Error: Workgroup '{caiduser.workgroup}' has no identification_model assigned. Cannot run identification."))
            sys.exit(1)

        self.run_identification_check(ua_id, caiduser.workgroup)

        self.stdout.write(self.style.SUCCESS("\nAll inference healthchecks passed!"))
        sys.exit(0)

    def run_taxon_check(self, ua):
        self.stdout.write(f"Using UploadedArchive ID: {ua.id} for Taxon Classification.")
        self.stdout.write("Triggering taxon inference task (force_init=True)...")
        try:
            run_species_prediction_async(uploaded_archive=ua, force_init=True)
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error triggering taxon inference: {e}"))
            sys.exit(1)
        
        if not self.poll_status(ua, 'taxon_status', ["TAID", "TKN", "TV", "IR"]):
            self.stderr.write(self.style.ERROR("Taxon Classification healthcheck failed."))
            sys.exit(1)
        self.stdout.write(self.style.SUCCESS("Taxon Classification passed."))

    def run_identification_check(self, ua, workgroup):
        from caidapp.views import run_identification
        
        self.stdout.write(f"Using UploadedArchive ID: {ua.id} for Identification.")
        self.stdout.write("Triggering identification task...")
        
        # Reset identification status to IR (Ready for ID) to ensure task can be triggered
        ua.identification_status = "IR"
        ua.save()

        try:
            success = run_identification(ua, workgroup)
            if not success:
                self.stderr.write(self.style.ERROR("Failed to trigger identification (run_identification returned False - possibly no observations found)."))
                sys.exit(1)
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error calling run_identification: {e}"))
            import traceback
            self.stderr.write(traceback.format_exc())
            sys.exit(1)

        if not self.poll_status(ua, 'identification_status', ["IAID"]):
            self.stderr.write(self.style.ERROR("Identification healthcheck failed."))
            sys.exit(1)
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
                return True
            
            if status == "F":
                 self.stderr.write(self.style.ERROR(f"Task failed. {field_name} is 'F'. Message: {ua.status_message}"))
                 return False
            
            elapsed = int(time.time() - start_time)
            if elapsed > 0 and (elapsed // poll_interval) % 3 == 0: # Every 30s approx
                 # We use // and % to avoid multiple writes in the same 10s window if we wanted, 
                 # but elapsed % 30 is simpler if we only call sleep at the end.
                 if elapsed % 30 < poll_interval:
                    self.stdout.write(f"Still waiting... {field_name}: {status} ({elapsed}s elapsed)")

            time.sleep(poll_interval)
            
        self.stderr.write(self.style.ERROR(f"Timeout waiting for {field_name}. Current status: {getattr(ua, field_name)}"))
        return False
