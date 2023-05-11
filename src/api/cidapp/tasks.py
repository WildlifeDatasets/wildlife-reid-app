import logging

import django
from celery import shared_task
from cidapp.models import UploadedArchive

logger = logging.getLogger("app")


"""
Celery tasks used by "API worker" inside of the API docker container.
The API worker is different from e.g. Inference worker with its own docker container
because it has access to the database and other django resources,
and functions as a queue for processing worker responses.
"""


@shared_task(bind=True)
def predict_on_success(self, *args, uploaded_archive_id, **kwargs):
    """Success callback invoked after running predict function in inference worker."""
    logger.info("Inference task finished. Updating database record.")
    uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
    uploaded_archive.finished_at = django.utils.timezone.now()
    uploaded_archive.save()


@shared_task
def predict_on_error(request, exc, traceback):
    """Error callback invoked after running predict function in inference worker."""
    logger.error(f"Task {request.id} raised exception: {exc!r}\n{traceback!r}")
