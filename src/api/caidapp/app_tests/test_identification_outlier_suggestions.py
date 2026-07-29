from django.urls import reverse
from pathlib import Path

from django.conf import settings
from django.test import TestCase

from caidapp import tasks
from caidapp.app_tests.factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    UploadedArchiveFactory,
)
from caidapp.models import IdentificationOutlierSuggestionResult, MediaFile


class IdentificationOutlierSuggestionsCallbackTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.workgroup = self.caiduser.workgroup
        self.archive = UploadedArchiveFactory(owner=self.caiduser)
        self.current_identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Current")
        self.suggested_identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Suggested")

    def test_success_callback_resolves_candidate_mediafile_from_worker_path(self):
        suspicious_mediafile = MediaFile.objects.create(
            parent=self.archive,
            media_type="image",
            mediafile="images/suspicious.jpg",
            image_file="images/suspicious.jpg",
            preview="previews/suspicious.jpg",
            original_filename="suspicious.jpg",
        )
        AnimalObservationFactory(mediafile=suspicious_mediafile, identity=self.current_identity)
        candidate_mediafile = MediaFile.objects.create(
            parent=self.archive,
            media_type="image",
            mediafile="images/candidate.jpg",
            image_file="images/candidate.jpg",
            preview="previews/candidate.jpg",
            original_filename="candidate.jpg",
        )
        AnimalObservationFactory(mediafile=candidate_mediafile, identity=self.suggested_identity)
        result = IdentificationOutlierSuggestionResult.objects.create(
            workgroup=self.workgroup,
            status="processing",
            suggestions=[],
        )

        output = {
            "status": "DONE",
            "message": "finished",
            "suggestions": [
                {
                    "query_idx": 0,
                    "suspicious_mediafile_id": suspicious_mediafile.id,
                    "current_identity_id": self.current_identity.id,
                    "reason": "Alternative identity matched this image better than the current identity.",
                    "suggestions": [
                        {
                            "identity_id": self.suggested_identity.id,
                            "db_idx": 7,
                            "mediafile_path": str(Path(settings.MEDIA_ROOT) / candidate_mediafile.image_file.name),
                            "score": 0.91,
                            "reason": "Top alternative identity based on identification similarity.",
                        }
                    ],
                }
            ],
        }

        tasks.identification_outlier_detection_on_success(output, result_id=result.id)
        result.refresh_from_db()

        self.assertEqual(result.status, "done")
        self.assertEqual(result.suggestions[0]["suggestions"][0]["mediafile_id"], candidate_mediafile.id)
        self.assertEqual(result.suggestions[0]["suggestions"][0]["db_idx"], 7)


class IdentificationOutlierSuggestionsAcceptViewTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.client.force_login(self.caiduser.user)
        self.workgroup = self.caiduser.workgroup
        self.archive = UploadedArchiveFactory(owner=self.caiduser)
        self.current_identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Current")
        self.suggested_identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Suggested")

    def test_accept_suggestion_updates_identity_and_removes_card_from_result(self):
        suspicious_mediafile = MediaFile.objects.create(
            parent=self.archive,
            media_type="image",
            mediafile="images/suspicious.jpg",
            image_file="images/suspicious.jpg",
            preview="previews/suspicious.jpg",
            original_filename="suspicious.jpg",
        )
        observation = AnimalObservationFactory(mediafile=suspicious_mediafile, identity=self.current_identity)
        result = IdentificationOutlierSuggestionResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            suggestions=[
                {
                    "suspicious_mediafile_id": suspicious_mediafile.id,
                    "current_identity_id": self.current_identity.id,
                    "suggestions": [
                        {
                            "identity_id": self.suggested_identity.id,
                            "mediafile_id": suspicious_mediafile.id,
                            "score": 0.9,
                        }
                    ],
                }
            ],
        )

        response = self.client.post(
            reverse("caidapp:accept_identification_outlier_suggestion"),
            {
                "suspicious_mediafile_id": suspicious_mediafile.id,
                "suggested_identity_id": self.suggested_identity.id,
                "result_id": result.id,
                "next": reverse("caidapp:identification_outlier_suggestions_result", args=[result.id]),
            },
        )

        self.assertEqual(response.status_code, 302)
        observation.refresh_from_db()
        result.refresh_from_db()
        self.assertEqual(observation.identity, self.suggested_identity)
        self.assertEqual(result.suggestions, [])
