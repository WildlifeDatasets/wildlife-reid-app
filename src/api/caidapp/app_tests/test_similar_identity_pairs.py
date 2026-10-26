from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from caidapp import tasks
from caidapp.app_tests.factories import (
    AnimalObservationFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    UploadedArchiveFactory,
)
from caidapp.models import (
    IdentificationOutlierSuggestionResult,
    IdentificationSimilarPairResult,
    IndividualIdentity,
    MediaFile,
    NotificationRecipient,
)
from django.test import TestCase
from django.urls import reverse


class SimilarIdentityPairsTest(TestCase):
    def setUp(self):
        """Prepare staff, a regular member, and temporary media for similar-identity pair searches."""
        self.staff = CaidUserFactory(admin=False)
        self.staff.user.is_staff = True
        self.staff.user.save(update_fields=["is_staff"])
        self.workgroup = self.staff.workgroup
        self.nonstaff = CaidUserFactory(admin=True, workgroup=self.workgroup)
        self.client.force_login(self.staff.user)
        self.temporary_media = TemporaryDirectory()
        self.addCleanup(self.temporary_media.cleanup)
        media_settings = self.settings(MEDIA_ROOT=self.temporary_media.name)
        media_settings.enable()
        self.addCleanup(media_settings.disable)

    def test_start_queues_csv_preparation_without_scanning_media_in_http_request(self):
        """Queue CSV preparation outside the request and dispatch the similarity worker from that task."""
        csv_data = {
            "image_path": [
                str(Path(self.temporary_media.name) / "images/a.jpg"),
                str(Path(self.temporary_media.name) / "images/b.jpg"),
            ],
            "class_id": [11, 22],
            "label": ["A", "B"],
        }
        with patch("caidapp.tasks.prepare_similar_identity_pairs.apply_async") as prepare_mock, patch(
            "caidapp.tasks._prepare_dataframe_for_identification",
            side_effect=AssertionError("CSV prepared in HTTP request"),
        ):
            response = self.client.post(reverse("caidapp:run_similar_identity_pairs"))

        result = IdentificationSimilarPairResult.objects.get()
        self.assertRedirects(
            response,
            reverse("caidapp:similar_identity_pairs_result", args=[result.id]),
            fetch_redirect_response=False,
        )
        self.assertEqual(result.status, "processing")
        self.assertTrue(result.task_id)
        self.assertIn("Preparing", result.message)
        self.assertFalse(self.workgroup.file_path(result.input_csv_filename).exists())
        prepare_mock.assert_called_once_with(args=[result.id], task_id=result.task_id, queue="celery", retry=False)

        job = MagicMock(id="similar-job-1")
        with patch("caidapp.tasks._prepare_dataframe_for_identification", return_value=csv_data), patch(
            "caidapp.tasks.signature"
        ) as signature_mock:
            signature_mock.return_value.apply_async.return_value = job
            tasks.prepare_similar_identity_pairs(result.id)

        result.refresh_from_db()
        self.assertEqual(result.task_id, "similar-job-1")
        self.assertTrue(self.workgroup.file_path(result.input_csv_filename).is_file())
        self.assertEqual(result.output_csv_filename, f"identification_similar_pairs_{result.id}_most_similar.csv")
        signature_mock.assert_called_once()
        self.assertEqual(signature_mock.call_args.args[0], "detect_most_similar")
        self.assertEqual(signature_mock.call_args.kwargs["kwargs"]["organization_id"], self.workgroup.id)
        self.assertEqual(signature_mock.return_value.apply_async.call_count, 1)
        recipients = NotificationRecipient.objects.values_list("user_id", flat=True)
        self.assertEqual(list(recipients), [self.staff.id])

    def test_csv_preparation_error_is_saved_by_background_task(self):
        """Persist CSV preparation errors without dispatching the similarity worker."""
        with patch("caidapp.tasks.prepare_similar_identity_pairs.apply_async"):
            result = tasks.run_similar_identity_pairs_for_workgroup(self.workgroup)
        with patch(
            "caidapp.tasks._prepare_dataframe_for_identification", side_effect=OSError("image store unavailable")
        ), patch("caidapp.tasks.signature") as signature_mock:
            tasks.prepare_similar_identity_pairs(result.id)
        result.refresh_from_db()
        self.assertEqual(result.status, "error")
        self.assertIn("image store unavailable", result.message)
        self.assertIsNotNone(result.finished_at)
        signature_mock.assert_not_called()

    def test_callback_stores_both_sources_and_links_to_result(self):
        """Save both similarity sources, notify staff, and link pair results to comparison and merge review."""
        first_identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="First")
        second_identity = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Second")
        archive = UploadedArchiveFactory(owner=self.staff)
        first_media = MediaFile.objects.create(
            parent=archive,
            media_type="image",
            mediafile="images/a.jpg",
            image_file="images/a.jpg",
            preview="previews/a.jpg",
            original_filename="a.jpg",
        )
        second_media = MediaFile.objects.create(
            parent=archive,
            media_type="image",
            mediafile="images/b.jpg",
            image_file="images/b.jpg",
            preview="previews/b.jpg",
            original_filename="b.jpg",
        )
        result = IdentificationSimilarPairResult.objects.create(workgroup=self.workgroup)
        pair = {
            "path_a": str(Path(self.temporary_media.name) / first_media.image_file.name),
            "path_b": str(Path(self.temporary_media.name) / second_media.image_file.name),
            "class_id_a": first_identity.id,
            "class_id_b": second_identity.id,
            "label_a": first_identity.name,
            "label_b": second_identity.name,
            "similarity": 0.91,
        }

        tasks.similar_identity_pairs_on_success(
            {
                "status": "DONE",
                "message": "finished",
                "mega_pairs": [pair],
                "local_pairs": [{**pair, "similarity": 18}],
            },
            result_id=result.id,
        )
        result.refresh_from_db()
        self.assertEqual(result.status, "done")
        self.assertEqual(result.mega_pairs[0]["mediafile_a_id"], first_media.id)
        self.assertEqual(result.local_pairs[0]["mediafile_b_id"], second_media.id)
        self.assertEqual(result.local_pairs[0]["similarity"], 18)
        self.assertIsNotNone(result.finished_at)
        recipients = NotificationRecipient.objects.select_related("notification").all()
        self.assertEqual(recipients.count(), 1)
        self.assertEqual(recipients[0].user_id, self.staff.id)
        self.assertEqual(
            recipients[0].notification.get_link_url(),
            reverse("caidapp:similar_identity_pairs_result", args=[result.id]),
        )

        response = self.client.get(reverse("caidapp:similar_identity_pairs_result", args=[result.id]))
        self.assertContains(response, "Cosine similarity: 0.910")
        self.assertContains(response, "Matched points: 18")
        self.assertContains(
            response,
            f'{reverse("caidapp:comparison")}?a_kind=mediafile&amp;a_id={first_media.id}'
            f"&amp;b_kind=mediafile&amp;b_id={second_media.id}",
        )
        self.assertContains(response, "Merge preview", count=2)
        self.assertNotContains(response, "Review A into B")
        self.assertNotContains(response, "Review B into A")
        self.assertContains(response, f"from_mediafile_id={first_media.id}")
        self.assertContains(response, f"to_mediafile_id={second_media.id}")
        info = self.client.get(reverse("caidapp:identification_information"))
        self.assertContains(info, "Similar identity pairs")
        self.assertContains(info, "1 Mega pairs, 1 Local pairs")

    def test_pair_mediafile_count_deduplicates_observations_and_excludes_placeholders(self):
        """Count real media files once even when an identity has multiple observations per file."""
        identity = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        archive = UploadedArchiveFactory(owner=self.staff)
        mediafile = MediaFile.objects.create(parent=archive, media_type="image")
        AnimalObservationFactory(mediafile=mediafile, identity=identity)
        AnimalObservationFactory(mediafile=mediafile, identity=identity)
        placeholder_mediafile = MediaFile.objects.create(parent=archive, media_type="image")
        placeholder_mediafile.observations.update(identity=identity)
        other_identity = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            mega_pairs=[{"class_id_a": identity.id, "class_id_b": other_identity.id}],
        )

        response = self.client.get(reverse("caidapp:similar_identity_pairs_result", args=[result.id]))

        self.assertEqual(response.context["mega_cards"][0]["a"]["identity"].mediafile_count, 1)
        self.assertContains(response, "1 media file")

    def test_merged_pairs_are_hidden_by_default_and_can_be_shown(self):
        """Hide stale pairs in both methods after merging, retaining the saved result for optional review."""
        source = IndividualIdentityFactory(owner_workgroup=self.workgroup, name="Old identity")
        target = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        remaining = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        merged_pair = {
            "class_id_a": source.pk,
            "class_id_b": target.pk,
            "label_a": source.name,
            "similarity": 0.99,
        }
        active_pair = {"class_id_a": target.pk, "class_id_b": remaining.pk, "similarity": 0.8}
        pairs = [merged_pair, active_pair]
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup, status="done", mega_pairs=pairs, local_pairs=pairs
        )
        url = reverse("caidapp:similar_identity_pairs_result", args=[result.pk])
        before = self.client.get(url)
        self.assertEqual(len(before.context["mega_cards"]), 2)
        self.assertEqual(before.context["resolved_count"], 0)

        merged = self.client.post(
            reverse("caidapp:merge_identities", args=[source.pk, target.pk]),
            {"name": target.name, "sex": "U", "coat_type": "U", "next": url},
        )
        self.assertRedirects(merged, url, fetch_redirect_response=False)

        for view_url in (url, reverse("caidapp:similar_identity_pairs")):
            with self.subTest(view_url=view_url):
                hidden = self.client.get(view_url)
                self.assertEqual(hidden.context["resolved_count"], 2)
                self.assertFalse(hidden.context["show_resolved"])
                for key in ("mega_cards", "local_cards"):
                    self.assertEqual(len(hidden.context[key]), 1)
                    self.assertFalse(hidden.context[key][0]["is_resolved"])
                self.assertNotContains(hidden, "Old identity")
                shown = self.client.get(view_url, {"show_resolved": "1"})
                self.assertTrue(shown.context["show_resolved"])
                self.assertContains(shown, 'value="1" checked')
                self.assertContains(shown, "Old identity")
                self.assertContains(shown, "Merge preview", count=2)
                for key in ("mega_cards", "local_cards"):
                    self.assertEqual(len(shown.context[key]), 2)
                    self.assertTrue(shown.context[key][0]["is_resolved"])
                active_card = shown.context["mega_cards"][1]
                self.assertEqual(
                    parse_qs(urlparse(active_card["preview_url"]).query)["next"], [view_url + "?show_resolved=1"]
                )

        result.refresh_from_db()
        self.assertEqual(result.mega_pairs, pairs)
        self.assertEqual(result.local_pairs, pairs)

    def test_deleted_identity_and_identical_sides_are_resolved_without_mediafile_requirements(self):
        """Resolve unavailable or identical identity pairs while keeping valid pairs without media previews."""
        first = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        second = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        deleted_id = second.pk
        second.delete()
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            mega_pairs=[
                {"class_id_a": first.pk, "class_id_b": deleted_id},
                {"class_id_a": first.pk, "class_id_b": first.pk},
            ],
        )
        url = reverse("caidapp:similar_identity_pairs_result", args=[result.pk])

        hidden = self.client.get(url)
        self.assertEqual(hidden.context["mega_cards"], [])
        self.assertEqual(hidden.context["resolved_count"], 2)
        self.assertContains(hidden, "No unresolved pairs")
        shown = self.client.get(url, {"show_resolved": "1"})
        self.assertEqual(len(shown.context["mega_cards"]), 2)
        self.assertNotContains(shown, "Merge preview")

    def test_review_cursor_keeps_saved_order_when_resolved_pairs_are_hidden(self):
        """Link both methods to stable saved indices even when earlier cards are hidden."""
        first = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        second = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        active = {"class_id_a": first.pk, "class_id_b": second.pk}
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup, status="done", mega_pairs=[{}, active], local_pairs=[active]
        )

        response = self.client.get(reverse("caidapp:similar_identity_pairs_result", args=[result.pk]))

        self.assertEqual(response.context["mega_cards"][0]["review_index"], 1)
        self.assertEqual(response.context["local_cards"][0]["review_index"], 2)
        self.assertContains(response, f"review_result={result.pk}&amp;review_index=1")
        self.assertContains(response, f"review_result={result.pk}&amp;review_index=2")
        self.assertNotContains(response, 'class="small text-body-secondary">A</div>')
        self.assertNotContains(response, 'class="small text-body-secondary">B</div>')

    def test_merge_and_review_next_skips_unavailable_pairs_and_continues_across_methods(self):
        """Open the next valid preview after one merge, and return to the filtered list when finished."""
        first, second, third = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(3)]
        foreign = IndividualIdentityFactory(owner_workgroup=CaidUserFactory().workgroup)
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            mega_pairs=[
                {"class_id_a": first.pk, "class_id_b": second.pk},
                {"class_id_a": first.pk, "class_id_b": third.pk},
                {"class_id_a": second.pk, "class_id_b": second.pk},
                {"class_id_a": second.pk, "class_id_b": foreign.pk},
            ],
            local_pairs=[{"class_id_a": second.pk, "class_id_b": third.pk}],
        )
        return_url = reverse("caidapp:similar_identity_pairs_result", args=[result.pk]) + "?show_resolved=1"
        data = {
            "name": second.name,
            "sex": "U",
            "coat_type": "U",
            "next": return_url,
            "review_result": result.pk,
            "review_index": 0,
            "action": "merge_next",
        }

        response = self.client.post(reverse("caidapp:merge_identities", args=[first.pk, second.pk]), data)

        next_url = urlparse(response.url)
        self.assertEqual(next_url.path, reverse("caidapp:merge_identities", args=[second.pk, third.pk]))
        self.assertEqual(parse_qs(next_url.query)["review_index"], ["4"])
        self.assertEqual(parse_qs(next_url.query)["next"], [return_url])
        self.assertTrue(type(third).objects.filter(pk=third.pk).exists())
        preview = self.client.get(response.url)
        self.assertContains(preview, "Merge and review next")
        self.assertNotContains(preview, "Browse with the arrows")
        self.assertEqual(preview.context["review_queue"]["index"], 4)

        data.update(name=third.name, review_index=4)
        finished = self.client.post(response.url, data)

        self.assertRedirects(finished, return_url, fetch_redirect_response=False)
        self.assertFalse(type(second).objects.filter(pk=second.pk).exists())
        self.assertTrue(type(third).objects.filter(pk=third.pk).exists())
        self.assertContains(self.client.get(finished.url), "No further unresolved suggestions")

    def test_review_queue_survives_swap_and_invalid_form(self):
        """Keep the original queue cursor when reversing merge direction and correcting validation errors."""
        first, second, third = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(3)]
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            mega_pairs=[
                {"class_id_a": first.pk, "class_id_b": second.pk},
                {"class_id_a": first.pk, "class_id_b": third.pk},
            ],
        )
        return_url = reverse("caidapp:similar_identity_pairs_result", args=[result.pk])
        preview = self.client.get(
            reverse("caidapp:merge_identities", args=[first.pk, second.pk]),
            {"review_result": result.pk, "review_index": 0, "next": return_url},
        )
        swapped = self.client.get(preview.context["swap_url"])
        self.assertContains(swapped, "Merge and review next")
        self.assertEqual(swapped.context["review_queue"]["index"], 0)
        data = {
            "name": "",
            "sex": "U",
            "coat_type": "U",
            "next": return_url,
            "review_result": result.pk,
            "review_index": 0,
            "action": "merge_next",
        }
        merge_url = reverse("caidapp:merge_identities", args=[second.pk, first.pk])
        invalid = self.client.post(merge_url, data)
        self.assertEqual(invalid.status_code, 200)
        self.assertContains(invalid, "Merge and review next")
        self.assertContains(invalid, 'name="review_index" value="0"')
        self.assertTrue(type(second).objects.filter(pk=second.pk).exists())

        data["name"] = first.name
        response = self.client.post(merge_url, data)

        self.assertEqual(urlparse(response.url).path, reverse("caidapp:merge_identities", args=[first.pk, third.pk]))

    def test_queue_requires_matching_pair_workgroup_and_staff_access(self):
        """Ignore missing, malformed, mismatched, and inaccessible saved-review cursors."""
        first, second, third = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(3)]
        own_result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            mega_pairs=[{"class_id_a": first.pk, "class_id_b": second.pk}],
        )
        foreign_result = IdentificationSimilarPairResult.objects.create(
            workgroup=CaidUserFactory().workgroup, status="done", mega_pairs=own_result.mega_pairs
        )
        merge_url = reverse("caidapp:merge_identities", args=[first.pk, second.pk])
        for params in (
            {},
            {"review_result": "bad", "review_index": "bad"},
            {"review_result": own_result.pk, "review_index": -1},
            {"review_result": own_result.pk, "review_index": 10},
            {"review_result": foreign_result.pk, "review_index": 0},
        ):
            with self.subTest(params=params):
                self.assertNotContains(self.client.get(merge_url, params), "Merge and review next")
        valid_params = {"review_result": own_result.pk, "review_index": 0}
        mismatched = reverse("caidapp:merge_identities", args=[first.pk, third.pk])
        self.assertNotContains(self.client.get(mismatched, valid_params), "Merge and review next")
        self.client.force_login(self.nonstaff.user)
        self.assertNotContains(self.client.get(merge_url, valid_params), "Merge and review next")

    def test_regular_merge_with_review_cursor_still_returns_to_list(self):
        """Advance through previews only when the explicit merge-and-review action was chosen."""
        first, second, third = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(3)]
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            mega_pairs=[
                {"class_id_a": first.pk, "class_id_b": second.pk},
                {"class_id_a": second.pk, "class_id_b": third.pk},
            ],
        )
        return_url = reverse("caidapp:similar_identity_pairs_result", args=[result.pk])

        response = self.client.post(
            reverse("caidapp:merge_identities", args=[first.pk, second.pk]),
            {
                "name": second.name,
                "sex": "U",
                "coat_type": "U",
                "next": return_url,
                "review_result": result.pk,
                "review_index": 0,
            },
        )

        self.assertRedirects(response, return_url, fetch_redirect_response=False)
        self.assertTrue(type(third).objects.filter(pk=third.pk).exists())

    def test_nonadmin_direct_merge_controls_are_visible_disabled_and_enforced(self):
        """Advertise the admin action to staff members without allowing forged direct-merge requests."""
        first, second = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(2)]
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup, status="done", mega_pairs=[{"class_id_a": first.pk, "class_id_b": second.pk}]
        )
        response = self.client.get(reverse("caidapp:similar_identity_pairs_result", args=[result.pk]))
        self.assertContains(response, "Workgroup admin")
        self.assertContains(response, "Only workgroup administrators can use this action")
        self.assertContains(response, 'form="bulk-similar-merge-form" disabled')
        self.assertContains(response, "Merge preview")
        denied = self.client.post(reverse("caidapp:merge_similar_identity_pairs", args=[result.pk]), {"pairs": ["0"]})
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(
            self.client.post(
                reverse("caidapp:merge_selected_identities"), {"suggestions": [f"{first.pk}|{second.pk}"]}
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.get(reverse("caidapp:merge_identities_no_preview", args=[first.pk, second.pk])).status_code, 403
        )
        self.assertTrue(IndividualIdentity.objects.filter(pk=first.pk).exists())

    def test_admin_bulk_merge_deduplicates_pairs_and_preserves_selected_direction(self):
        """Merge independent selected pairs once, moving observations into the displayed targets."""
        self.staff.workgroup_admin = True
        self.staff.save(update_fields=["workgroup_admin"])
        first, second, third, fourth, unselected = [
            IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(5)
        ]
        mediafile = MediaFile.objects.create(parent=UploadedArchiveFactory(owner=self.staff), media_type="image")
        observation = AnimalObservationFactory(mediafile=mediafile, identity=first)
        pairs = [
            {"class_id_a": first.pk, "class_id_b": second.pk},
            {"class_id_a": third.pk, "class_id_b": fourth.pk},
            {"class_id_a": fourth.pk, "class_id_b": unselected.pk},
        ]
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup, status="done", mega_pairs=pairs, local_pairs=[pairs[0]]
        )
        return_url = reverse("caidapp:similar_identity_pairs_result", args=[result.pk]) + "?show_resolved=1"

        response = self.client.post(
            reverse("caidapp:merge_similar_identity_pairs", args=[result.pk]),
            {"pairs": ["0", "1", "3"], "next": return_url},
        )

        self.assertRedirects(response, return_url, fetch_redirect_response=False)
        self.assertFalse(IndividualIdentity.objects.filter(pk__in=[first.pk, third.pk]).exists())
        self.assertEqual(IndividualIdentity.objects.filter(pk__in=[second.pk, fourth.pk, unselected.pk]).count(), 3)
        observation.refresh_from_db()
        self.assertEqual(observation.identity_id, second.pk)
        second.refresh_from_db()
        self.assertEqual(second.updated_by_id, self.staff.pk)
        self.assertContains(self.client.get(return_url), "Merged 2 selected identity pairs.")

    def test_bulk_merge_rejects_overlapping_pairs_without_changes(self):
        """Reject an ambiguous batch before modifying any identity."""
        self.staff.workgroup_admin = True
        self.staff.save(update_fields=["workgroup_admin"])
        first, second, third = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(3)]
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            mega_pairs=[
                {"class_id_a": first.pk, "class_id_b": second.pk},
                {"class_id_a": second.pk, "class_id_b": third.pk},
            ],
        )

        response = self.client.post(
            reverse("caidapp:merge_similar_identity_pairs", args=[result.pk]), {"pairs": ["0", "1"]}
        )

        self.assertContains(self.client.get(response.url), "Select non-overlapping pairs")
        self.assertEqual(IndividualIdentity.objects.filter(pk__in=[first.pk, second.pk, third.pk]).count(), 3)

    def test_bulk_merge_rejects_stale_foreign_and_invalid_selections(self):
        """Validate every saved pair before merging, including workgroup ownership and submitted indices."""
        self.staff.workgroup_admin = True
        self.staff.save(update_fields=["workgroup_admin"])
        first, second = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(2)]
        foreign = IndividualIdentityFactory(owner_workgroup=CaidUserFactory().workgroup)
        deleted = IndividualIdentityFactory(owner_workgroup=self.workgroup)
        deleted_id = deleted.pk
        deleted.delete()
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            mega_pairs=[
                {"class_id_a": first.pk, "class_id_b": second.pk},
                {"class_id_a": deleted_id, "class_id_b": foreign.pk},
            ],
        )
        url = reverse("caidapp:merge_similar_identity_pairs", args=[result.pk])
        for indices in ([], ["-1"], ["bad"], ["99"], ["0", "1"]):
            with self.subTest(indices=indices):
                response = self.client.post(url, {"pairs": indices, "next": "https://example.org/outside"})
                self.assertRedirects(
                    response,
                    reverse("caidapp:similar_identity_pairs_result", args=[result.pk]),
                    fetch_redirect_response=False,
                )
                self.assertEqual(IndividualIdentity.objects.filter(pk__in=[first.pk, second.pk]).count(), 2)
        other_result = IdentificationSimilarPairResult.objects.create(workgroup=foreign.owner_workgroup, status="done")
        self.assertEqual(
            self.client.post(
                reverse("caidapp:merge_similar_identity_pairs", args=[other_result.pk]), {"pairs": ["0"]}
            ).status_code,
            404,
        )
        self.assertEqual(self.client.get(url).status_code, 405)

    def test_bulk_merge_rolls_back_the_batch_on_failure(self):
        """Roll back earlier merges if a later pair fails during the same batch."""
        from caidapp.views import merge_identities_helper

        self.staff.workgroup_admin = True
        self.staff.save(update_fields=["workgroup_admin"])
        identities = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(4)]
        original_ids = [identity.pk for identity in identities]
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            mega_pairs=[
                {"class_id_a": original_ids[0], "class_id_b": original_ids[1]},
                {"class_id_a": original_ids[2], "class_id_b": original_ids[3]},
            ],
        )

        def fail_second(request, source, target):
            if source.pk == original_ids[2]:
                raise RuntimeError("Simulated second-pair failure")
            merge_identities_helper(request, source, target)

        with patch("caidapp.views.merge_identities_helper", side_effect=fail_second), self.assertRaises(RuntimeError):
            self.client.post(reverse("caidapp:merge_similar_identity_pairs", args=[result.pk]), {"pairs": ["0", "1"]})

        self.assertEqual(IndividualIdentity.objects.filter(pk__in=original_ids).count(), 4)

    def test_row_layout_and_direction_preserve_list_state_and_media_previews(self):
        """Render a full-width row and carry its selected direction and list filters into merge preview."""
        first, second = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(2)]
        archive = UploadedArchiveFactory(owner=self.staff)
        first_media = MediaFile.objects.create(parent=archive, media_type="image")
        second_media = MediaFile.objects.create(parent=archive, media_type="image")
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            mega_pairs=[
                {
                    "class_id_a": first.pk,
                    "class_id_b": second.pk,
                    "mediafile_a_id": first_media.pk,
                    "mediafile_b_id": second_media.pk,
                }
            ],
        )
        url = reverse("caidapp:similar_identity_pairs_result", args=[result.pk])
        response = self.client.get(url, {"layout": "rows", "show_resolved": "1", "reversed": "0"})

        self.assertTemplateUsed(response, "caidapp/similar_identity_pair_rows.html")
        self.assertContains(response, 'class="similar-pair-row"')
        self.assertContains(response, "js-pair-direction")
        self.assertContains(response, 'name="reverse_pairs" value="0"')
        self.assertContains(response, 'aria-current="true"')
        card = response.context["mega_cards"][0]
        self.assertTrue(card["is_reversed"])
        self.assertEqual(card["merge_source"]["identity"], second)
        preview = urlparse(card["preview_url"])
        self.assertEqual(preview.path, reverse("caidapp:merge_identities", args=[second.pk, first.pk]))
        params = parse_qs(preview.query)
        self.assertEqual(params["from_mediafile_id"], [str(second_media.pk)])
        self.assertEqual(params["to_mediafile_id"], [str(first_media.pk)])
        self.assertIn("layout=rows", params["next"][0])
        self.assertIn("reversed=0", params["next"][0])
        cards_url = response.context["cards_layout_url"]
        self.assertIn("show_resolved=1", cards_url)
        self.assertIn("reversed=0", cards_url)
        cards = self.client.get(url + cards_url)
        self.assertTemplateUsed(cards, "caidapp/similar_identity_pair_cards.html")
        self.assertEqual(cards.context["mega_cards"][0]["merge_source"]["identity"], second)

    def test_unknown_layout_and_malformed_direction_fall_back_to_cards(self):
        """Use the original layout for unsupported values and ignore invalid direction indices."""
        first, second = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(2)]
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup, status="done", mega_pairs=[{"class_id_a": first.pk, "class_id_b": second.pk}]
        )
        response = self.client.get(
            reverse("caidapp:similar_identity_pairs_result", args=[result.pk]),
            {"layout": "unknown", "reversed": "-1,bad,999"},
        )

        self.assertEqual(response.context["layout"], "cards")
        self.assertFalse(response.context["mega_cards"][0]["is_reversed"])
        self.assertEqual(response.context["reversed_pairs_value"], "")

    def test_bulk_merge_uses_row_direction_and_rejects_invalid_direction_indices(self):
        """Keep the row's chosen target and apply no changes for malformed direction choices."""
        self.staff.workgroup_admin = True
        self.staff.save(update_fields=["workgroup_admin"])
        first, second = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(2)]
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup, status="done", mega_pairs=[{"class_id_a": first.pk, "class_id_b": second.pk}]
        )
        url = reverse("caidapp:merge_similar_identity_pairs", args=[result.pk])
        for direction in ("bad", "-1", "99"):
            response = self.client.post(url, {"pairs": ["0"], "reverse_pairs": [direction]})
            self.assertEqual(response.status_code, 302)
            self.assertTrue(IndividualIdentity.objects.filter(pk=second.pk).exists())
        return_url = reverse("caidapp:similar_identity_pairs_result", args=[result.pk]) + "?layout=rows&reversed=0"

        response = self.client.post(url, {"pairs": ["0"], "reverse_pairs": ["0"], "next": return_url})

        self.assertRedirects(response, return_url, fetch_redirect_response=False)
        self.assertTrue(IndividualIdentity.objects.filter(pk=first.pk).exists())
        self.assertFalse(IndividualIdentity.objects.filter(pk=second.pk).exists())

    def test_review_next_respects_direction_choices_on_the_original_rows(self):
        """Use the next row's saved direction when continuing from merge preview."""
        first, second, third, fourth = [IndividualIdentityFactory(owner_workgroup=self.workgroup) for _ in range(4)]
        result = IdentificationSimilarPairResult.objects.create(
            workgroup=self.workgroup,
            status="done",
            mega_pairs=[
                {"class_id_a": first.pk, "class_id_b": second.pk},
                {"class_id_a": third.pk, "class_id_b": fourth.pk},
            ],
        )
        return_url = reverse("caidapp:similar_identity_pairs_result", args=[result.pk]) + "?layout=rows&reversed=1"

        response = self.client.post(
            reverse("caidapp:merge_identities", args=[first.pk, second.pk]),
            {
                "name": second.name,
                "sex": "U",
                "coat_type": "U",
                "next": return_url,
                "action": "merge_next",
                "review_result": result.pk,
                "review_index": 0,
            },
        )

        self.assertEqual(urlparse(response.url).path, reverse("caidapp:merge_identities", args=[fourth.pk, third.pk]))
        self.assertEqual(parse_qs(urlparse(response.url).query)["next"], [return_url])

    def test_nonstaff_cannot_start_view_or_download_results(self):
        """Deny pair search, result pages, and CSV downloads to non-staff members."""
        result = IdentificationSimilarPairResult.objects.create(workgroup=self.workgroup)
        self.client.force_login(self.nonstaff.user)
        dashboard = self.client.get(reverse("caidapp:dash_identities"))
        self.assertNotContains(dashboard, "Find similar identity pairs")
        self.assertEqual(self.client.post(reverse("caidapp:run_similar_identity_pairs")).status_code, 403)
        self.assertEqual(
            self.client.get(reverse("caidapp:similar_identity_pairs_result", args=[result.id])).status_code, 403
        )
        self.assertEqual(
            self.client.get(
                reverse("caidapp:download_similar_identity_pairs_csv", args=[result.id, "input"])
            ).status_code,
            403,
        )

    def test_staff_cannot_view_another_workgroups_result(self):
        """Hide pair results and downloads from staff in another workgroup."""
        other_workgroup_user = CaidUserFactory()
        result = IdentificationSimilarPairResult.objects.create(workgroup=other_workgroup_user.workgroup)
        self.assertEqual(
            self.client.get(reverse("caidapp:similar_identity_pairs_result", args=[result.id])).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                reverse("caidapp:download_similar_identity_pairs_csv", args=[result.id, "input"])
            ).status_code,
            404,
        )

    def test_worker_error_is_reported_to_staff(self):
        """Store worker error details and notify staff without notifying regular members."""
        result = IdentificationSimilarPairResult.objects.create(workgroup=self.workgroup)
        tasks.similar_identity_pairs_on_success({"status": "ERROR", "error": "No embeddings"}, result_id=result.id)
        result.refresh_from_db()
        self.assertEqual(result.status, "error")
        self.assertIn("No embeddings", result.message)
        self.assertEqual(NotificationRecipient.objects.filter(user=self.staff).count(), 1)
        self.assertFalse(NotificationRecipient.objects.filter(user=self.nonstaff).exists())

    def test_outlier_completion_notification_reaches_only_staff(self):
        """Send outlier completion notifications only to staff with a link to the saved result."""
        result = IdentificationOutlierSuggestionResult.objects.create(workgroup=self.workgroup, suggestions=[])
        tasks.identification_outlier_detection_on_success(
            {"status": "DONE", "message": "finished", "suggestions": []},
            result_id=result.id,
        )
        recipients = NotificationRecipient.objects.select_related("notification").all()
        self.assertEqual(recipients.count(), 1)
        self.assertEqual(recipients[0].user_id, self.staff.id)
        self.assertEqual(
            recipients[0].notification.get_link_url(),
            reverse("caidapp:identification_outlier_suggestions_result", args=[result.id]),
        )
