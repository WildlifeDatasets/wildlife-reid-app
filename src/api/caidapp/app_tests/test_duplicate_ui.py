"""Compact processing status and identity merge offers within duplicate groups."""

from unittest.mock import patch

from caidapp import duplicate_hashes, duplicate_services, models, views_duplicates
from django.test import TestCase
from django.urls import NoReverseMatch, reverse

from . import test_duplicates as fixtures
from .factories import AnimalObservationFactory, IndividualIdentityFactory


class DuplicateInterfaceTest(TestCase):
    setUp = fixtures.DuplicateReviewTest.setUp
    media = fixtures.DuplicateReviewTest.media
    proposal = fixtures.DuplicateReviewTest.proposal

    def group_proposal(self, rows, kind="pixel_exact"):
        proposal = self.proposal(rows, kind=kind)
        proposal.scan.workgroup = self.user.workgroup
        proposal.scan.save(update_fields=["workgroup"])
        return proposal

    def copies(self, different_boxes=False):
        first, second = self.media(), self.media()
        a = IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        b = IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        box = dict(bbox_x_center=0.3, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2)
        obs_a = AnimalObservationFactory(mediafile=first, identity=a, **box)
        if different_boxes:
            box["bbox_x_center"] = 0.7
        obs_b = AnimalObservationFactory(mediafile=second, identity=b, **box)
        return first, second, obs_a, obs_b

    def test_ring_handles_complete_partial_empty_and_failed_groups(self):
        for total, ready, failed, label, pending in [
            (8969, 8969, 0, "100%", 0),
            (10, 8, 1, "80%", 1),
            (10, 0, 10, "0%", 0),
            (0, 0, 0, "—", 0),
        ]:
            with self.subTest(total=total, ready=ready, failed=failed):
                summary = views_duplicates._processing_summary(total, ready, failed)
                self.assertEqual(summary["label"], label)
                self.assertEqual(summary["pending"], pending)
                ready_dash = float(summary["ready_dash"].split()[0])
                failed_dash = float(summary["error_dash"].split()[0])
                self.assertLessEqual(ready_dash + failed_dash, 100)
                self.assertNotIn("nan", str(summary).lower())

    def test_compact_summary_keeps_counts_and_removes_category_switches(self):
        ready, failed = self.media(), self.media()
        self.media()
        # Source-file factory hooks invalidate hashes; set the completed worker states afterwards.
        models.MediaFile.objects.filter(pk=ready.pk).update(
            hash_status="ready", hash_version=duplicate_hashes.HASH_VERSION
        )
        models.MediaFile.objects.filter(pk=failed.pk).update(hash_status="error")
        with patch("caidapp.duplicate_hashes.recover_group"):
            response = self.client.get(reverse("caidapp:duplicates"))
        self.assertEqual(response.context["summary_stats"]["total"], 3)
        self.assertEqual(response.context["summary_stats"]["ready"], 1)
        self.assertEqual(response.context["summary_stats"]["errors"], 1)
        self.assertEqual(response.context["summary_stats"]["pending"], 1)
        self.assertContains(response, 'class="duplicate-summary-ring"')
        self.assertContains(response, 'role="tooltip"')
        self.assertContains(response, "media files in completed imports")
        self.assertContains(response, "with current fingerprints")
        self.assertContains(response, "failed inputs")
        self.assertNotContains(response, 'aria-label="Suggestion category"')
        self.assertNotContains(response, "Identities sharing an image")
        self.assertNotContains(response, ">Media file duplicates</a>")

    def test_identity_offer_is_in_list_and_review_without_merging(self):
        a, b, obs_a, obs_b = self.copies()
        proposal = self.group_proposal([a, b])
        before = models.IndividualIdentity.objects.count()
        with patch("caidapp.duplicate_hashes.recover_group"):
            listing = self.client.get(reverse("caidapp:duplicates"))
        review = self.client.get(reverse("caidapp:duplicate_review", args=[proposal.uuid]))
        pair = review.context["identity_merges"][0]
        self.assertEqual({pair["source"].pk, pair["target"].pk}, {obs_a.identity_id, obs_b.identity_id})
        url = reverse("caidapp:merge_identities", args=[pair["source"].pk, pair["target"].pk])
        self.assertContains(listing, url)
        self.assertContains(review, url)
        self.assertContains(review, "Review identity merge")
        preview = self.client.get(url, {"next": reverse("caidapp:duplicate_review", args=[proposal.uuid])})
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(models.IndividualIdentity.objects.count(), before)

    def test_different_animals_in_copies_do_not_offer_a_merge(self):
        a, b, _, _ = self.copies(different_boxes=True)
        self.assertEqual(duplicate_services.identity_merge_candidates(self.group_proposal([a, b])), [])

    def test_same_identity_needs_no_merge(self):
        a, b, first, second = self.copies()
        second.identity = first.identity
        second.save(update_fields=["identity"])
        self.assertEqual(duplicate_services.identity_merge_candidates(self.group_proposal([a, b])), [])

    def test_missing_boxes_do_not_infer_an_identity_match(self):
        a, b, first, _ = self.copies()
        first.bbox_width = None
        first.save(update_fields=["bbox_width"])
        self.assertEqual(duplicate_services.identity_merge_candidates(self.group_proposal([a, b])), [])

    def test_foreign_and_excluded_identities_are_not_offered(self):
        a, b, first, second = self.copies()
        proposal = self.group_proposal([a, b])
        pair = sorted([first.identity_id, second.identity_id])
        models.MergeIdentitySuggestionExclusion.objects.create(
            workgroup=self.user.workgroup, identity_a_id=pair[0], identity_b_id=pair[1]
        )
        self.assertEqual(duplicate_services.identity_merge_candidates(proposal), [])
        models.MergeIdentitySuggestionExclusion.objects.all().delete()
        second.identity = IndividualIdentityFactory()
        second.save(update_fields=["identity"])
        self.assertEqual(duplicate_services.identity_merge_candidates(proposal), [])

    def test_video_reference_and_legacy_private_scans_do_not_offer_merges(self):
        a, b, _, _ = self.copies()
        self.assertEqual(
            duplicate_services.identity_merge_candidates(self.group_proposal([a, b], kind="reference_exact")), []
        )
        self.assertEqual(duplicate_services.identity_merge_candidates(self.proposal([a, b])), [])

    def test_review_offers_only_identities_from_the_selected_pair(self):
        a, b, first, second = self.copies()
        third = self.media()
        other = AnimalObservationFactory(
            mediafile=third, identity=IndividualIdentityFactory(owner_workgroup=self.user.workgroup),
            bbox_x_center=0.3, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2,
        )
        proposal = self.group_proposal([a, b, third])
        response = self.client.get(
            reverse("caidapp:duplicate_review", args=[proposal.uuid]),
            {"keep": a.pk, "subset": "1", "remove": b.pk},
        )
        pairs = response.context["identity_merges"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0]["source"].pk, pairs[0]["target"].pk}, {first.identity_id, second.identity_id})
        self.assertNotIn(other.identity_id, {pairs[0]["source"].pk, pairs[0]["target"].pk})

    def test_obsolete_identity_view_is_removed(self):
        with self.assertRaises(NoReverseMatch):
            reverse("caidapp:duplicate_identity_suggestions")
        url = reverse("caidapp:duplicates").replace("duplicates/", "duplicate-identities/")
        self.assertEqual(self.client.get(url).status_code, 404)
