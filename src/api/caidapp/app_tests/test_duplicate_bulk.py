"""Explicit bulk confirmations, metadata completeness, authorization and rollback."""

from unittest.mock import patch

from caidapp import duplicate_services as service
from caidapp import models
from django.test import TestCase
from django.urls import reverse

from . import test_duplicates as fixtures
from .factories import AnimalObservationFactory, MediaFileFactory, TaxonFactory


class DuplicateBulkTest(TestCase):
    setUp = fixtures.DuplicateReviewTest.setUp
    media = fixtures.DuplicateReviewTest.media
    proposal = fixtures.DuplicateReviewTest.proposal

    def group_proposal(self, mediafiles, kind="pixel_exact"):
        """Make an authorized durable group suggestion using real source bytes."""
        proposal = self.proposal(mediafiles, kind=kind)
        proposal.scan.workgroup = self.user.workgroup
        proposal.scan.save(update_fields=["workgroup"])
        return proposal

    def preview(self, proposals):
        """Request selected whole-group deletion without applying it."""
        return self.client.post(
            reverse("caidapp:duplicate_bulk_review"),
            {"action": "selected", "proposals": [str(p.uuid) for p in proposals]},
        )

    def confirm(self, response):
        """Confirm exactly the server-signed preview."""
        with patch("caidapp.tasks.cleanup_duplicate_files.delay"):
            return self.client.post(
                reverse("caidapp:duplicate_bulk_review"), {"action": "apply", "token": response.context["token"]}
            )

    def test_difference_count_counts_fields_not_pairs_and_does_not_hash_sources(self):
        """A triplicate with three different notes has one differing field."""
        proposal = self.group_proposal([self.media(note=value) for value in ["a", "b", "c"]])
        with patch("caidapp.duplicate_services.source_hash", side_effect=AssertionError("Unexpected hashing")):
            summary = service.proposal_summary(proposal)
        self.assertEqual(summary["difference_count"], 1)
        self.assertFalse(summary["perfect"])
        self.assertEqual(sum(m["role"] == "delete" for m in summary["members"]), 2)

    def test_hidden_annotation_and_metadata_differences_are_not_perfect(self):
        """Zero visible fields never authorizes perfect-only deletion of different object data."""
        for difference in ["bbox", "verified", "metadata", "album", "cover"]:
            with self.subTest(difference=difference):
                a, b = self.media(), self.media()
                taxon = TaxonFactory()
                AnimalObservationFactory(mediafile=a, taxon=taxon)
                observation = AnimalObservationFactory(mediafile=b, taxon=taxon)
                if difference == "bbox":
                    observation.bbox_x_center = 0.3
                    observation.save(update_fields=["bbox_x_center"])
                elif difference == "verified":
                    observation.taxon_verified = False
                    observation.save(update_fields=["taxon_verified"])
                elif difference == "metadata":
                    b.metadata_json = {"camera": "extra"}
                    b.save(update_fields=["metadata_json"])
                elif difference == "album":
                    album = models.Album.objects.create(name="Source only", owner=self.user)
                    album.mediafiles.add(b)
                else:
                    album = models.Album.objects.create(name="Source cover", owner=self.user, cover=b)
                    album.mediafiles.add(a, b)
                summary = service.proposal_summary(self.group_proposal([a, b]))
                self.assertEqual(summary["difference_count"], 0)
                self.assertFalse(summary["perfect"])

    def test_preview_then_confirm_triplicate(self):
        """Preview changes nothing; confirmation removes both red files and preserves the green one."""
        a, b, c = [self.media() for _ in range(3)]
        proposal = self.group_proposal([a, b, c])
        before = models.MediaFile.objects.filter(pk=a.pk).values().get()
        preview = self.preview([proposal])
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.context["remove_count"], 2)
        self.assertEqual(models.MediaFile.objects.filter(pk__in=[a.pk, b.pk, c.pk]).count(), 3)
        self.confirm(preview)
        self.assertEqual(models.MediaFile.objects.filter(pk=a.pk).values().get(), before)
        self.assertFalse(models.MediaFile.objects.filter(pk__in=[b.pk, c.pk]).exists())
        proposal.refresh_from_db()
        self.assertEqual(proposal.status, "resolved")

    def test_stale_second_group_rolls_back_entire_batch(self):
        """No first-group deletion or cleanup intent may survive a later conflict."""
        files = [self.media() for _ in range(4)]
        first = self.group_proposal(files[:2])
        second = self.group_proposal(files[2:])
        preview = self.preview([first, second])
        models.MediaFile.objects.filter(pk=files[-1].pk).update(note="Changed after preview")
        self.confirm(preview)
        self.assertEqual(models.MediaFile.objects.filter(pk__in=[m.pk for m in files]).count(), 4)
        self.assertFalse(models.DuplicateFileCleanup.objects.exists())
        first.refresh_from_db()
        self.assertEqual(first.status, "pending")

    def test_perfect_action_skips_imperfect_and_video_reference_groups(self):
        """Perfect selection examines the group catalogue without deleting or including reference-only matches."""
        perfect = self.group_proposal([self.media(), self.media()])
        self.group_proposal([self.media(note="first"), self.media(note="second")])
        self.group_proposal([self.media(), self.media()], kind="reference_exact")
        response = self.client.post(reverse("caidapp:duplicate_bulk_review"), {"action": "perfect"})
        self.assertEqual([row["proposal"].pk for row in response.context["previews"]], [perfect.pk])
        self.assertEqual(response.context["skipped"], 2)
        self.assertEqual(models.MediaFile.objects.count(), 6)

    def test_foreign_selection_and_tampered_confirmation_cannot_delete(self):
        """Both preview scope and signed confirmation are authorization boundaries."""
        foreign = MediaFileFactory()
        hidden = self.group_proposal([self.media(), foreign])
        response = self.preview([hidden])
        self.assertEqual(response.status_code, 302)
        response = self.client.post(reverse("caidapp:duplicate_bulk_review"), {"action": "apply", "token": "forged"})
        self.assertEqual(response.status_code, 302)
        self.assertTrue(models.MediaFile.objects.filter(pk=foreign.pk).exists())

    def test_overlapping_proposals_are_not_double_deleted(self):
        """One source may appear in at most one reviewed group within a batch."""
        a, b, c = self.media(), self.media(), self.media()
        response = self.preview([self.group_proposal([a, b]), self.group_proposal([b, c])])
        self.assertEqual(len(response.context["previews"]), 1)
        self.assertEqual(response.context["skipped"], 1)

    def test_batch_limit_is_reported_in_preview(self):
        """Bound the transaction and explicitly report groups remaining beyond the batch."""
        proposals = [self.group_proposal([self.media(), self.media()]) for _ in range(2)]
        with patch("caidapp.duplicate_services.MAX_BULK_PROPOSALS", 1):
            preview = service.bulk_preview(proposals, self.user, perfect_only=True)
        self.assertEqual(len(preview["previews"]), 1)
        self.assertTrue(preview["limited"])

    def test_album_cover_changes_after_preview_reject_deletion(self):
        """Changing an external reference invalidates the reviewed deletion snapshot."""
        a, b = self.media(), self.media()
        proposal = self.group_proposal([a, b])
        preview = self.preview([proposal])
        models.Album.objects.create(name="New cover", owner=self.user, cover=b)
        self.confirm(preview)
        self.assertEqual(models.MediaFile.objects.filter(pk__in=[a.pk, b.pk]).count(), 2)

    def test_list_displays_roles_upload_and_info_without_visible_ids(self):
        """Catalogue display agrees with score-based preselection and exposes copyable metadata."""
        self.group_proposal([self.media(), self.media(note="richer")])
        with patch("caidapp.duplicate_hashes.recover_group"):
            response = self.client.get(reverse("caidapp:duplicates"))
        self.assertContains(response, 'class="mediafile-info-trigger"')
        self.assertContains(response, f"Upload: {self.archive.name}")
        self.assertContains(response, "text-success")
        self.assertContains(response, 'name="proposals"')
        self.assertNotContains(response, "(#")
