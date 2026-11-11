"""Destructive duplicate review: authorization, stale previews, groups, and observations."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from caidapp import duplicate_services as service
from caidapp import models
from django.core import signing
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .factories import (
    AnimalObservationFactory,
    IndividualIdentityFactory,
    LocalityFactory,
    MediaFileFactory,
    TaxonFactory,
    UploadedArchiveFactory,
)


class DuplicateReviewTest(TestCase):
    def test_review_includes_all_valid_observation_boxes_for_each_image(self):
        """Draw multiple objects and zero-valued coordinates, excluding boxless observations."""
        a, b = self.media(), self.media()
        first = dict(bbox_x_center=0.0, bbox_y_center=0.2, bbox_width=0.2, bbox_height=0.3)
        second = dict(bbox_x_center=0.6, bbox_y_center=0.5, bbox_width=0.1, bbox_height=0.2)
        AnimalObservationFactory(mediafile=a, **first)
        AnimalObservationFactory(mediafile=a, **second)
        AnimalObservationFactory(mediafile=a, bbox_width=None)
        AnimalObservationFactory(mediafile=b, **second)
        proposal = self.proposal([a, b])
        response = self.client.get(reverse("caidapp:duplicate_review", args=[proposal.uuid]))
        self.assertEqual(response.context["bbox_data"][a.pk], [[0.0, 0.2, 0.2, 0.3], [0.6, 0.5, 0.1, 0.2]])
        self.assertEqual(response.context["bbox_data"][b.pk], [[0.6, 0.5, 0.1, 0.2]])
        self.assertContains(response, 'id="duplicate-bboxes"')

    def test_equal_metadata_prefers_longer_original_filename(self):
        """Prefer potentially informative filenames after matching metadata scores."""
        short = self.media(original_filename="image.jpg")
        long = self.media(original_filename="image_salamandra_locality_2026.jpg")
        proposal = self.proposal([short, long])
        response = self.client.get(reverse("caidapp:duplicate_review", args=[proposal.uuid]))
        self.assertEqual(response.context["plan"]["keep_id"], long.pk)
        self.assertEqual(service.proposal_summary(proposal)["keep_id"], long.pk)

    def test_filename_length_does_not_override_richer_metadata_or_manual_choice(self):
        """Metadata quality and an explicit user choice remain more important than filename length."""
        short = self.media(original_filename="a.jpg", note="Field information")
        long = self.media(original_filename="much_longer_original_filename.jpg")
        proposal = self.proposal([short, long])
        self.assertEqual(service.preferred_media([short, long]).pk, short.pk)
        response = self.client.get(reverse("caidapp:duplicate_review", args=[proposal.uuid]), {"keep": long.pk})
        self.assertEqual(response.context["plan"]["keep_id"], long.pk)

    def test_default_retains_the_file_with_more_metadata(self):
        """Choose richer metadata even when that file is not the first group member."""
        a, b = self.media(), self.media(note="Field note", locality=LocalityFactory(owner=self.user))
        proposal = self.proposal([a, b])
        response = self.client.get(reverse("caidapp:duplicate_review", args=[proposal.uuid]))
        self.assertEqual(response.context["plan"]["keep_id"], b.pk)
        self.assertEqual(response.context["plan"]["remove_ids"], [a.pk])
        self.assertContains(response, f"Upload: {self.archive.name}")

    def test_verified_taxon_and_representative_identity_count_as_three_facts(self):
        """Reviewed labels outweigh one extra ordinary metadata value."""
        for field in ["taxon_verified", "identity_is_representative"]:
            with self.subTest(field=field):
                a, b = self.media(note="One extra fact"), self.media()
                AnimalObservationFactory(
                    mediafile=a,
                    identity=IndividualIdentityFactory(owner_workgroup=self.user.workgroup),
                    taxon_verified=False,
                    identity_is_representative=False,
                )
                AnimalObservationFactory(
                    mediafile=b,
                    identity=IndividualIdentityFactory(owner_workgroup=self.user.workgroup),
                    **{"taxon_verified": False, "identity_is_representative": False, field: True},
                )
                self.assertEqual(service.retention_score(b) - service.retention_score(a), 1)
                proposal = self.proposal([a, b])
                response = self.client.get(reverse("caidapp:duplicate_review", args=[proposal.uuid]))
                self.assertEqual(response.context["plan"]["keep_id"], b.pk)

    def test_manual_survivor_overrides_information_score(self):
        """Keep explicit choices, including the survivor selected for a subsequent pair."""
        a, b = self.media(), self.media(note="More information")
        proposal = self.proposal([a, b])
        response = self.client.get(reverse("caidapp:duplicate_review", args=[proposal.uuid]), {"keep": a.pk})
        self.assertEqual(response.context["plan"]["keep_id"], a.pk)
        swapped = self.client.get(response.context["swap_url"])
        self.assertEqual(swapped.context["plan"]["keep_id"], b.pk)

    def apply_edits(self, proposal, kept, edits):
        """Apply explicit edits against a freshly signed preview."""
        plan = service.build_plan(proposal, self.user, kept.pk)
        with patch("caidapp.tasks.cleanup_duplicate_files.delay"):
            return service.apply_merge(proposal.pk, self.user, service.preview_token(proposal, self.user, plan), edits)

    def test_explicit_replacements_and_note_preserve_retained_geometry_and_previews(self):
        """Save selected replacements without changing geometry or preview ownership."""
        a, b = self.media(note="old"), self.media(note="source")
        b.locality = LocalityFactory(owner=self.user)
        b.captured_at = self.when.replace(year=self.when.year - 1)
        b.save(update_fields=["locality", "captured_at"])
        kept = AnimalObservationFactory(
            mediafile=a, taxon=TaxonFactory(), identity=IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        )
        source = AnimalObservationFactory(
            mediafile=b, taxon=TaxonFactory(), identity=IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        )
        before_geometry = [getattr(kept, field) for field in service.BBOX_FIELDS]
        before_files = [getattr(a, field).name or "" for field in service.FILE_FIELDS]
        proposal = self.proposal([a, b])
        edits = {
            "note": "My combined note",
            "take": {field: b.pk for field in ["identities", "taxa", "locality", "captured_at"]},
        }
        self.apply_edits(proposal, a, edits)
        a.refresh_from_db()
        kept.refresh_from_db()
        self.assertEqual(a.note, "My combined note")
        self.assertEqual(a.locality_id, b.locality_id)
        self.assertEqual(a.captured_at, b.captured_at)
        self.assertEqual(kept.identity_id, source.identity_id)
        self.assertEqual(kept.taxon_id, source.taxon_id)
        self.assertTrue(kept.taxon_verified)
        self.assertEqual(kept.updated_by_id, self.user.pk)
        self.assertEqual([getattr(kept, field) for field in service.BBOX_FIELDS], before_geometry)
        self.assertEqual([getattr(a, field).name for field in service.FILE_FIELDS], before_files)
        proposal.refresh_from_db()
        self.assertEqual(proposal.decision["merges"][0]["edits"], edits)

    def test_multiple_observations_block_quick_identity_and_taxon_replacement(self):
        """Require manual observation editing when either file has multiple objects."""
        for multiple_on_source in [False, True]:
            with self.subTest(multiple_on_source=multiple_on_source):
                a, b = self.media(), self.media()
                for media in [a, b, b if multiple_on_source else a]:
                    AnimalObservationFactory(
                        mediafile=media, identity=IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
                    )
                proposal = self.proposal([a, b])
                plan = service.build_plan(proposal, self.user, a.pk)
                self.assertFalse(any(option["field"] in {"identities", "taxa"} for option in plan["transfer_options"]))
                with self.assertRaises(ValueError):
                    self.apply_edits(proposal, a, {"take": {"identities": b.pk}})
                self.assertTrue(models.MediaFile.objects.filter(pk=b.pk).exists())

    def test_placeholder_is_not_a_transfer_target(self):
        """Do not assign labels to a no-detection placeholder through quick replacement."""
        a, b = self.media(), self.media()
        AnimalObservationFactory(mediafile=b, identity=IndividualIdentityFactory(owner_workgroup=self.user.workgroup))
        proposal = self.proposal([a, b])
        with self.assertRaises(ValueError):
            self.apply_edits(proposal, a, {"take": {"taxa": b.pk}})
        self.assertTrue(a.observations.get().is_no_detection_placeholder)

    def test_forged_replacement_and_stale_note_edits_cannot_delete(self):
        """Reject unreviewed sources, unsupported fields and concurrent edits atomically."""
        a, b, unrelated = self.media(), self.media(), self.media()
        proposal = self.proposal([a, b])
        with self.assertRaises(ValueError):
            self.apply_edits(proposal, a, {"take": {"locality": unrelated.pk}})
        with self.assertRaises(ValueError):
            self.apply_edits(proposal, a, {"take": {"sequence": b.pk}})
        plan = service.build_plan(proposal, self.user, a.pk)
        token = service.preview_token(proposal, self.user, plan)
        models.MediaFile.objects.filter(pk=a.pk).update(note="Changed elsewhere")
        with self.assertRaises(ValueError):
            service.apply_merge(proposal.pk, self.user, token, {"note": "Stale edit"})
        self.assertTrue(models.MediaFile.objects.filter(pk=b.pk).exists())

    def test_html_note_and_selected_source_are_saved_together(self):
        """Accept the external-form note editor and explicitly selected source inputs."""
        a, b = self.media(note="first"), self.media(note="second", location="50,14")
        proposal = self.proposal([a, b])
        url = reverse("caidapp:duplicate_review", args=[proposal.uuid])
        response = self.client.get(url, {"keep": a.pk})
        self.assertContains(response, 'id="retained-note"')
        self.assertContains(response, 'name="take_location"')
        with patch("caidapp.tasks.cleanup_duplicate_files.delay"):
            response = self.client.post(
                url, {"token": response.context["token"], "note": "Edited", "take_location": b.pk}
            )
        self.assertEqual(response.status_code, 302)
        a.refresh_from_db()
        self.assertEqual(a.note, "Edited")
        self.assertEqual(a.location, "50,14")

    def test_sequence_difference_is_not_a_warning(self):
        """Do not distract from meaningful differences with sequence membership."""
        from .factories import SequenceFactory

        a, b = self.media(), self.media(sequence=SequenceFactory())
        proposal = self.proposal([a, b])
        plan = service.build_plan(proposal, self.user, a.pk)
        self.assertNotIn("sequence", {row["field"] for row in plan["warnings"]})

    def test_locality_can_be_explicitly_cleared_or_filled_from_other_file(self):
        """Offer both directions of an empty locality and persist the selected source's value."""
        locality = LocalityFactory(owner=self.user)
        for source_empty in [True, False]:
            with self.subTest(source_empty=source_empty):
                kept = self.media(locality=locality if source_empty else None)
                source = self.media(locality=None if source_empty else locality)
                proposal = self.proposal([kept, source])
                url = reverse("caidapp:duplicate_review", args=[proposal.uuid])
                response = self.client.get(url, {"keep": kept.pk})
                options = response.context["plan"]["transfer_options"]
                option = next(row for row in options if row["field"] == "locality")
                self.assertEqual(option["source_id"], source.pk)
                self.assertEqual(option["value"], "—" if source_empty else str(locality))
                self.assertContains(response, 'name="take_locality"')
                with patch("caidapp.tasks.cleanup_duplicate_files.delay"):
                    self.client.post(url, {"token": response.context["token"], "take_locality": source.pk})
                kept.refresh_from_db()
                self.assertEqual(kept.locality_id, source.locality_id)
                self.assertFalse(models.MediaFile.objects.filter(pk=source.pk).exists())

    def test_selected_edits_roll_back_if_deletion_fails(self):
        """Never retain an edit without its reviewed deletion when a transaction fails."""
        a, b = self.media(note="original"), self.media(note="source")
        proposal = self.proposal([a, b])
        with patch("caidapp.bbox_services._invalidate_reid", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                self.apply_edits(proposal, a, {"note": "edited"})
        a.refresh_from_db()
        self.assertEqual(a.note, "original")
        self.assertTrue(models.MediaFile.objects.filter(pk=b.pk).exists())

    def setUp(self):
        """Use real temporary source files and one authorized upload."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.override = override_settings(MEDIA_ROOT=directory.name, DUPLICATE_SCAN_ROOT=Path(directory.name) / "scans")
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.root = Path(directory.name)
        self.archive = UploadedArchiveFactory()
        self.user = self.archive.owner
        self.client.force_login(self.user.user)
        self.when = timezone.now()

    def media(self, **kwargs):
        """Create an image with stable acquisition metadata."""
        return MediaFileFactory(parent=self.archive, captured_at=self.when, **kwargs)

    def proposal(self, mediafiles, kind="pixel_exact", evidence=None):
        """Create evidence matching the current source content."""
        scan = models.DuplicateScan.objects.create(requested_by=self.user, status="succeeded")
        proposal = models.DuplicateProposal.objects.create(scan=scan, kind=kind, evidence=evidence or {})
        for position, media in enumerate(mediafiles):
            models.DuplicateProposalMember.objects.create(
                proposal=proposal,
                mediafile=media,
                original_id=media.pk,
                position=position,
                snapshot={
                    "source": media.mediafile.name,
                    "sha256": service.source_hash(media.mediafile.name),
                    "name": media.original_filename,
                    "size": [100, 100],
                },
            )
        return proposal

    def merge(self, proposal, kept, removed=None):
        """Apply the reviewed whole-file deletion without transferring any metadata."""
        plan = service.build_plan(proposal, self.user, kept.pk, removed)
        with patch("caidapp.tasks.cleanup_duplicate_files.delay"):
            service.apply_merge(proposal.pk, self.user, service.preview_token(proposal, self.user, plan))

    def test_triplicate_retains_chosen_file_and_deletes_other_observations(self):
        """The selected media file remains identical; both other files and their observations disappear."""
        a = self.media(note="first", metadata_json={"a": 1})
        b = self.media(note="second", metadata_json={"b": 2})
        c = self.media()
        first = AnimalObservationFactory(mediafile=b, taxon=TaxonFactory())
        second = AnimalObservationFactory(mediafile=c, taxon=TaxonFactory())
        proposal = self.proposal([a, b, c])
        before = models.MediaFile.objects.filter(pk=a.pk).values().get()
        kept_observations = list(a.observations.values())
        self.merge(proposal, a)
        proposal.refresh_from_db()
        self.assertEqual(proposal.status, "resolved")
        self.assertEqual(models.MediaFile.objects.filter(pk=a.pk).values().get(), before)
        self.assertEqual(list(a.observations.values()), kept_observations)
        self.assertFalse(models.AnimalObservation.objects.filter(pk__in=[first.pk, second.pk]).exists())
        self.assertFalse(models.MediaFile.objects.filter(pk__in=[b.pk, c.pk]).exists())
        self.assertEqual(proposal.members.filter(mediafile__isnull=True).count(), 2)
        self.assertTrue(proposal.decision["merges"][0]["before"]["observations"])

    def test_partial_triplicate_keeps_remaining_proposal_open(self):
        """Removing one member preserves the remaining exact group for the next action."""
        a, b, c = [self.media() for _ in range(3)]
        proposal = self.proposal([a, b, c])
        self.merge(proposal, a, [b.pk])
        proposal.refresh_from_db()
        self.assertEqual(proposal.status, "pending")
        self.assertEqual(proposal.members.filter(mediafile__isnull=False).count(), 2)
        self.merge(proposal, a)
        proposal.refresh_from_db()
        self.assertEqual(proposal.status, "resolved")
        self.assertEqual(len(proposal.decision["merges"]), 2)

    def test_external_delete_invalidates_crop_but_preserves_exact_remainder(self):
        """Never redirect a crop proposal to an unverified replacement file."""
        a, b, c = [self.media() for _ in range(3)]
        group = self.proposal([a, b, c])
        crop = self.proposal([b, c], "crop")
        b.delete()
        crop.refresh_from_db()
        group.refresh_from_db()
        self.assertEqual(crop.status, "stale")
        self.assertEqual(group.status, "pending")
        c.delete()
        group.refresh_from_db()
        self.assertEqual(group.status, "stale")

    def test_concurrent_annotation_edit_rejects_preview(self):
        """An old signed preview must never authorize deletion after an annotation edit."""
        a, b = self.media(), self.media()
        proposal = self.proposal([a, b])
        plan = service.build_plan(proposal, self.user, a.pk)
        token = service.preview_token(proposal, self.user, plan)
        observation = b.observations.get()
        observation.metadata_json = {"new": "reviewed"}
        observation.save()
        with self.assertRaisesMessage(ValueError, "changed"):
            service.apply_merge(proposal.pk, self.user, token)
        self.assertTrue(models.MediaFile.objects.filter(pk=b.pk).exists())

    def test_changed_file_rejects_merge(self):
        """A file overwritten in place cannot reuse stale duplicate evidence."""
        a, b = self.media(), self.media()
        proposal = self.proposal([a, b])
        (self.root / b.mediafile.name).write_bytes(b"changed")
        with self.assertRaisesMessage(ValueError, "source media file changed"):
            service.build_plan(proposal, self.user, a.pk)

    def test_metadata_differences_never_merge_into_retained_file(self):
        """Extra metadata in the removed file is not copied into the survivor."""
        a = self.media(metadata_json={"camera": "A"})
        b = self.media(metadata_json={"camera": "B", "additional": "source-only"})
        proposal = self.proposal([a, b])
        plan = service.build_plan(proposal, self.user, a.pk)
        self.assertNotIn("conflicts", plan)
        self.merge(proposal, a)
        a.refresh_from_db()
        self.assertEqual(a.metadata_json, {"camera": "A"})

    def test_crop_removal_warns_and_does_not_transfer_outside_observations(self):
        """Whole-file deletion is explicit even for legacy crop suggestions."""
        crop, original = self.media(), self.media()
        observation = AnimalObservationFactory(
            mediafile=original, bbox_x_center=0.9, bbox_y_center=0.9, bbox_width=0.1, bbox_height=0.1
        )
        proposal = self.proposal([crop, original], "crop")
        plan = service.build_plan(proposal, self.user, crop.pk)
        self.assertTrue(any(row["field"] == "crop" for row in plan["warnings"]))
        self.merge(proposal, crop)
        self.assertFalse(models.AnimalObservation.objects.filter(pk=observation.pk).exists())
        self.assertTrue(crop.observations.get().is_no_detection_placeholder)

    def test_removed_crop_observations_are_not_transformed_or_copied(self):
        """The retained original does not receive anonymous observations from the deleted crop."""
        crop, original = self.media(), self.media()
        observation = AnimalObservationFactory(
            mediafile=crop, bbox_x_center=0.5, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2
        )
        proposal = self.proposal([crop, original], "crop")
        before = list(original.observations.values())
        self.merge(proposal, original)
        self.assertFalse(models.AnimalObservation.objects.filter(pk=observation.pk).exists())
        self.assertEqual(list(original.observations.values()), before)

    def test_permanent_cleanup_preserves_shared_paths(self):
        """Post-commit cleanup deletes only paths without remaining media-file references."""
        a, b = self.media(), self.media()
        shared_name = a.mediafile.name
        original_name = b.mediafile.name
        b.image_file = a.mediafile.name
        b.save()
        proposal = self.proposal([a, b])
        self.merge(proposal, a)
        intents = list(models.DuplicateFileCleanup.objects.values_list("pk", flat=True))
        service.cleanup_files(intents)
        self.assertTrue((self.root / shared_name).exists())
        self.assertFalse((self.root / original_name).exists())
        self.assertTrue(models.DuplicateFileCleanup.objects.filter(storage_name=shared_name, status="shared").exists())

    def test_review_page_and_api_hide_inaccessible_counterpart(self):
        """A proposal owned by the user must not expose a now-inaccessible member."""
        a, b = self.media(), MediaFileFactory()
        proposal = self.proposal([a, b])
        response = self.client.get(reverse("caidapp:duplicate_review", args=[proposal.uuid]))
        self.assertEqual(response.status_code, 404)
        response = self.client.get(reverse("caidapp:duplicate_scan_matches", args=[proposal.scan.uuid]))
        self.assertEqual(response.json()["count"], 0)

    def test_review_renders_and_post_advances(self):
        """The browser workflow renders a preview, merges, and opens the next suggestion."""
        a, b = self.media(), self.media()
        proposal = self.proposal([a, b])
        url = reverse("caidapp:duplicate_review", args=[proposal.uuid])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Swap direction")
        self.assertContains(response, "DELETE THIS MEDIA FILE")
        self.assertContains(response, "RETAIN THIS MEDIA FILE")
        with patch("caidapp.tasks.cleanup_duplicate_files.delay"):
            response = self.client.post(url, {"token": response.context["token"]})
        self.assertRedirects(response, reverse("caidapp:duplicate_scan", args=[proposal.scan.uuid]))

    def test_scan_callback_is_durable_and_idempotent(self):
        """Duplicate callbacks must not create repeated suggestions."""
        a, b = self.media(), self.media()
        with patch("caidapp.duplicate_services.dispatch_scan"):
            scan = service.create_scan(self.user, [a.pk, b.pk], False)
        directory = service.scan_directory(scan)
        directory.mkdir(parents=True)
        records = [{**item, "sha256": service.source_hash(item["source"])} for item in scan.items]
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "version": "duplicates-v1",
                    "items": records,
                    "proposals": [{"kind": "pixel_exact", "members": [a.pk, b.pk], "evidence": {}}],
                }
            ),
            encoding="utf-8",
        )
        service.finish_scan(scan.pk)
        service.finish_scan(scan.pk)
        scan.refresh_from_db()
        self.assertEqual(scan.status, "succeeded")
        self.assertEqual(scan.proposals.count(), 1)

    def test_dismiss_after_partial_merge_keeps_audit(self):
        """Dismissing a remainder must not erase the completed destructive action."""
        a, b, c = [self.media() for _ in range(3)]
        proposal = self.proposal([a, b, c])
        self.merge(proposal, a, [b.pk])
        service.dismiss_proposal(proposal.pk, self.user)
        proposal.refresh_from_db()
        self.assertEqual(proposal.status, "dismissed")
        self.assertEqual(proposal.decision["merges"][0]["removed"], [b.pk])

    def test_identity_and_taxon_differences_warn_without_editing_retained_annotations(self):
        """Different labels are warnings; the selected file keeps its own identity and verification."""
        a, b = self.media(), self.media()
        first_identity = IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        second_identity = IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        first_taxon, second_taxon = TaxonFactory(), TaxonFactory()
        kept = AnimalObservationFactory(mediafile=a, identity=first_identity, taxon=first_taxon, taxon_verified=True)
        removed = AnimalObservationFactory(mediafile=b, identity=second_identity, taxon=second_taxon)
        proposal = self.proposal([a, b])
        plan = service.build_plan(proposal, self.user, a.pk)
        self.assertTrue({"identities", "taxa"}.issubset({row["field"] for row in plan["warnings"]}))
        before = models.AnimalObservation.objects.filter(pk=kept.pk).values().get()
        self.merge(proposal, a)
        self.assertEqual(models.AnimalObservation.objects.filter(pk=kept.pk).values().get(), before)
        self.assertFalse(models.AnimalObservation.objects.filter(pk=removed.pk).exists())

    def test_anonymous_observation_differences_are_not_reconciled(self):
        """Different counts and bboxes without identities require no observation-level choices."""
        a, b = self.media(), self.media()
        box = dict(bbox_x_center=0.5, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2)
        kept = AnimalObservationFactory(mediafile=a, taxon=None, **box)
        AnimalObservationFactory(mediafile=b, taxon=None, **box)
        AnimalObservationFactory(
            mediafile=b, taxon=None, bbox_x_center=0.2, bbox_y_center=0.2, bbox_width=0.1, bbox_height=0.1
        )
        proposal = self.proposal([a, b])
        plan = service.build_plan(proposal, self.user, a.pk)
        self.assertEqual(plan["warnings"], [])
        self.merge(proposal, a)
        self.assertEqual(list(a.observations.values_list("pk", flat=True)), [kept.pk])

    def test_json_scan_start_rejects_inaccessible_ids(self):
        """An explicit scan list cannot be used to probe another workgroup's media."""
        a, b = self.media(), MediaFileFactory()
        response = self.client.post(
            reverse("caidapp:duplicate_scan_start"),
            json.dumps({"mediafile_ids": [a.pk, b.pk]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(models.DuplicateScan.objects.exists())

    def test_scan_request_key_is_idempotent(self):
        """Submitting the same start form twice creates one durable scan."""
        import uuid

        a, b = self.media(), self.media()
        key = uuid.uuid4()
        with patch("caidapp.duplicate_services.dispatch_scan"):
            first = service.create_scan(self.user, [a.pk, b.pk], request_key=key)
            second = service.create_scan(self.user, [a.pk, b.pk], request_key=key)
        self.assertEqual(first.pk, second.pk)

    def test_json_merge_preview_is_available(self):
        """API clients can obtain a signed preview without parsing HTML."""
        a, b = self.media(), self.media()
        proposal = self.proposal([a, b])
        response = self.client.get(reverse("caidapp:duplicate_review", args=[proposal.uuid]), {"format": "json"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["token"])
        self.assertEqual(response.json()["plan"]["keep_id"], a.pk)

    def test_merge_failure_rolls_back_all_database_changes(self):
        """A failed deletion must not leave partial database changes or deletion intents."""
        a, b = self.media(note="kept"), self.media(note="removed")
        observation = AnimalObservationFactory(mediafile=b, taxon=TaxonFactory())
        proposal = self.proposal([a, b])
        plan = service.build_plan(proposal, self.user, a.pk)
        with patch("caidapp.bbox_services._invalidate_reid", side_effect=RuntimeError("storage failure")):
            with self.assertRaises(RuntimeError):
                service.apply_merge(proposal.pk, self.user, service.preview_token(proposal, self.user, plan))
        observation.refresh_from_db()
        a.refresh_from_db()
        proposal.refresh_from_db()
        self.assertEqual(observation.mediafile_id, b.pk)
        self.assertEqual(a.note, "kept")
        self.assertEqual(proposal.status, "pending")
        self.assertFalse(models.DuplicateFileCleanup.objects.exists())
        self.assertTrue((self.root / b.mediafile.name).exists())

    def test_removing_old_representative_invalidates_reid(self):
        """The pre-merge representative flag must be considered even after its row is merged away."""
        a, b = self.media(), self.media()
        taxon = TaxonFactory()
        identity = IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        box = dict(bbox_x_center=0.5, bbox_y_center=0.5, bbox_width=0.2, bbox_height=0.2)
        AnimalObservationFactory(mediafile=a, identity=identity, taxon=taxon, **box)
        AnimalObservationFactory(mediafile=b, identity=identity, taxon=taxon, identity_is_representative=True, **box)
        proposal = self.proposal([a, b])
        with patch("caidapp.tasks.schedule_init_identification_for_workgroup") as schedule:
            with self.captureOnCommitCallbacks(execute=True):
                self.merge(proposal, a)
        self.assertTrue(schedule.called)

    def test_swap_direction_changes_signed_survivor_and_deletes_previous_choice(self):
        """Swapping updates the actual signed action as well as the colored cards."""
        first = self.media(note="first")
        second = self.media(note="second")
        proposal = self.proposal([first, second])
        url = reverse("caidapp:duplicate_review", args=[proposal.uuid])
        initial = self.client.get(url)
        self.assertEqual(initial.context["plan"]["keep_id"], first.pk)
        swapped = self.client.get(initial.context["swap_url"])
        self.assertEqual(swapped.context["plan"]["keep_id"], second.pk)
        self.assertEqual(swapped.context["plan"]["remove_ids"], [first.pk])
        self.assertEqual([card["role"] for card in swapped.context["members"]], ["delete", "keep"])
        before = models.MediaFile.objects.filter(pk=second.pk).values().get()
        with patch("caidapp.tasks.cleanup_duplicate_files.delay"):
            response = self.client.post(initial.context["swap_url"], {"token": swapped.context["token"]})
        self.assertEqual(response.status_code, 302)
        self.assertFalse(models.MediaFile.objects.filter(pk=first.pk).exists())
        self.assertEqual(models.MediaFile.objects.filter(pk=second.pk).values().get(), before)

    def test_triplicate_ui_keeps_selected_survivor_for_next_pair(self):
        """A three-file group advances one deletion at a time, retaining the selected direction."""
        a, b, c = [self.media() for _ in range(3)]
        proposal = self.proposal([a, b, c])
        url = reverse("caidapp:duplicate_review", args=[proposal.uuid])
        preview = self.client.get(url, {"keep": c.pk})
        self.assertEqual(preview.context["plan"]["remove_ids"], [a.pk])
        with patch("caidapp.tasks.cleanup_duplicate_files.delay"):
            response = self.client.post(url, {"token": preview.context["token"]})
        self.assertEqual(response.url, f"{url}?keep={c.pk}")
        following = self.client.get(response.url)
        self.assertEqual(following.context["plan"]["keep_id"], c.pk)
        self.assertEqual(following.context["plan"]["remove_ids"], [b.pk])

    def test_capture_date_difference_is_visible_without_value_selection(self):
        """Capture differences warn before deletion while the chosen date remains authoritative."""
        a = self.media()
        b = self.media()
        b.captured_at = self.when.replace(year=self.when.year - 1)
        b.save(update_fields=["captured_at"])
        proposal = self.proposal([a, b])
        response = self.client.get(reverse("caidapp:duplicate_review", args=[proposal.uuid]))
        self.assertContains(response, "Capture dates differ")
        self.assertNotContains(response, 'name="choice_')
        self.merge(proposal, a)
        a.refresh_from_db()
        self.assertEqual(a.captured_at, self.when)

    def test_source_identity_missing_on_survivor_is_a_warning(self):
        """Removing an assigned identity must also warn when the retained file has no identity."""
        a, b = self.media(), self.media()
        identity = IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
        AnimalObservationFactory(mediafile=b, identity=identity)
        proposal = self.proposal([a, b])
        plan = service.build_plan(proposal, self.user, a.pk)
        self.assertTrue(any(row["field"] == "identities" for row in plan["warnings"]))
        self.merge(proposal, a)
        self.assertFalse(a.observations.filter(identity__isnull=False).exists())

    def test_all_retained_observations_stay_unchanged(self):
        """Multiple objects on the retained file remain separate and byte-for-byte equivalent as records."""
        a, b = self.media(), self.media()
        for media in [a, a, b]:
            AnimalObservationFactory(
                mediafile=media, identity=IndividualIdentityFactory(owner_workgroup=self.user.workgroup)
            )
        before = list(a.observations.order_by("pk").values())
        proposal = self.proposal([a, b])
        self.merge(proposal, a)
        self.assertEqual(list(a.observations.order_by("pk").values()), before)

    def test_deleted_sources_previews_thumbnails_and_crops_are_removed(self):
        """The survivor retains its own files while every unique derivative of the removed file is cleaned up."""
        self.archive.outputdir = str(self.root / "upload")
        self.archive.save(update_fields=["outputdir"])
        kept, removed = self.media(), self.media()
        for media in [kept, removed]:
            for field in ["preview", "thumbnail", "static_thumbnail"]:
                getattr(media, field).save(f"{media.pk}-{field}.webp", ContentFile(f"{media.pk}-{field}".encode()))
        detection = self.root / "upload" / "detection_images"
        detection.mkdir(parents=True)
        for media in [kept, removed]:
            basename = Path(media.mediafile.name)
            (detection / basename.name).write_bytes(b"crop")
            (detection / f"{basename.stem}.0{basename.suffix}").write_bytes(b"numbered crop")
        unrelated = detection / "unrelated.jpg"
        unrelated.write_bytes(b"unrelated")
        proposal = self.proposal([kept, removed])
        kept_names = service.owned_file_names(kept)
        removed_names = service.owned_file_names(removed)
        kept_bytes = {name: service.media_path(name).read_bytes() for name in kept_names}
        self.merge(proposal, kept)
        service.cleanup_files(list(models.DuplicateFileCleanup.objects.values_list("pk", flat=True)))
        for name in removed_names - kept_names:
            self.assertFalse(service.media_path(name).exists(), name)
        for name, contents in kept_bytes.items():
            self.assertEqual(service.media_path(name).read_bytes(), contents)
        self.assertTrue(unrelated.exists())

    def test_deletion_does_not_transfer_album_membership(self):
        """Deleting a file does not silently enroll the survivor into its albums."""
        a, b = self.media(), self.media()
        album = models.Album.objects.create(name="Source only", owner=self.user)
        album.mediafiles.add(b)
        proposal = self.proposal([a, b])
        self.merge(proposal, a)
        self.assertFalse(album.mediafiles.exists())

    def test_previous_metadata_merge_token_cannot_authorize_new_deletion(self):
        """Old tabs must reload before using the changed whole-file semantics."""
        a, b = self.media(), self.media()
        proposal = self.proposal([a, b])
        old = signing.dumps(
            {"proposal": str(proposal.uuid), "user": self.user.pk, "keep": a.pk, "remove": [b.pk]},
            salt="duplicate-merge",
        )
        with self.assertRaises(signing.BadSignature):
            service.apply_merge(proposal.pk, self.user, old)
        self.assertTrue(models.MediaFile.objects.filter(pk=b.pk).exists())
