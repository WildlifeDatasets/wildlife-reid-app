"""Exercise exact retrieval, stale inputs and observation-scoped identity writes."""

from tempfile import TemporaryDirectory

from caidapp import models
from caidapp.observation_embeddings import snapshot
from caidapp.services.identity_candidates import _source_digest, current_input
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from .factories import CaidUserFactory, IndividualIdentityFactory, MediaFileFactory, UploadedArchiveFactory


class IdentityCandidateTests(TestCase):
    """Keep retrieval and confirmation tied to the saved observation input."""

    def setUp(self):
        """Build a shared-media query and reference gallery in one workgroup."""
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        override = self.settings(MEDIA_ROOT=directory.name)
        override.enable()
        self.addCleanup(override.disable)
        _source_digest.cache_clear()
        self.member = CaidUserFactory()
        self.group = self.member.workgroup
        self.model = models.IdentificationModel.objects.create(
            name="Test encoder", public=True, model_path="hf-hub:example/model"
        )
        self.group.identification_model = self.model
        self.group.save(update_fields=["identification_model"])
        self.archive = UploadedArchiveFactory(owner=self.member)
        mediafile = MediaFileFactory(parent=self.archive)
        self.query = mediafile.observations.first()
        self.query.is_no_detection_placeholder = False
        self.query.save()
        self.sibling = models.AnimalObservation.objects.create(mediafile=mediafile)
        self.identity = IndividualIdentityFactory(owner_workgroup=self.group)
        self.profile = models.EmbeddingProfile.objects.create(
            specification={"dimension": 2, "model_source": self.model.get_runtime_model_source()}, dimension=2
        )
        self.run = models.TaskRun.objects.create(
            workgroup=self.group,
            status="succeeded",
            finished_at=timezone.now(),
            model_snapshot={
                "model_id": self.model.pk,
                "model_source": self.model.get_runtime_model_source(),
                "weights_path": self.model.get_runtime_checkpoint_path(),
            },
            result={"profile_uuid": str(self.profile.uuid)},
        )
        self.embed(self.query, [10, 0])
        self.reference = self.reference_for(self.identity, [100, 0])
        self.url = reverse("caidapp:observation_identity_candidates", args=[self.query.uuid])
        self.accept_url = reverse("caidapp:observation_accept_identity", args=[self.query.uuid])
        self.client.force_login(self.member.user)

    def embed(self, observation, vector, profile=None):
        """Persist an embedding of exactly the current source and crop."""
        row = snapshot(observation)
        return models.ObservationEmbedding.objects.create(
            observation=observation,
            profile=profile or self.profile,
            input_fingerprint=row["input_fingerprint"],
            input_specification=row["input_specification"],
            vector=vector,
        )

    def reference_for(self, identity, vector):
        """Create another reference observation in the same gallery."""
        observation = MediaFileFactory(parent=self.archive).observations.first()
        observation.is_no_detection_placeholder = False
        observation.identity = identity
        observation.save()
        self.embed(observation, vector)
        return observation

    def accept(self, payload=None, **changes):
        """Submit the dialog's optimistic concurrency fields."""
        payload = payload or self.client.get(self.url).json()
        data = {
            "identity_id": self.identity.pk,
            "input_fingerprint": payload["input_fingerprint"],
            "expected_identity_id": payload["expected_identity_id"] or "",
        }
        data.update(changes)
        return self.client.post(self.accept_url, data)

    def test_distinct_identities_cosine_order_and_reference_crops(self):
        """Deduplicate identities and normalize vector magnitude before ranking."""
        self.reference_for(self.identity, [1, 0.1])
        other = IndividualIdentityFactory(owner_workgroup=self.group)
        self.reference_for(other, [1000, 1000])
        self.reference.bbox_x_center = self.reference.bbox_y_center = 0.5
        self.reference.bbox_width = self.reference.bbox_height = 0.4
        self.reference.save()
        self.embed(self.reference, [100, 0])
        payload = self.client.get(self.url).json()
        self.assertEqual(payload["status"], "complete")
        self.assertEqual([row["identity_id"] for row in payload["candidates"]], [self.identity.pk, other.pk])
        self.assertAlmostEqual(payload["candidates"][0]["score"], 1)
        self.assertEqual(payload["candidates"][0]["reference"]["bbox"], [0.5, 0.5, 0.4, 0.4])
        self.assertEqual(payload["reference_count"], 3)

    def test_stale_reference_and_incompatible_profile_are_not_used(self):
        """Never rank historical crops or embeddings from another feature space."""
        self.reference.orientation = "L"
        self.reference.save()
        profile = models.EmbeddingProfile.objects.create(specification={"dimension": 2, "other": True}, dimension=2)
        self.embed(self.reference, [1, 0], profile=profile)
        payload = self.client.get(self.url).json()
        self.assertEqual(payload["candidates"], [])
        self.assertEqual(payload["reference_count"], 0)
        self.assertEqual(payload["eligible_reference_count"], 1)

    def test_changed_query_or_selected_model_requires_embedding(self):
        """Do not silently fall back to an old crop or model profile."""
        self.query.orientation = "L"
        self.query.save()
        self.assertEqual(self.client.get(self.url).json()["status"], "embedding_required")
        self.query.orientation = "N"
        self.query.save()
        self.group.identification_model = models.IdentificationModel.objects.create(
            name="Other encoder", model_path="hf-hub:other/model", public=True
        )
        self.group.save(update_fields=["identification_model"])
        self.assertEqual(self.client.get(self.url).json()["status"], "embedding_required")

    def test_source_replacement_invalidates_cached_digest(self):
        """A source edit invalidates the cheap stat-keyed source cache."""
        old = current_input(self.query)
        self.assertEqual(old, snapshot(self.query)["input_fingerprint"])
        with open(self.query.mediafile.image_file.path, "ab") as stream:
            stream.write(b"changed source")
        self.assertNotEqual(current_input(self.query), old)
        self.assertEqual(self.client.get(self.url).json()["status"], "embedding_required")

    def test_accept_changes_only_selected_observation_and_records_author(self):
        """Sibling observations and the legacy media-file queue remain independent."""
        self.assertEqual(self.accept().status_code, 200)
        self.query.refresh_from_db()
        self.sibling.refresh_from_db()
        self.assertEqual(self.query.identity_id, self.identity.pk)
        self.assertEqual(self.query.updated_by_id, self.member.pk)
        self.assertIsNone(self.sibling.identity_id)

    def test_stale_dialog_cannot_overwrite_new_annotation(self):
        """Reject concurrent crop and identity edits."""
        payload = self.client.get(self.url).json()
        self.query.identity = IndividualIdentityFactory(owner_workgroup=self.group)
        self.query.save()
        self.assertEqual(self.accept(payload).status_code, 409)
        self.query.identity = None
        self.query.orientation = "L"
        self.query.save()
        self.assertEqual(self.accept(payload).status_code, 409)

    def test_permissions_and_post_csrf(self):
        """Knowing UUIDs or identity IDs grants no cross-workgroup access."""
        foreign = IndividualIdentityFactory()
        self.assertEqual(self.accept(identity_id=foreign.pk).status_code, 404)
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.member.user)
        self.assertEqual(client.post(self.accept_url, {}).status_code, 403)
        self.assertEqual(self.client.get(self.accept_url).status_code, 405)
        self.client.force_login(CaidUserFactory().user)
        self.assertEqual(self.client.get(self.url).status_code, 404)
        self.assertEqual(self.client.post(self.accept_url, {}).status_code, 404)

    def test_placeholder_and_video_are_rejected(self):
        """Do not identify absent objects or video boxes without frame locators."""
        self.query.is_no_detection_placeholder = True
        self.query.save()
        self.assertEqual(self.client.get(self.url).status_code, 400)
        self.query.is_no_detection_placeholder = False
        self.query.save()
        self.query.mediafile.media_type = "video"
        self.query.mediafile.save(update_fields=["media_type"])
        self.assertEqual(self.client.get(self.url).status_code, 400)
        self.assertEqual(self.accept({"input_fingerprint": "old", "expected_identity_id": None}).status_code, 400)

    def test_foreign_identity_reference_never_enters_gallery(self):
        """Ignore incorrectly labelled references from another workgroup."""
        self.reference.identity = IndividualIdentityFactory()
        self.reference.save()
        self.assertEqual(self.client.get(self.url).json()["candidates"], [])

    def test_same_source_hash_is_reused_for_siblings_and_repeated_searches(self):
        """Warm retrieval does not repeatedly read the same source media file."""
        current_input(self.query)
        current_input(self.sibling)
        current_input(self.query)
        self.assertEqual(_source_digest.cache_info().misses, 1)
        self.assertEqual(_source_digest.cache_info().hits, 2)

    def test_no_workgroup_cannot_read_or_modify_workgroup_observation(self):
        """A missing workgroup must not broaden the scope to legacy NULL owners."""
        self.client.force_login(CaidUserFactory(workgroup=None).user)
        self.assertEqual(self.client.get(self.url).status_code, 404)
        self.assertEqual(self.client.post(self.accept_url, {}).status_code, 404)

    def test_taxon_filter_and_invalid_reference_vector(self):
        """Keep known taxa separate and tolerate unusable stored reference vectors."""
        from .factories import TaxonFactory

        self.query.taxon = TaxonFactory()
        self.query.save()
        self.assertEqual(self.client.get(self.url).json()["eligible_reference_count"], 0)
        self.reference.taxon = self.query.taxon
        self.reference.save()
        self.reference.embeddings.update(vector=[0, 0])
        payload = self.client.get(self.url).json()
        self.assertEqual(payload["eligible_reference_count"], 1)
        self.assertEqual(payload["candidates"], [])

    def test_buttons_render_for_saved_real_observations(self):
        """Expose the same dialog from both observation layouts and media-file detail."""
        for name in ["observation_card.html", "observation_list_row.html"]:
            from django.template.loader import render_to_string

            self.query.bulk_editable = True
            html = render_to_string("caidapp/includes/" + name, {"observation": self.query})
            self.assertIn(self.url, html)
        response = self.client.get(reverse("caidapp:media_file_update", args=[self.query.mediafile_id]))
        self.assertContains(response, self.url)
        self.assertContains(response, "identity-candidates-modal")
