"""Verify review and confirmation of saved observation comparisons."""

from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from caidapp import models, observation_comparisons
from caidapp.observation_embeddings import embedding_generation
from caidapp.services.identity_review import input_version

from .factories import AnimalObservationFactory, CaidUserFactory, MediaFileFactory, UploadedArchiveFactory


class ObservationComparisonTest(TestCase):
    def setUp(self):
        self.member = CaidUserFactory(admin=True)
        self.member.user.is_staff = True
        self.member.user.save(update_fields=["is_staff"])
        self.client.force_login(self.member.user)
        self.group = self.member.workgroup
        self.media = MediaFileFactory(parent=UploadedArchiveFactory(owner=self.member))
        self.observations = [AnimalObservationFactory(mediafile=self.media) for _ in range(3)]

    def test_comparison_runs_appear_alongside_existing_identification_tasks(self):
        with patch("caidapp.observation_comparisons.prepare.apply_async") as submit:
            response = self.client.post(
                reverse("caidapp:start_observation_comparison"), {"mode": "review_known", "gallery": "all_identified"}
            )
        run = models.TaskRun.objects.get(kind="review_known")
        self.assertTrue(run.model_snapshot["include_non_representative"])
        self.assertRedirects(response, reverse("caidapp:task_detail", args=[run.uuid]))
        submit.assert_called_once()
        models.IdentificationRunStatistic.objects.create(workgroup=self.group, operation="init")
        models.IdentificationOutlierSuggestionResult.objects.create(workgroup=self.group, status="processing")
        models.IdentificationSimilarPairResult.objects.create(workgroup=self.group)
        tasks = self.client.get(reverse("caidapp:task_list"), {"state": "active"})
        for label in ("Review assigned identities", "Init identification", "Identification outliers",
                      "Identity merge suggestions"):
            self.assertContains(tasks, label)

    def test_confirmed_pairs_extend_one_temporary_identity(self):
        def evidence(observation):
            return {"id": observation.pk, "mediafile_id": self.media.pk, "identity": "",
                    "version": input_version(observation, self.media)}

        first, second, third = self.observations
        run = models.TaskRun.objects.create(
            kind="compare_unknown", status="running", workgroup=self.group,
            model_snapshot={"generation": embedding_generation(self.group.pk)},
        )
        observation_comparisons.finish({"rows": [
            {"query": evidence(first), "candidates": [{**evidence(second), "score": 0.9, "role": "similar"}]},
            {"query": evidence(second), "candidates": [{**evidence(third), "score": 0.8, "role": "similar"}]},
        ]}, run.pk)
        detail = self.client.get(reverse("caidapp:task_detail", args=[run.uuid]))
        self.assertContains(detail, "Group pair", count=2)
        url = reverse("caidapp:group_observation_pair", args=[run.uuid])
        self.assertEqual(self.client.post(url, {"query_id": first.pk, "candidate_id": second.pk}).status_code, 302)
        self.assertEqual(self.client.post(url, {"query_id": second.pk, "candidate_id": third.pk}).status_code, 302)
        for observation in self.observations:
            observation.refresh_from_db()
        self.assertTrue(first.identity.name.startswith("Unknown-"))
        self.assertEqual([item.identity_id for item in self.observations], [first.identity_id] * 3)
