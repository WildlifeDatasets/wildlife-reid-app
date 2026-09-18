from django.test import TestCase
from django.conf import settings
from django.shortcuts import resolve_url
from django.urls import reverse

from caidapp import models

from .factories import (
    AlbumFactory,
    AnimalObservationFactory,
    CaidUserFactory,
    IndividualIdentityFactory,
    LocalityFactory,
    MediaFileFactory,
    UploadedArchiveFactory,
)


class ComparisonEndpointsTest(TestCase):
    def setUp(self):
        self.caiduser = CaidUserFactory(admin=True)
        self.client.force_login(self.caiduser.user)
        self.archive = UploadedArchiveFactory(owner=self.caiduser)

    def source(self, **params):
        return self.client.get(reverse("caidapp:comparison_source"), params)

    def test_comparison_page_requires_authentication(self):
        self.client.logout()

        response = self.client.get(reverse("caidapp:comparison"))

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith(resolve_url(settings.LOGIN_URL)))
        self.assertIn("next=" + reverse("caidapp:comparison"), response.url)

    def test_comparison_page_renders_panel_and_tray_wiring(self):
        response = self.client.get(reverse("caidapp:comparison"))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "caidapp/comparison.html")
        self.assertContains(response, 'id="comparison-app"')
        self.assertContains(response, reverse("caidapp:comparison_source"))
        self.assertContains(response, reverse("caidapp:comparison_search"))
        self.assertContains(response, 'id="comparison-tray"')
        self.assertContains(response, reverse("caidapp:comparison"))
        self.assertContains(response, "caidapp/js/comparison.js")
        self.assertContains(response, "caidapp/js/comparison_tray.js")

    def test_observation_source_keeps_multiple_observations_from_one_mediafile(self):
        mediafile = MediaFileFactory(parent=self.archive)
        first = AnimalObservationFactory(
            mediafile=mediafile, bbox_x_center=0.2, bbox_y_center=0.3, bbox_width=0.1, bbox_height=0.2
        )
        second = AnimalObservationFactory(
            mediafile=mediafile, bbox_x_center=0.8, bbox_y_center=0.7, bbox_width=0.3, bbox_height=0.4
        )

        response = self.source(kind="observation", id=second.id)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["items"][0]["observation_id"], second.id)
        self.assertEqual(payload["items"][0]["mediafile_id"], first.mediafile_id)
        self.assertEqual(payload["items"][0]["bbox"], [0.8, 0.7, 0.3, 0.4])

    def test_identity_gallery_excludes_placeholder_and_puts_representative_first(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Lynx A")
        placeholder_mediafile = MediaFileFactory(parent=self.archive)
        placeholder = placeholder_mediafile.observations.get()
        # This represents old/invalid data; comparison must still never show it as an animal.
        models.AnimalObservation.objects.filter(pk=placeholder.pk).update(identity=identity)
        normal = AnimalObservationFactory(mediafile=MediaFileFactory(parent=self.archive), identity=identity)
        representative = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=self.archive), identity=identity, identity_is_representative=True
        )

        response = self.source(kind="identity", id=identity.id)

        self.assertEqual(response.status_code, 200)
        item_ids = [item["observation_id"] for item in response.json()["items"]]
        self.assertEqual(item_ids, [representative.id, normal.id])
        self.assertNotIn(placeholder.id, item_ids)

    def test_identity_gallery_paginates_at_24(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)
        for _ in range(25):
            AnimalObservationFactory(mediafile=MediaFileFactory(parent=self.archive), identity=identity)

        first_page = self.source(kind="identity", id=identity.id, page=1).json()
        second_page = self.source(kind="identity", id=identity.id, page=2).json()

        self.assertEqual((first_page["count"], first_page["num_pages"], len(first_page["items"])), (25, 2, 24))
        self.assertEqual((second_page["page"], len(second_page["items"])), (2, 1))

    def test_non_album_sources_accept_the_ui_mode_parameter(self):
        observation = AnimalObservationFactory(mediafile=MediaFileFactory(parent=self.archive))

        response = self.source(kind="observation", id=observation.id, mode="mediafiles")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["observation_id"], observation.id)

    def test_album_modes_distinguish_mediafiles_and_observations(self):
        album = AlbumFactory(owner=self.caiduser)
        mediafile = MediaFileFactory(parent=self.archive)
        first = AnimalObservationFactory(mediafile=mediafile)
        second = AnimalObservationFactory(mediafile=mediafile)
        placeholder_mediafile = MediaFileFactory(parent=self.archive)
        album.mediafiles.add(mediafile, placeholder_mediafile)

        mediafile_payload = self.source(kind="album", id=album.hash).json()
        observation_payload = self.source(kind="album", id=album.hash, mode="observations").json()

        self.assertEqual(mediafile_payload["count"], 2)
        self.assertEqual(observation_payload["count"], 2)
        self.assertEqual(
            {item["observation_id"] for item in observation_payload["items"]}, {first.id, second.id}
        )

    def test_other_workgroup_cannot_read_or_search_sources(self):
        other = CaidUserFactory(admin=True)
        other_archive = UploadedArchiveFactory(owner=other)
        other_mediafile = MediaFileFactory(parent=other_archive, original_filename="secret-lynx.jpg")
        other_observation = AnimalObservationFactory(mediafile=other_mediafile)
        other_identity = IndividualIdentityFactory(owner_workgroup=other.workgroup, name="Secret Lynx")

        response = self.source(kind="observation", id=other_observation.id)
        search = self.client.get(reverse("caidapp:comparison_search"), {"kind": "mediafile", "q": "secret"})
        identity_search = self.client.get(reverse("caidapp:comparison_search"), {"kind": "identity", "q": "secret"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(search.json(), {"results": []})
        self.assertEqual(identity_search.json(), {"results": []})

    def test_identity_search_includes_code_and_observation_items_are_safe_for_overlay(self):
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup, name="Plain name", code="LYNX-17")
        image_observation = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=self.archive),
            identity=identity,
            bbox_x_center=0.5,
            bbox_y_center=0.5,
            bbox_width=0,
            bbox_height=0.2,
        )
        video_observation = AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=self.archive, media_type="video"),
            identity=identity,
            bbox_x_center=0.5,
            bbox_y_center=0.5,
            bbox_width=0.2,
            bbox_height=0.2,
        )

        search = self.client.get(reverse("caidapp:comparison_search"), {"kind": "identity", "q": "lynx-17"})
        image_item = self.source(kind="observation", id=image_observation.id).json()["items"][0]
        video_item = self.source(kind="observation", id=video_observation.id).json()["items"][0]

        self.assertEqual(search.json()["results"], [{"kind": "identity", "id": identity.id, "label": "Plain name"}])
        self.assertIsNone(image_item["bbox"])
        self.assertIsNone(video_item["bbox"])
        self.assertIn(f"Observation {image_observation.id}", image_item["label"])
        self.assertTrue(image_item["detail_url"].endswith(f"#observation-{image_observation.id}"))

    def test_item_actions_include_available_edit_targets(self):
        locality = LocalityFactory(owner=self.caiduser)
        identity = IndividualIdentityFactory(owner_workgroup=self.caiduser.workgroup)
        mediafile = MediaFileFactory(parent=self.archive, locality=locality)
        observation = AnimalObservationFactory(mediafile=mediafile, identity=identity)

        item = self.source(kind="observation", id=observation.id).json()["items"][0]

        mediafile_url = reverse("caidapp:media_file_update", args=[mediafile.id])
        self.assertEqual(item["mediafile_url"], mediafile_url)
        self.assertEqual(item["mediafile_edit_url"], mediafile_url)
        self.assertEqual(item["observation_edit_url"], f"{mediafile_url}#observation-{observation.id}")
        self.assertEqual(
            item["identity_url"],
            reverse("caidapp:individual_identity_update", args=[identity.id]),
        )
        self.assertEqual(item["locality_url"], reverse("caidapp:update_locality", args=[locality.id]))

    def test_shared_album_includes_only_mediafiles_explicitly_shared_with_user(self):
        other = CaidUserFactory(admin=True)
        shared_album = AlbumFactory(owner=other)
        shared_mediafile = MediaFileFactory(parent=UploadedArchiveFactory(owner=other))
        unrelated_mediafile = MediaFileFactory(parent=UploadedArchiveFactory(owner=other))
        shared_album.mediafiles.add(shared_mediafile)
        models.AlbumShareRole.objects.create(album=shared_album, user=self.caiduser)

        response = self.source(kind="album", id=shared_album.hash)

        self.assertEqual(response.status_code, 200)
        item = response.json()["items"][0]
        self.assertEqual(item["mediafile_id"], shared_mediafile.id)
        self.assertNotEqual(item["mediafile_id"], unrelated_mediafile.id)
        self.assertEqual(item["mediafile_edit_url"], "")
        self.assertEqual(item["locality_url"], "")
        self.assertEqual(
            item["mediafile_url"], reverse("caidapp:media_file_update", args=[shared_mediafile.id])
        )

    def test_user_without_workgroup_cannot_search_unrelated_legacy_null_workgroup_identities(self):
        personal_user = CaidUserFactory()
        personal_user.workgroup = None
        personal_user.save(update_fields=["workgroup"])
        personal_archive = UploadedArchiveFactory(owner=personal_user)
        visible_identity = IndividualIdentityFactory(owner_workgroup=None, name="Visible legacy identity")
        hidden_identity = IndividualIdentityFactory(owner_workgroup=None, name="Hidden legacy identity")
        AnimalObservationFactory(
            mediafile=MediaFileFactory(parent=personal_archive), identity=visible_identity
        )
        self.client.force_login(personal_user.user)

        response = self.client.get(
            reverse("caidapp:comparison_search"), {"kind": "identity", "q": "legacy identity"}
        )

        self.assertEqual(
            response.json()["results"],
            [{"kind": "identity", "id": visible_identity.id, "label": visible_identity.name}],
        )
        self.assertNotIn(
            hidden_identity.id,
            [result["id"] for result in response.json()["results"]],
        )

    def test_invalid_source_input_is_rejected(self):
        self.assertEqual(self.source(kind="unknown", id="1").status_code, 400)
        self.assertEqual(self.source(kind="identity", id="not-a-number").status_code, 400)
        self.assertEqual(self.source(kind="album", id="missing", mode="bad").status_code, 400)
