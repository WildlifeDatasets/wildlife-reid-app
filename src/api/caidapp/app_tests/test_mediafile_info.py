"""Metadata components preserve privacy, escaping, and observation multiplicity."""

import json
from html.parser import HTMLParser

from caidapp.templatetags.mediafile_info import mediafile_info
from django.template.loader import render_to_string
from django.test import RequestFactory, TestCase

from .factories import AnimalObservationFactory, IndividualIdentityFactory, LocalityFactory, MediaFileFactory


class TriggerParser(HTMLParser):
    """Read the escaped payload as a browser would, without depending on a DOM library."""

    def handle_starttag(self, tag, attrs):
        """Capture the metadata trigger's data attributes."""
        if tag == "button":
            self.attributes = dict(attrs)


class MediafileInfoTest(TestCase):
    def test_multiple_observations_and_safe_payload_roundtrip(self):
        """One shared payload describes all real objects and keeps markup as text."""
        media = MediaFileFactory(
            note='<img src=x onerror="alert(1)">', metadata_json={"camera": "</script><script>bad()</script>"}
        )
        first = AnimalObservationFactory(mediafile=media, identity=IndividualIdentityFactory())
        second = AnimalObservationFactory(mediafile=media, identity=IndividualIdentityFactory())
        context = mediafile_info(media)
        payload = json.loads(context["payload"])
        labels = {row["label"] for row in payload["rows"]}
        self.assertIn(f"Observation #{first.pk}", labels)
        self.assertIn(f"Observation #{second.pk}", labels)
        self.assertIn(str(first.identity), context["summary"])
        self.assertIn(f"Upload: {media.parent.name}", context["summary"])
        self.assertIn({"label": "Upload", "value": media.parent.name}, payload["rows"])
        html = render_to_string("caidapp/includes/mediafile_info.html", context)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img", html)
        parser = TriggerParser()
        parser.feed(html)
        self.assertEqual(json.loads(parser.attributes["data-mediafile-info"]), payload)

    def test_private_mode_masks_location_and_raw_metadata(self):
        """The hover, modal and clipboard share the same privacy-filtered payload."""
        locality = LocalityFactory(name="Secret den", location="49.12,15.24")
        media = MediaFileFactory(locality=locality, location="48.98,15.67", metadata_json={"GPS": "49.12,15.24"})
        request = RequestFactory().get("/")
        request.session = {"private_mode": True}
        private = mediafile_info(media, request)
        self.assertNotIn("Secret den", private["payload"])
        self.assertNotIn("49.12", private["payload"])
        self.assertNotIn("48.98", private["payload"])
        self.assertNotIn("GPS", private["payload"])
        self.assertNotIn("Secret den", private["summary"])
        request.session = {}
        public = mediafile_info(media, request)
        self.assertIn("Secret den", public["payload"])
        self.assertIn("GPS", public["payload"])

    def test_placeholder_is_not_a_detected_object(self):
        """No-detection placeholders do not appear as real observations."""
        media = MediaFileFactory()
        payload = json.loads(mediafile_info(media)["payload"])
        self.assertIn({"label": "Detected objects", "value": "0"}, payload["rows"])
        self.assertFalse(any(row["label"].startswith("Observation #") for row in payload["rows"]))

    def test_prefetched_relations_need_no_extra_queries(self):
        """Repeated card/list metadata does not query observations individually."""
        media = MediaFileFactory()
        AnimalObservationFactory(mediafile=media)
        media = (
            type(media)
            .objects.select_related("parent", "sequence", "locality")
            .prefetch_related("observations__taxon", "observations__identity")
            .get(pk=media.pk)
        )
        with self.assertNumQueries(0):
            mediafile_info(media)
            mediafile_info(media)
