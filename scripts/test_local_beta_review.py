"""Exercise local review claiming, explicit decisions and HTTP access boundaries."""

import json
import sqlite3
import threading
import unittest
from contextlib import closing
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from local_beta_review import ReviewSession, make_handler


class LocalReviewTests(unittest.TestCase):
    def setUp(self):
        """Prepare a synthetic dataset without any production media or decisions."""
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "image.jpg").write_bytes(b"synthetic image")
        (self.dataset / "annotations.json").write_text(
            json.dumps(
                {
                    "images": [{"uuid": "media", "file_name": "image.jpg", "width": 100, "height": 100}],
                    "annotations": [
                        {"uuid": value, "image_uuid": "media", "bbox": [10, 10, 40, 40]} for value in ("left", "right")
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.output = self.root / "output"
        self.output.mkdir()
        self.session = ReviewSession(self.dataset, self.output, Mock(poll=Mock(return_value=None)))

    def prepare_database(self):
        """Create the worker-owned schema with one waiting synthetic pair."""
        with closing(sqlite3.connect(self.session.db)) as db, db:
            db.execute(
                "CREATE TABLE image_verification (id TEXT PRIMARY KEY, uuid1 TEXT, uuid2 TEXT, priority INTEGER, "
                "status TEXT, decision TEXT, instance_id TEXT, started_at TEXT, completed_at TEXT)"
            )
            db.execute(
                "INSERT INTO image_verification (id,uuid1,uuid2,priority,status,decision) "
                "VALUES ('pair','left','right',0,'awaiting','none')"
            )

    def test_waiting_does_not_create_empty_database(self):
        """Let the actual beta-stability backend initialize its own database."""
        self.assertIsNone(self.session.state()["pair"])
        self.assertFalse(self.session.db.exists())

    def test_claim_and_explicit_answer_are_session_owned_and_idempotent(self):
        """Preserve a displayed pair and reject another tab's duplicate verdict."""
        self.prepare_database()
        pair = self.session.state()["pair"]
        self.assertEqual(pair["id"], self.session.state()["pair"]["id"])
        other = ReviewSession(self.dataset, self.output, self.session.process)
        with self.assertRaises(ValueError):
            other.answer("pair", "incorrect")
        self.session.answer("pair", "correct")
        with self.assertRaises(ValueError):
            self.session.answer("pair", "incorrect")
        with closing(self.session.connect()) as db, db:
            self.assertEqual(
                db.execute("SELECT status,decision FROM image_verification").fetchone(), ("checked", "correct")
            )

    def test_http_requires_session_path_and_explicit_json_decisions(self):
        """Serve only allowlisted images and accept valid same-session votes."""
        self.prepare_database()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.session))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"/{self.session.token}/"

            def request(route, body=None):
                client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                try:
                    client.request(
                        "GET" if body is None else "POST",
                        base + route,
                        body=None if body is None else json.dumps(body),
                        headers={"Content-Type": "application/json"},
                    )
                    response = client.getresponse()
                    return response.status, response.read()
                finally:
                    client.close()

            self.assertEqual(json.loads(request("pair")[1])["pair"]["left"]["uuid"], "left")
            self.assertEqual(request("image/../../secret")[0], 404)
            self.assertEqual(request("image/left")[1], b"synthetic image")
            self.assertTrue(json.loads(request("answer", {"id": "pair", "decision": "cant_tell"})[1])["ok"])
            self.assertEqual(request("stop", {})[0], 200)
            self.assertTrue((self.output / "stop.request").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_source_must_stay_inside_dataset(self):
        """Do not expose arbitrary local paths through an exported image reference."""
        (self.root / "outside.jpg").write_bytes(b"private")
        path = self.dataset / "annotations.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["images"][0]["file_name"] = "../outside.jpg"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            ReviewSession(self.dataset, self.output, self.session.process)


if __name__ == "__main__":
    unittest.main()
