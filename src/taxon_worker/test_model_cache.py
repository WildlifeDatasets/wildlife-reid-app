"""Exercise persistent checkpoint downloads without a network connection or ML dependencies."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from detection_utils.model_cache import download_checkpoint_if_missing


class ModelCacheTest(unittest.TestCase):
    def setUp(self):
        """Prepare isolated model storage and a mocked streaming HTTP response."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.checkpoint = Path(directory.name) / "models" / "md_v5a.0.0.pt"
        self.response = MagicMock()
        self.response.__enter__.return_value = self.response
        self.response.headers = {"content-length": "6"}
        self.response.iter_content.return_value = iter([b"abc", b"def"])
        self.get = Mock(return_value=self.response)
        requests = patch.dict(sys.modules, {"requests": SimpleNamespace(get=self.get)})
        requests.start()
        self.addCleanup(requests.stop)

    def download(self):
        """Fetch the fixture checkpoint through the production cache helper."""
        return download_checkpoint_if_missing("https://example.test/model.pt", self.checkpoint)

    def test_repeated_loads_reuse_persisted_weights(self):
        """Download once and reuse the file on later calls with the same volume contents."""
        self.assertEqual(self.download(), self.checkpoint)
        self.assertEqual(self.checkpoint.read_bytes(), b"abcdef")
        self.get.reset_mock()
        self.assertEqual(self.download(), self.checkpoint)
        self.get.assert_not_called()
        self.assertEqual(list(self.checkpoint.parent.iterdir()), [self.checkpoint])

    def test_incomplete_stream_does_not_publish_checkpoint(self):
        """A short response must leave no final model or temporary download to reuse."""
        self.response.iter_content.return_value = iter([b"abc"])
        with self.assertRaisesRegex(ValueError, "Incomplete model download"):
            self.download()
        self.assertFalse(self.checkpoint.exists())
        self.assertEqual(list(self.checkpoint.parent.iterdir()), [])

    def test_interrupted_download_is_cleaned_up_and_retried(self):
        """An interrupted transfer must be retried successfully on the next invocation."""

        def interrupted_chunks(**kwargs):
            yield b"abc"
            raise OSError("connection interrupted")

        self.response.iter_content.side_effect = interrupted_chunks
        with self.assertRaisesRegex(OSError, "connection interrupted"):
            self.download()
        self.assertFalse(self.checkpoint.exists())
        self.assertEqual(list(self.checkpoint.parent.iterdir()), [])
        self.response.iter_content.side_effect = None
        self.response.iter_content.return_value = iter([b"abcdef"])
        self.download()
        self.assertEqual(self.checkpoint.read_bytes(), b"abcdef")

    def test_http_error_does_not_create_model_file(self):
        """Reject HTTP failures before treating an error page as model weights."""
        self.response.raise_for_status.side_effect = RuntimeError("HTTP 404")
        with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
            self.download()
        self.assertFalse(self.checkpoint.exists())
        self.response.iter_content.assert_not_called()

    def test_empty_response_without_content_length_is_rejected(self):
        """Reject an empty response even when the server does not report a length."""
        self.response.headers = {}
        self.response.iter_content.return_value = iter([])
        with self.assertRaisesRegex(ValueError, "Incomplete model download"):
            self.download()
        self.assertFalse(self.checkpoint.exists())

    def test_response_without_content_length_can_be_cached(self):
        """Accept a non-empty complete response when no content length is supplied."""
        self.response.headers = {}
        self.download()
        self.assertEqual(self.checkpoint.read_bytes(), b"abcdef")

    def test_empty_existing_file_is_replaced(self):
        """Recover an empty checkpoint instead of treating it as a cache hit."""
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.touch()
        self.download()
        self.assertEqual(self.checkpoint.read_bytes(), b"abcdef")


if __name__ == "__main__":
    unittest.main()
