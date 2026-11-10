"""Exercise reference loading without importing GPU models or starting a Celery worker."""

import ast
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd


class ReferenceFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the production function while excluding worker startup and GPU imports."""
        tree = ast.parse(Path(__file__).with_name("worker.py").read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_features"
        )
        namespace = {"pd": pd, "json": json}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "worker.py", "exec"), namespace)
        cls.load_features = staticmethod(namespace["load_features"])

    def test_empty_database_reports_initialization_required(self):
        """Handle empty results both with and without DataFrame columns."""
        for frame in (pd.DataFrame(), pd.DataFrame(columns=["embedding", "image_path"])):
            with self.subTest(columns=list(frame.columns)):
                connection = Mock()
                connection.reference_image.get_reference_images.return_value = frame
                with self.assertRaisesRegex(ValueError, "No reference embeddings found for workgroup 42"):
                    self.load_features(connection, 42)

    def test_available_references_keep_embeddings_and_metadata(self):
        """Continue decoding available vectors without losing metadata or organization scoping."""
        frame = pd.DataFrame([{"image_path": "one.jpg", "embedding": "[0.1, 0.9]"}])
        connection = Mock()
        connection.reference_image.get_reference_images.return_value = frame
        features, metadata = self.load_features(connection, 42)
        self.assertEqual(features, [[0.1, 0.9]])
        self.assertIs(metadata, frame)
        connection.reference_image.get_reference_images.assert_called_once_with(42, start=-1, end=-1, rows=[])


if __name__ == "__main__":
    unittest.main()
