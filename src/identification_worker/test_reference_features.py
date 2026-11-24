"""Exercise stored reference inputs without starting a GPU worker."""

import ast
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pandas as pd

from utils.feature_snapshot import FeatureSnapshot, load_analysis_inputs, load_match_inputs


class ReferenceFeaturesTest(unittest.TestCase):
    def test_observation_snapshot_keeps_same_file_rows_and_candidate_order(self):
        """Ranked row reads and paged reads address the same observations."""
        rows = [
            {"observation_id": 9, "image_path": "same.jpg", "embedding": [[0, 1], {"descriptors": [[1]]}]},
            {"observation_id": 8, "image_path": "same.jpg", "embedding": [[1, 0], {"descriptors": [[2]]}]},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            snapshot = FeatureSnapshot(path)
            self.assertEqual(snapshot.read([0]), rows[:1])
            self.assertEqual(snapshot.read([1, 0]), rows[::-1])

    def test_analysis_uses_original_media_path_without_collapsing_observations(self):
        """Both boxes retain their own features and share the original review link."""
        rows = [
            {"observation_id": pk, "image_path": f"crop-{pk}.png", "original_path": "same.jpg",
             "embedding": [[pk, 0], {"descriptors": [[pk]]}]}
            for pk in (8, 9)
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = load_analysis_inputs(path)
        self.assertEqual(result.observation_id.tolist(), [8, 9])
        self.assertEqual(result.image_path.tolist(), ["same.jpg", "same.jpg"])
        self.assertEqual([json.loads(value) for value in result.embedding], [row["embedding"] for row in rows])

    def test_identify_uses_snapshot_in_original_matcher_dispatch(self):
        """Keep the stored query and reference identity in the worker task contract."""
        tree = ast.parse(Path(__file__).with_name("worker.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "predict")
        function.decorator_list = []
        namespace = {
            "pd": pd, "json": json, "Path": Path, "load_match_inputs": load_match_inputs,
            "os": Mock(environ={"DATABASE_BATCH_SIZE": "10", "ENCODING_BATCH_SIZE": "10"}),
            "logger": Mock(), "ProgressReporter": Mock(), "traceback": Mock(),
            "predict_full": Mock(return_value=({"pred_class_ids": [[3]]}, {3: "Lynx"})), "predict_batch": Mock(),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "worker.py", "exec"), namespace)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results.json"
            metadata = {"image_path": "crop.png", "original_path": "source.jpg", "mediafile_id": 5, "observation_id": 7}
            (root / "queries.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "references.jsonl").write_text("{}\n", encoding="utf-8")
            json_path = root / "query_metadata.json"
            json_path.write_text(json.dumps([metadata]), encoding="utf-8")
            result = namespace["predict"](Mock(), str(json_path), 42, str(output), feature_config={"weights_sha256": ""})
            self.assertEqual(result["status"], "DONE")
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["observation_ids"], [7])
        self.assertEqual(payload["pred_labels"], [["Lynx"]])
        self.assertEqual(payload["query_image_path"], ["source.jpg"])
        self.assertIsInstance(namespace["predict_full"].call_args.kwargs["stored_queries"], FeatureSnapshot)


if __name__ == "__main__":
    unittest.main()
