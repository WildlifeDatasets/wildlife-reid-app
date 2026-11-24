"""Read the ordered observation features exported by Django for one job."""

import json
from pathlib import Path

import pandas as pd

from .observation_embeddings import file_digest


class FeatureSnapshot:
    """Read only the requested feature rows from a shared JSON Lines file."""

    def __init__(self, path):
        self.path = path
        self.offsets = []
        with open(path, "rb") as stream:
            while True:
                offset = stream.tell()
                if not stream.readline():
                    break
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def read(self, rows):
        """Preserve the caller's row order, including ranked candidate order."""
        selected = []
        with open(self.path, "rb") as stream:
            for index in rows:
                stream.seek(self.offsets[index])
                selected.append(json.loads(stream.readline()))
        return selected


def load_match_inputs(metadata_path, weights_path, feature_config):
    """Load the selected comparison inputs prepared from the application database."""
    expected_weights = feature_config["weights_sha256"]
    if expected_weights and file_digest(weights_path) != expected_weights:
        raise ValueError("The model checkpoint changed after the features were prepared.")
    path = Path(metadata_path)
    metadata = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    queries = FeatureSnapshot(path.with_name("queries.jsonl"))
    references = FeatureSnapshot(path.with_name("references.jsonl"))
    if len(metadata) != len(queries):
        raise ValueError("The prepared query features and metadata have different sizes.")
    return metadata, queries, references


def load_analysis_inputs(path):
    """Use original media paths in analysis result links."""
    metadata = pd.read_json(path, lines=True, convert_dates=False)
    metadata["image_path"] = metadata["original_path"]
    metadata["embedding"] = metadata["embedding"].map(json.dumps)
    return metadata
