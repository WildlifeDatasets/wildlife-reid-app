"""Version identification inputs without loading media or embeddings."""

import hashlib
import json


def input_version(observation, mediafile):
    """Fingerprint the persisted observation and its source."""
    fields = [
        str(observation.uuid),
        observation.identity_id,
        observation.identity_is_representative,
        str(observation.updated_at),
        observation.taxon_id,
        observation.orientation,
        observation.bbox_x_center,
        observation.bbox_y_center,
        observation.bbox_width,
        observation.bbox_height,
        observation.is_no_detection_placeholder,
        mediafile.pk,
        mediafile.image_file.name,
        mediafile.mediafile.name,
        mediafile.media_type,
    ]
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest()


def model_version(model):
    """Record which identification model supplied a result."""
    if model is None:
        return {}
    return {
        "id": model.pk,
        "source": model.get_runtime_model_source(),
        "checkpoint": model.get_runtime_checkpoint_path(),
    }
