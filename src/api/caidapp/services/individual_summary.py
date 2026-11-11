"""Conservative individual counts shared by sequence and observation summaries."""


def summarize_individuals(mediafile_observations):
    """Count identities and simultaneous image objects from (media type, observations) pairs."""
    known_ids = set()
    minimum = 0
    unidentified_count = 0
    for media_type, observations in mediafile_observations:
        real_observations = [obs for obs in observations if not obs.is_no_detection_placeholder]
        file_known_ids = {obs.identity_id for obs in real_observations if obs.identity_id is not None}
        unknown = sum(obs.identity_id is None for obs in real_observations)
        known_ids.update(file_known_ids)
        unidentified_count += unknown
        if media_type == "image":
            minimum = max(minimum, len(file_known_ids) + unknown)
        elif real_observations:
            # Video observations have no frame/time coordinate and may show repeated sightings.
            minimum = max(minimum, 1)
    return {
        "known": len(known_ids),
        "minimum": max(minimum, len(known_ids)),
        "is_exact": unidentified_count == 0,
        "unidentified_observations": unidentified_count,
    }
