# Wildlife ReID domain invariants

These rules describe the intended domain model. Preserve them when changing models, imports, exports, processing pipelines, or the Observations UI.

## Media files and observations

- A media file may be an image or a video. Use **media file** in generic domain and UI text; use **image** only when an operation is genuinely image-specific.
- `AnimalObservation` is the atomic editable unit. A normal observation represents one observed object and at most one bbox in one media file.
- Multiple observations may belong to the same media file. Do not introduce a one-observation-per-media-file assumption.
- A no-detection placeholder is the explicit exception to the object/bbox meaning: it records that no object was found in a media file. It must not be treated as a real detected object.
- New media files receive a no-detection placeholder. When detections are produced, reuse that placeholder for the first real detection and create additional observations for additional detections.
- Grouping in the Observations UI is visual. Selecting, editing, importing, and exporting continue to operate on individual observations unless an action explicitly targets their containing media files.

## Sequences, future tracks, and identities

- A `Sequence` groups temporally related media files.
- A future `ObservationTrack` will connect observations that show the same visible object within exactly one sequence.
- A track is sequence-local. Every observation assigned to a track must belong to the track's sequence, and one observation may belong to at most one track.
- Tracks may contain gaps; the tracked object does not need an observation in every media file.
- No-detection placeholders must never be members of tracks.
- Do not assume one track member per media file until video observations have an explicit frame number or timestamp. For videos, the media-file reference alone does not uniquely locate a bbox in time.
- A track and an `IndividualIdentity` are different concepts. A track is a local temporal association inside one sequence; an identity can associate the same biological individual across multiple sequences, dates, and localities.
- A track may be unassigned to an identity. Do not infer global identity merely from track membership.
- Avoid duplicating taxon or identity as independent authoritative values on a future track unless synchronization rules are explicitly designed. Observation values remain the source of truth by default.

## Preparing track functionality

- Do not add an unused track table or nullable track field before track functionality is being implemented.
- When tracks are implemented, prefer a stable public UUID for import/export rather than relying on database integer IDs.
- Track creation, merging, splitting, and membership changes should go through shared service functions that validate sequence ownership and reject no-detection placeholders.
- Automatic tracking metadata should distinguish automatic, imported, and manual provenance and must not silently overwrite reviewed manual membership.
- `Group by: Track` should preserve observation-level selection. A collapsed track checkbox selects all editable observations in its displayed scope and must show an indeterminate state for partial selection.
- Decide pagination semantics before adding track grouping: either paginate whole tracks or clearly indicate when only part of a track is present on the current page.

