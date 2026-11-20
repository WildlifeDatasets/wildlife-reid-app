from collections import Counter
from datetime import timedelta

from caidapp.models import MediaFile, Sequence, UploadedArchive
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Max, Q


def _capture_groups(mediafiles, max_gap):
    group = []
    for media in mediafiles:
        if group:
            previous = group[-1]
            if (
                media.locality_id != previous.locality_id
                or media.captured_at is None
                or previous.captured_at is None
                or media.captured_at - previous.captured_at > max_gap
            ):
                yield group
                group = []
        group.append(media)
    if group:
        yield group


@transaction.atomic
def recalculate_upload_sequences(upload_id):
    """Rebuild one upload's sequences from the current locality and capture time."""
    archive = UploadedArchive.objects.select_for_update().get(pk=upload_id)
    workgroup = archive.owner.workgroup if archive.owner_id else None
    max_gap = timedelta(seconds=workgroup.sequence_time_limit if workgroup else 120)

    mediafiles = list(
        MediaFile.objects.select_for_update()
        .filter(parent=archive)
        .only("id", "sequence_id", "captured_at", "locality_id")
        .order_by("locality_id", "captured_at", "id")
    )
    sequences = Sequence.objects.filter(uploaded_archive=archive)
    local_sequence_ids = set(sequences.values_list("id", flat=True))
    assigned_ids = {media.sequence_id for media in mediafiles if media.sequence_id is not None}
    if assigned_ids - local_sequence_ids or MediaFile.objects.filter(sequence_id__in=local_sequence_ids).filter(
        Q(parent__isnull=True) | ~Q(parent=archive)
    ).exists():
        raise ValidationError("This upload has sequences shared with another collection.")

    max_local_id = sequences.aggregate(value=Max("local_id"))["value"]
    next_local_id = 0 if max_local_id is None else max_local_id + 1
    used_ids = set()
    changed = []
    for group in _capture_groups(mediafiles, max_gap):
        overlap = Counter(
            media.sequence_id for media in group if media.sequence_id is not None and media.sequence_id not in used_ids
        )
        if overlap:
            sequence_id = min(overlap, key=lambda candidate: (-overlap[candidate], candidate))
        else:
            sequence_id = Sequence.objects.create(uploaded_archive=archive, local_id=next_local_id).pk
            next_local_id += 1
        used_ids.add(sequence_id)
        for media in group:
            if media.sequence_id != sequence_id:
                media.sequence_id = sequence_id
                changed.append(media)

    MediaFile.objects.bulk_update(changed, ["sequence"], batch_size=500)
    sequences.filter(id__in=assigned_ids - used_ids, mediafile__isnull=True).delete()
    return len(used_ids)
