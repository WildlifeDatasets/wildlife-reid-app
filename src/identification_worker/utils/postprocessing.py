

def _sequence_voting(seq_preds, top_k):
    candidate_stats = {}

    for image_preds in seq_preds:
        for pred in image_preds:
            if pred.name not in candidate_stats:
                candidate_stats[pred.name] = {
                    "count": 0,
                    "total_score": 0.0,
                    "best_pred": pred,
                }

            candidate_stats[pred.name]["count"] += 1
            candidate_stats[pred.name]["total_score"] += pred.score

            if pred.score > candidate_stats[pred.name]["best_pred"].score:
                candidate_stats[pred.name]["best_pred"] = pred

    sorted_candidates = sorted(
        candidate_stats.values(),
        key=lambda x: (x["count"], x["total_score"]),
        reverse=True,
    )

    return [c["best_pred"] for c in sorted_candidates[:top_k]]


def _sequence_weighted_voting(seq_preds, top_k):
    candidate_stats = {}

    for image_preds in seq_preds:
        for pred in image_preds:
            if pred.name not in candidate_stats:
                candidate_stats[pred.name] = {
                    "total_score": 0.0,
                    "best_pred": pred,
                }

            candidate_stats[pred.name]["total_score"] += pred.score

            if pred.score > candidate_stats[pred.name]["best_pred"].score:
                candidate_stats[pred.name]["best_pred"] = pred

    sorted_candidates = sorted(
        candidate_stats.values(),
        key=lambda x: x["total_score"],
        reverse=True,
    )

    return [c["best_pred"] for c in sorted_candidates[:top_k]]


def _sequence_max_conf(seq_preds, top_k):
    best_image_preds = None
    best_score = -float("inf")

    for image_preds in seq_preds:
        max_score_in_image = max(pred.score for pred in image_preds)

        if max_score_in_image > best_score:
            best_score = max_score_in_image
            best_image_preds = image_preds

    return best_image_preds[:top_k]
