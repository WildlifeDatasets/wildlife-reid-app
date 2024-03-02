from copy import copy

import numpy as np
from numpy import arccos
from numpy.linalg import norm


def _get_angle(u, v):
    u = np.array(u)
    v = np.array(v)
    angle = arccos(u.dot(v) / (norm(u) * norm(v)))
    degrees = np.degrees(angle)
    return degrees


def get_angle(u, v):
    if len(np.array(u).shape) == 1:
        u = [u]
    if len(np.array(v).shape) == 1:
        v = [v]

    angles = []
    for _u in u:
        for _v in v:
            angle = _get_angle(_u, _v)
            angles.append(angle)
    return np.min(angles)


def get_angle_mat(vectors):
    angle_mat = np.ones([len(vectors), len(vectors)]) + 360
    for i, v in enumerate(vectors):
        for j, u in enumerate(vectors):
            if i <= j:
                continue
            angle_mat[i, j] = get_angle(v, u)
    return angle_mat


def assign_feature(query_data, feature, idx):
    for i in idx:
        query_data[i] = feature
    return query_data


def group_features(representation: list, positions: list | tuple):
    """Takes a list of elements and groups values in the list into lists based on provided index positions."""
    new_representation = []
    grouped = []
    for i, r in enumerate(representation):
        if i in positions:
            if not isinstance(r, list):
                r = [r]
            grouped.extend(r)
        else:
            new_representation.append(r)
    new_representation.append(grouped)
    return new_representation


def flatten(data):
    new_data = []
    for d in data:
        if isinstance(d, list):
            new_data.extend(d)
        else:
            new_data.append(d)
    return new_data


def feature_clustering(
    features,
    features_data,
    similarity,
    angle_threshold: float = 45,
    multiplier: float = 2,
):
    new_features = copy(features)

    # group oid data
    iid_to_oid = {
        row["mediafile_id"]: row["sequence_number"] for rid, row in features_data.iterrows()
    }
    oids = set(list(iid_to_oid.values()))
    oid_score = {o: [] for o in oids}
    oid_representations = {o: [] for o in oids}
    oid_idx = {o: [] for o in oids}
    for idx, (feature, iid, scr_row) in enumerate(
        zip(features, features_data["mediafile_id"], similarity)
    ):
        scr = np.max(scr_row)
        oid = iid_to_oid[iid]

        oid_score[oid].append(scr)
        oid_representations[oid].append(feature)
        oid_idx[oid].append(idx)

    for oid in oids:
        # get oid data
        score = oid_score[oid]
        representation = oid_representations[oid]
        idx = oid_idx[oid]
        if not score or not representation or not idx:
            continue

        # only one image in sequence
        if len(representation) == 1:
            continue

        distance_matrix = get_angle_mat(representation)
        threshold_min_angle = np.min(distance_matrix)
        positions = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)

        # no two vector are close enough, select image with best score
        if threshold_min_angle > angle_threshold:
            top_score_idx = np.argmax(score)
            new_features = assign_feature(new_features, representation[top_score_idx], idx)
            continue

        representation = group_features(representation, positions)
        idx = group_features(idx, positions)

        # iteratively group data
        for _ in range(len(representation) - 1):
            distance_matrix = get_angle_mat(representation)
            min_angle = np.min(distance_matrix)
            positions = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)

            if min_angle > (multiplier * threshold_min_angle):
                break

            representation = group_features(representation, positions)
            idx = group_features(idx, positions)

        # get final vector
        group_average_angle = []
        for group_idx in range(len(idx)):
            if not isinstance(idx[group_idx], list):
                group_average_angle.append(361)
                continue

            distance_matrix = get_angle_mat(representation[group_idx])
            distance_matrix[distance_matrix > 360] = 0
            average_angle = np.sum(distance_matrix) / np.sum(distance_matrix > 0)

            group_average_angle.append(average_angle)
        min_angle_idx = np.argmin(group_average_angle)
        new_representation = np.sum(np.array(representation[min_angle_idx]), 0)
        new_features = assign_feature(new_features, new_representation, flatten(idx))

    return new_features


def feature_average(features, features_data):
    new_features = copy(features)

    iid_to_oid = {
        row["mediafile_id"]: row["sequence_number"] for rid, row in features_data.iterrows()
    }
    oids = set(list(iid_to_oid.values()))
    oid_idx = {o: [] for o in oids}
    oid_representations = {o: [] for o in oids}

    for idx, (feature, iid) in enumerate(zip(new_features, features_data["mediafile_id"])):
        oid = iid_to_oid[iid]
        oid_representations[oid].append(feature)
        oid_idx[oid].append(idx)

    for oid in oids:
        if not oid_representations[oid]:
            continue
        new_feature = np.mean(np.array(oid_representations[oid]), 0)
        new_features = assign_feature(new_features, new_feature, oid_idx[oid])

    return new_features


def feature_top(features, features_data, similarity, method):
    new_features = copy(features)

    iid_to_oid = {
        row["mediafile_id"]: row["sequence_number"] for rid, row in features_data.iterrows()
    }
    oids = set(list(iid_to_oid.values()))
    oid_score = {o: [] for o in oids}
    oid_representations = {o: [] for o in oids}
    oid_idx = {o: [] for o in oids}
    oid_individual = {o: [] for o in oids}
    for idx, (feature, iid, scr_row) in enumerate(
        zip(features, features_data["mediafile_id"], similarity)
    ):
        scr = np.max(scr_row)
        individual = np.argmax(scr_row)
        oid = iid_to_oid[iid]

        oid_score[oid].append(scr)
        oid_representations[oid].append(feature)
        oid_idx[oid].append(idx)
        oid_individual[oid].append(individual)

    for oid in oids:
        if not oid_representations[oid]:
            continue
        if len(oid_representations[oid]) == 1:
            continue

        if method == "top_score":
            """Assign feature vector with max score to all with same observation id."""
            idx = np.argmax(oid_score[oid])
            new_feature = oid_representations[oid][idx]
        elif method == "top_frequent":
            """Get most frequent individual and assign feature vector with max/mean score to all with same
            observation id."""
            unique, counts = np.unique(oid_individual[oid], return_counts=True)
            idx = np.argmax(counts)
            individual_id = unique[idx]
            individual_representations = [
                r
                for r, i in zip(oid_representations[oid], oid_individual[oid])
                if individual_id == i
            ]
            individual_scores = [
                s for s, i in zip(oid_score[oid], oid_individual[oid]) if individual_id == i
            ]
            new_feature = individual_representations[np.argmax(individual_scores)]

        new_features = assign_feature(new_features, new_feature, oid_idx[oid])

    return new_features
