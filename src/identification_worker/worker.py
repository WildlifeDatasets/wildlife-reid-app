import json
import logging
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np

print(f"numpy version: {np.__version__}")
import pandas as pd
import torch
from celery import Celery, shared_task
from gpu_monitor import register_gpu_monitor
from progress import ProgressReporter
from train_model import train_identification_model
from wildlife_model_cache import log_model_operation
from wildlife_tools.data import FeatureDataset
from wildlife_tools.similarity.pairwise.collectors import CollectAll
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue

from utils import config
from utils.feature_snapshot import FeatureSnapshot, load_analysis_inputs, load_match_inputs
from utils.embedding_processing import EmbeddingProcessing
from utils.inference_identification import (
    calibrate_models,
    compute_partial,
    del_models,
    get_identification_model,
    get_keypoints,
    identify,
    identify_from_similarity,
    prepare_feature_types,
)
from utils.log import setup_logging

os.environ.setdefault("TZ", "Europe/Prague")
if hasattr(time, "tzset"):
    time.tzset()

setup_logging()
logger = logging.getLogger("app")

logger.debug(f"{config.RABBITMQ_URL=}")
logger.debug(f"{config.REDIS_URL=}")

identification_worker = Celery("identification_worker", broker=config.RABBITMQ_URL, backend=config.REDIS_URL)
identification_worker.conf.timezone = os.environ.get("TZ", "Europe/Prague")
identification_worker.conf.enable_utc = False
identification_worker.conf.task_track_started = True
register_gpu_monitor(identification_worker)


@identification_worker.task(name="compute_observation_embeddings")
def compute_observation_embeddings(manifest_path):
    """Compute missing global and ALIKED features for a set of observations.

    `manifest_path` points to `input.json` (UTF-8 JSON) on shared storage.
    It describes the feature profile, cache keys and observations, including
    the source path, bbox and input version for each observation. Write newly
    computed features and their profile to `output.json` next to the manifest
    and return its path for Django to import.
    """
    from utils.observation_embeddings import compute_manifest

    return compute_manifest(manifest_path)


@identification_worker.task(bind=True, name="train_identification")
def train_identification(
    self,
    input_metadata_file: str,
    organization_id: int,
    identification_model: dict = None,
    **kwargs,
):
    """Train an identification model from labeled observations.

    `input_metadata_file` points to a CSV with `image_path`, `label` and
    `sequence_number` columns. The trainer derives the masked-image paths
    from `image_path`. `identification_model` selects the base architecture,
    optional initial weights and output checkpoint (.pth). Return the task
    status and error details if training fails.
    """
    logger.debug(f"{identification_model=}")
    if identification_model is None:
        identification_model = {
            "name": "derived from LynxV4-MegaDescriptor-v2-T-256",
            "base_model_source": "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
            "initial_weights_path": "",
            "output_weights_path": "/models/model1/LynxV4-MegaDescriptor-v2-T-256.pth",
        }
    # outputdir = Path(identification_model["output_weights_path"]).parent
    try:
        # outputdir.mkdir(parents=True, exist_ok=True)
        # pass
        train_identification_model(
            input_metadata_file=input_metadata_file,
            organization_id=organization_id,
            identification_model=identification_model,
            **kwargs,
        )
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        return {"status": "ERROR", "error": error}

    return {"status": "DONE"}


# @identification_worker.task(bind=True, name="iworker_simple_log")
# def iworker_simple_log(self, *args, **kwargs):
#     """Simple log task."""
#     logger.info(f"Applying simple log task with args: {args=}, {kwargs=}.")
#     return {"status": "DONE"}
#
#
# @shared_task(bind=True, name="shared_simple_log")
# def shared_simple_log(self, *args, **kwargs):
#     """Simple log task."""
#     logger.info(f"Applying simple log task with args: {args=}, {kwargs=}.")
#     return {"status": "DONE"}


def load_features(references, *, start=0, end=None, rows=None):
    """Read the feature pairs and metadata of the selected reference rows.

    `references` is a FeatureSnapshot backed by `references.jsonl`. Read the
    half-open range `start:end` or the explicit zero-based indices in `rows`.
    Return the feature pairs and a parallel DataFrame with the remaining
    fields, including the reference observation IDs.
    """
    table = pd.DataFrame(references.read(rows if rows is not None else range(len(references))[start:end]))
    return table.pop("embedding").tolist(), table


def get_priority_pairs_from_parts(priority_parts: list, image_budget: int):
    """Select up to `image_budget` references per query for local matching.

    Each part is a query-by-reference priority matrix for one reference
    batch. Parts must share the query order and follow the reference
    snapshot order. Return the (query index, reference index) pairs and
    the set of reference indices to load again.
    """
    priority = np.concatenate(priority_parts, 1)

    _, idx1 = torch.topk(torch.tensor(priority), min(image_budget, priority.shape[1]))
    idx0 = np.indices(idx1.numpy().shape)[0]
    grid_indices = np.stack([idx0.flatten(), idx1.flatten()]).T

    flatten_idx1 = []
    for _idx1 in idx1:
        flatten_idx1.extend(_idx1.numpy())
    return grid_indices, set(flatten_idx1)


def predict_full(
    metadata: pd.DataFrame,
    references: FeatureSnapshot,
    identification_model_path,
    identification_model_weights_path="",
    top_k: int = 1,
    progress: ProgressReporter | None = None,
    *,
    stored_queries: FeatureSnapshot,
):
    """Compare all prepared queries and references together in memory.

    `metadata` is the DataFrame for the observations being identified,
    loaded from `query_metadata.json`. Its indices locate the corresponding
    global and ALIKED features in `stored_queries` (`queries.jsonl`).
    `references` (`references.jsonl`) holds labeled observations with their
    metadata and features. Return ranked results and an identity-ID-to-label
    mapping.
    """
    # Read the full reference snapshot.
    if progress:
        progress.stage("load_references", "Loading reference embeddings")
    database_features, reference_images = load_features(references)

    # Load stored query features in metadata order.
    if progress:
        progress.stage("identify", "Loading query features")
    query_features = [row["embedding"] for row in stored_queries.read(metadata.index)]

    # Build the metadata expected by WildFusion.
    query_metadata = pd.DataFrame(
        {
            "path": metadata["image_path"],
            "identity": [-1] * len(metadata["image_path"]),
            "split": ["test"] * len(metadata["image_path"]),
            "sequence_number": metadata["sequence_number"] if "sequence_number" in metadata else None,
        }
    )
    database_metadata = pd.DataFrame(
        {
            "path": reference_images["image_path"],
            "identity": reference_images["class_id"],
            "split": ["train"] * len(reference_images["class_id"]),
            "sequence_number": reference_images["sequence_number"] if "sequence_number" in reference_images else None,
        }
    )

    database_metadata["observation_id"] = reference_images["observation_id"]
    database_metadata["review_version"] = reference_images["review_version"]

    if progress:
        progress.stage("identify", "Comparing query images with references")
        progress.update(1, 3)
    identification_output = identify(
        query_features=query_features,
        database_features=database_features,
        query_metadata=query_metadata,
        database_metadata=database_metadata,
        identification_model_path=identification_model_path,
        identification_model_weights_path=identification_model_weights_path,
        top_k=top_k,
        cal_images=int(os.environ["CALIBRATION_IMAGES"]),
        image_budget=int(os.environ["IMAGE_BUDGET"]),
        use_sequence_post_processing=False,
    )
    if progress:
        progress.update(1, 1)

    id2label = dict(zip(reference_images["class_id"], reference_images["label"]))

    return identification_output, id2label


def predict_batch(
    metadata: pd.DataFrame,
    references: FeatureSnapshot,
    database_size: int,
    identification_model_path: str,
    identification_model_weights_path: str = "",
    top_k: int = 1,
    progress: ProgressReporter | None = None,
    *,
    stored_queries: FeatureSnapshot,
):
    """Compare the prepared query and reference snapshots in batches.

    `metadata` comes from `query_metadata.json` and indexes the same rows in
    `stored_queries` (`queries.jsonl`). `references` is the ordered snapshot
    of labeled observations (`references.jsonl`); `database_size` is its row
    count. Use the same row order for calibration, candidate scoring and
    paired-point extraction. Return ranked results and identity labels.
    """
    database_batch_size = int(os.environ["DATABASE_BATCH_SIZE"])
    encoding_batch_size = int(os.environ["ENCODING_BATCH_SIZE"])
    cal_images = int(os.environ["CALIBRATION_IMAGES"])
    image_budget = int(os.environ["IMAGE_BUDGET"])

    # initialize and calibrate models
    if progress:
        progress.stage("load_references", "Initializing models and calibration data")
    get_identification_model(identification_model_path, model_checkpoint=identification_model_weights_path)

    # TODO: get random calibration images?
    calibration_features, reference_images = load_features(references, start=0, end=cal_images)
    calibration_metadata = pd.DataFrame(
        {
            "path": reference_images["image_path"],
            "identity": reference_images["class_id"],
            "split": ["train"] * len(reference_images["class_id"]),
        }
    )
    calibrate_models(calibration_features, calibration_metadata)
    if progress:
        progress.update(1, 1)

    # prepare query metadata splits
    target_num_splits = math.ceil(len(metadata) / encoding_batch_size)
    metadata_splits = np.array_split(metadata, target_num_splits)
    # TODO: split by sequence_id

    # Split the reference snapshot into contiguous row ranges.
    database_split_idx = [*range(0, database_size, database_batch_size), database_size]

    # iterate over query splits/batches
    identification_output = {}
    for qi, _metadata in enumerate(metadata_splits):
        progress_str = f"[{qi + 1}/{target_num_splits}] - {len(_metadata)}"
        logger.debug(f"predict_batch: {progress_str}")
        if progress:
            progress.stage("identify", f"Loading query batch {qi + 1}/{target_num_splits}")
            progress.update(qi, target_num_splits)
        query_features = [row["embedding"] for row in stored_queries.read(_metadata.index)]
        # prepare query metadata
        query_metadata = pd.DataFrame(
            {
                "path": _metadata["image_path"],
                "identity": [-1] * len(_metadata["image_path"]),
                "split": ["test"] * len(_metadata["image_path"]),
                "sequence_number": _metadata["sequence_number"] if "sequence_number" in _metadata else None,
            }
        )

        logger.debug("*" * 50)
        logger.debug("STARTING PRIORITY CALCULATION")
        logger.debug("*" * 50)

        # priority calculation iterate over
        full_database_metadata = []
        priority_matrix = []
        for db_idx in range(1, len(database_split_idx)):
            if progress:
                progress.update(
                    qi + (0.25 * db_idx / max(len(database_split_idx) - 1, 1)),
                    target_num_splits,
                    message=f"Computing priority matrix {db_idx}/{len(database_split_idx) - 1}",
                )
            database_features, reference_images = load_features(
                references,
                start=database_split_idx[db_idx - 1],
                end=database_split_idx[db_idx],
            )
            # prepare database metadata
            database_metadata = pd.DataFrame(
                {
                    "path": reference_images["image_path"],
                    "identity": reference_images["class_id"],
                    "split": ["train"] * len(reference_images["class_id"]),
                    "label": reference_images["label"],
                    "sequence_number": (
                        reference_images["sequence_number"] if "sequence_number" in reference_images else None
                    ),
                }
            )

            database_metadata["observation_id"] = reference_images["observation_id"].to_numpy()
            database_metadata["review_version"] = reference_images["review_version"].to_numpy()

            # accumulate priority matrix for query_features
            logger.debug("Computing priority matrix")
            _priority_matrix = compute_partial(
                query_features=query_features,
                database_features=database_features,
                query_metadata=query_metadata,
                database_metadata=database_metadata,
                identification_model_path=identification_model_path,
                target="priority",
                identification_model_weights_path=identification_model_weights_path,
            )
            priority_matrix.append(_priority_matrix)
            full_database_metadata.append(database_metadata)
            logger.debug(f"Priority matrix shape: {_priority_matrix.shape}")

        full_database_metadata = pd.concat(full_database_metadata).reset_index(drop=True)

        # get priority pairs and database idx
        pairs, database_idx = get_priority_pairs_from_parts(priority_matrix, image_budget=image_budget)
        database_idx = [int(i) for i in database_idx]
        database_idx.sort()

        logger.debug("*" * 50)
        logger.debug("STARTING SCORE CALCULATION")
        logger.debug("*" * 50)
        logger.debug(f"Unique database indexes: {len(database_idx)}")

        # score calculation
        num_local_descriptors = 2
        scores = [np.zeros([len(query_features), database_size]) for _ in range(num_local_descriptors)]

        split_idx = [*range(0, len(database_idx), database_batch_size), len(database_idx)]

        for sidx in range(1, len(split_idx)):
            if progress:
                progress.update(
                    qi + (0.5 + 0.25 * sidx / max(len(split_idx) - 1, 1)),
                    target_num_splits,
                    message=f"Computing score matrix {sidx}/{len(split_idx) - 1}",
                )
            _database_idx = database_idx[split_idx[(sidx - 1)] : split_idx[sidx]]
            _pairs = [p for p in pairs if p[1] in _database_idx]

            # replace database idx with idx in list
            idx_to_dbidx = {idx: dbidx for idx, dbidx in enumerate(_database_idx)}
            dbidx_to_idx = {dbidx: idx for idx, dbidx in enumerate(_database_idx)}
            for pi in range(len(_pairs)):
                _pairs[pi][1] = dbidx_to_idx[_pairs[pi][1]]

            # get features
            database_features, reference_images = load_features(references, rows=_database_idx)

            # prepare database metadata
            database_metadata = pd.DataFrame(
                {
                    "path": reference_images["image_path"],
                    "identity": reference_images["class_id"],
                    "split": ["train"] * len(reference_images["class_id"]),
                    "sequence_number": (
                        reference_images["sequence_number"] if "sequence_number" in reference_images else None
                    ),
                }
            )

            logger.debug("Computing score matrix")
            _partial_scores = compute_partial(
                query_features=query_features,
                database_features=database_features,
                query_metadata=query_metadata,
                database_metadata=database_metadata,
                identification_model_path=identification_model_path,
                target="scores",
                pairs=_pairs,
                identification_model_weights_path=identification_model_weights_path,
            )

            # accumulate results in pre-alocated matrix
            for pair in _pairs:
                qidx, idx = pair
                dbidx = idx_to_dbidx[idx]
                for score_idx in range(len(_partial_scores)):
                    score = _partial_scores[score_idx][qidx, idx]
                    scores[score_idx][qidx, dbidx] = score

        # combine scores
        similarity = np.mean(scores, axis=0)
        similarity = np.where(similarity == 0, -np.inf, similarity)

        # calculate and merge results
        _identification_output, result_idx = identify_from_similarity(
            similarity,
            full_database_metadata,
            query_metadata,
            top_k=top_k,
            post_process=None,
        )

        # calculate keypoints
        collector = CollectAll()
        with log_model_operation("LightGlue model load", "ALIKED matcher"):
            keypoint_matcher = MatchLightGlue(features="aliked", collector=collector)

        keypoints = []
        total_keypoints = max(len(result_idx), 1)
        for keypoint_i, (qidx, didx) in enumerate(result_idx.items(), start=1):
            if progress:
                progress.update(
                    qi + (0.75 + 0.25 * keypoint_i / total_keypoints),
                    target_num_splits,
                    message=f"Computing keypoints {keypoint_i}/{total_keypoints}",
                )
            query_aliked_features, _ = prepare_feature_types([query_features[qidx]])
            keypoint_query_features = FeatureDataset(query_aliked_features, query_metadata.iloc[[qidx]])

            database_features, reference_images = load_features(references, rows=didx)
            database_metadata = pd.DataFrame(
                {
                    "path": reference_images["image_path"],
                    "identity": reference_images["class_id"],
                    "split": ["train"] * len(reference_images["class_id"]),
                    "sequence_number": (
                        reference_images["sequence_number"] if "sequence_number" in reference_images else None
                    ),
                }
            )

            database_aliked_features, _ = prepare_feature_types(database_features)
            keypoint_database_features = FeatureDataset(database_aliked_features, database_metadata)

            _keypoints = get_keypoints(keypoint_matcher, keypoint_query_features, keypoint_database_features, max_kp=10)
            keypoints.append(_keypoints)

        _identification_output["keypoints"] = keypoints

        # merge batch outputs
        if not identification_output:
            identification_output = _identification_output
        else:
            for k, v in _identification_output.items():
                identification_output[k].extend(v)
        if progress:
            progress.update(qi + 1, target_num_splits)

    id2label = dict(zip(full_database_metadata["identity"], full_database_metadata["label"]))

    return identification_output, id2label


@identification_worker.task(bind=True, name="identify")
def predict(
    self,
    input_metadata_file_path: str,
    organization_id: int,
    output_json_file_path: str,
    top_k: int = 1,
    sequence_time: str = "480s",
    identification_model: dict = None,
    *,
    feature_config: dict,
    **kwargs,
):
    """Run the Celery `identify` task over prepared observation features.

    `input_metadata_file_path` points to `query_metadata.json` on shared
    storage. Its directory must also contain `queries.jsonl` in matching row
    order and `references.jsonl` with the selected labeled observations.
    `feature_config` provides the expected checkpoint digest.

    Choose the full or batched matcher and write JSON to
    `output_json_file_path`. The result contains query and reference
    observation IDs, candidate identities, scores and paired points.
    Return the task status and the output path on success.
    """
    logger.debug(f"{identification_model=}")

    if identification_model is None:
        identification_model = {
            "name": "",
            # model_source is required: HF/timm model used to create the architecture.
            "model_source": "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
            # weights_path is optional: local checkpoint loaded into model_source before inference.
            "weights_path": "",
        }

    # identification_model["name"]
    # identification_model["model_source"]
    try:
        progress = ProgressReporter(self, operation="identify")
        progress.stage("load_metadata", "Loading identification metadata")
        logger.info(f"Applying identify task with args: {input_metadata_file_path=}, {organization_id=}.")
        logger.debug(f"celery task id={self.request.id=}")

        # Load metadata JSON and its two companion JSONL snapshots.
        metadata, queries, references = load_match_inputs(
            input_metadata_file_path, identification_model.get("weights_path", ""), feature_config
        )
        progress.update(1, 1)
        if len(metadata) == 0:
            logger.info("Input data is empty. Finishing the job.")
            out = {"status": "ERROR", "error": "Input data is empty."}
        else:
            progress.stage("load_references", "Checking reference features")
            database_size = len(references)
            progress.update(1, 1)

            if database_size == 0:
                logger.info(f"Identification worker was not initialized for {organization_id=}. " "Finishing the job.")
                out = {
                    "status": "ERROR",
                    "error": "Identification worker was not initialized.",
                }
            else:
                logger.debug(f"Starting identification with: {len(metadata)} query files.")
                query_image_path = list(metadata.original_path)
                query_masked_path = list(metadata.image_path)

                database_batch_size = int(os.environ["DATABASE_BATCH_SIZE"])
                encoding_batch_size = int(os.environ["ENCODING_BATCH_SIZE"])

                if (database_batch_size >= database_size) and (encoding_batch_size >= len(metadata)):
                    logger.info("Starting full identification.")
                    # model_source creates the architecture;
                    # optional weights_path loads locally trained weights into it.
                    identification_model_path = identification_model["model_source"]
                    identification_model_weights_path = identification_model.get("weights_path", "")
                    identification_output, id2label = predict_full(
                        metadata,
                        references,
                        identification_model_path=identification_model_path,
                        identification_model_weights_path=identification_model_weights_path,
                        top_k=top_k,
                        progress=progress,
                        stored_queries=queries,
                    )
                else:
                    logger.info("Starting batched identification.")
                    # model_source creates the architecture;
                    # optional weights_path loads locally trained weights into it.
                    identification_model_path = identification_model["model_source"]
                    identification_model_weights_path = identification_model.get("weights_path", "")
                    identification_output, id2label = predict_batch(
                        metadata,
                        references,
                        database_size,
                        identification_model_path=identification_model_path,
                        identification_model_weights_path=identification_model_weights_path,
                        top_k=top_k,
                        progress=progress,
                        stored_queries=queries,
                    )

                progress.stage("save_output", "Saving identification suggestions")
                pred_labels = [[id2label[x] for x in row] for row in identification_output["pred_class_ids"]]
                identification_output["mediafile_ids"] = metadata["mediafile_id"].tolist()
                if "observation_id" in metadata:
                    identification_output["observation_ids"] = [
                        None if pd.isna(observation_id) else int(observation_id)
                        for observation_id in metadata["observation_id"].tolist()
                    ]
                identification_output["pred_labels"] = pred_labels
                identification_output["query_image_path"] = query_image_path
                identification_output["query_masked_path"] = query_masked_path

                # save output to json
                with open(output_json_file_path, "w") as f:
                    json.dump(identification_output, f)
                progress.update(1, 1)

                progress.stage("finalize", "Finalizing identification")
                logger.info("Finished identify processing.")
                progress.update(1, 1)
                out = {"status": "DONE", "output_json_file": output_json_file_path}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out


@identification_worker.task(bind=True, name="detect_identification_outliers")
def detect_identification_outliers(
    self,
    organization_id: int,
    input_metadata_file: str = "",
    min_cluster_size: int = 3,
    # mediafile_paths=None,
    **kwargs,
):
    """Flag potentially misassigned observations using their global vectors.

    `input_metadata_file` is a JSONL snapshot from Django with a labeled
    observation on each line, including its original media path and feature
    pair. Only identities with at least `min_cluster_size` usable rows
    participate. Return suggestions with the current and alternative
    identities, their similarity difference and the media paths for review.
    """
    try:
        # Load the labeled observation snapshot (JSONL).
        metadata = load_analysis_inputs(input_metadata_file)
        assert "image_path" in metadata
        assert "class_id" in metadata, "Identity id should be in `class_id` column"
        assert "label" in metadata, "Label should be in `label` column"

        mediafile_paths = list(metadata["image_path"])
        class_ids = list(metadata["class_id"])

        # Drop rows with missing embeddings
        logger.info(f"Dropping rows with missing embeddings: {metadata.embedding.isna().sum()}")
        metadata = metadata.dropna(subset=["embedding"])

        # Drop class_ids that are underrepresented < min_cluster_size
        class_ids = metadata["class_id"].value_counts()
        class_ids = class_ids[class_ids < min_cluster_size]
        logger.info(f"Dropping class_ids that are underrepresented < {min_cluster_size}: {list(class_ids.index)}")
        metadata = metadata[~metadata["class_id"].isin(class_ids.index)].reset_index(drop=True).copy()

        # Embeddings are saved as a string and contain megadescriptor and local descriptor features [[mega, local], ...]
        embeddings = [json.loads(e) for e in metadata["embedding"]]
        metadata["embedding"] = [r[0] for r in embeddings]

        # Calculate the likely mislabeled embeddings
        embeddings = np.array(list(metadata["embedding"]))
        logger.info(f"Calculating likely mislabeled embeddings for {len(metadata)} embeddings.")
        ep = EmbeddingProcessing(
            embeddings,
            metadata=metadata,
            label_col="class_id",
        )
        suspects = ep.likely_mislabeled(margin=0.0, min_cluster_size=min_cluster_size)
        logger.debug("Calculated %s suspect rows for %s metadata rows.", len(suspects), len(metadata))

        # Add the suspects data to the metadata
        suspects_by_idx = suspects.set_index("idx").reindex(metadata.index)
        best_other_image_path_by_idx = {}
        for row in suspects.itertuples(index=False):
            best_other_metadata_idx = ep.best_other_member_index(row.idx, row.best_other_label)
            best_other_image_path_by_idx[row.idx] = (
                None if best_other_metadata_idx is None else metadata.iloc[best_other_metadata_idx]["image_path"]
            )

        best_other_image_path = pd.Series(best_other_image_path_by_idx).reindex(metadata.index)
        metadata["own_similarity"] = suspects_by_idx["own_similarity"].to_numpy()
        metadata["best_other_similarity"] = suspects_by_idx["best_other_similarity"].to_numpy()
        metadata["delta"] = suspects_by_idx["delta"].to_numpy()
        metadata["best_other_label"] = suspects_by_idx["best_other_label"].to_numpy()
        metadata["best_other_image_path"] = best_other_image_path.to_numpy()
        metadata["is_suspect"] = suspects_by_idx["is_suspect"].to_numpy()

        # Add reduced embeddings to the metadata
        logger.info("Starting t-SNE reduction for %s embeddings.", len(metadata))
        tsne_start = time.perf_counter()
        try:
            tsne_embeddings = ep.reduce_embeddings(method="tsne", n_components=2, random_state=0)
        except Exception as e:
            logger.error(f"Error in tsne embeddings: {e}")
            tsne_embeddings = None
        logger.info("Finished t-SNE reduction in %.2f s.", time.perf_counter() - tsne_start)
        if tsne_embeddings is not None:
            metadata["tsne_x"] = tsne_embeddings[:, 0]
            metadata["tsne_y"] = tsne_embeddings[:, 1]

        logger.info("Starting UMAP reduction for %s embeddings.", len(metadata))
        umap_start = time.perf_counter()
        try:
            umap_embeddings = ep.reduce_embeddings(method="umap", n_components=2, random_state=0)
        except Exception as e:
            logger.error(f"Error in umap embeddings: {e}")
            umap_embeddings = None
        logger.info("Finished UMAP reduction in %.2f s.", time.perf_counter() - umap_start)
        if umap_embeddings is not None:
            metadata["umap_x"] = umap_embeddings[:, 0]
            metadata["umap_y"] = umap_embeddings[:, 1]

        # Create suggestions
        logger.info("Creating outlier suggestions.")
        suggestions = []
        # try:
        for idx, row in metadata.iterrows():
            if not bool(row.get("is_suspect")):
                continue

            best_other_label = row.get("best_other_label")
            best_other_image_path = row.get("best_other_image_path")
            delta = row.get("delta")

            if pd.isna(best_other_label):
                logger.warning(
                    "Skipping suspect idx=%s mediafile_id=%s image_path=%s \
                    class_id=%s because best_other_label is missing.",
                    idx,
                    row.get("mediafile_id"),
                    row.get("image_path"),
                    row.get("class_id"),
                )
                continue

            suggestions.append(
                {
                    "query_idx": idx,
                    "suspicious_path": row["image_path"],
                    "current_identity_id": int(row["class_id"]),
                    "reason": (
                        f"The embedding is likely to be mislabeled. "
                        f"The own similarity is {row['own_similarity']:.2f} "
                        f"and the best other similarity is {row['best_other_similarity']:.2f}."
                    ),
                    "suggestions": [
                        {
                            "identity_id": int(best_other_label),
                            "mediafile_path": None if pd.isna(best_other_image_path) else best_other_image_path,
                            "score": None if pd.isna(delta) else float(delta),
                            "reason": (
                                (
                                    f"Delta is {delta:.2f}. Delta > 0 means the embedding is closer "
                                    "to the best other cluster center than to its own center."
                                )
                                if not pd.isna(delta)
                                else "No delta available."
                            ),
                        },
                    ],
                }
            )

        # Remove embeddings and save the metadata file in a new file with "extended" in the filename
        if "embedding" in metadata.columns:
            metadata = metadata.drop(columns=["embedding"])

        # Save metadata with extended information
        extended_metadata_file = input_metadata_file.replace(".csv", "_extended.csv")
        metadata.to_csv(extended_metadata_file, index=False)
        logger.info(f"Saved extended metadata file to {extended_metadata_file}")
        logger.info("Prepared %s suspect suggestions.", len(suggestions))

    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
        return out

    return {
        "status": "DONE",
        "message": "Demo identification outlier output with suggestions for the first two media files.",
        "suggestions": suggestions,
        "organization_id": organization_id,
        "input_metadata_file": input_metadata_file,
    }


def _most_similar_cross_label_pairs(embeddings, class_ids, top_k: int = 20, image_paths=None) -> pd.DataFrame:
    """Return the most similar unordered pairs with different identity labels.

    `embeddings` contains one global vector per row of `class_ids`.
    When `image_paths` is supplied in the same order, exclude pairs from
    the same source media file. Return row indices and cosine similarities.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    class_ids = np.asarray(class_ids)
    columns = ["idx_a", "idx_b", "similarity"]
    if embeddings.shape[0] < 2 or top_k <= 0:
        return pd.DataFrame(columns=columns)

    # Cosine similarity of unique unordered pairs with different class_id
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    # Select unique pairs
    idx_a, idx_b = np.triu_indices(len(embeddings), k=1)

    # Calculate cosine similarity between pairs
    values = (embeddings @ embeddings.T)[idx_a, idx_b]

    # Filter pairs that do not have different class_id
    cross_label_mask = class_ids[idx_a] != class_ids[idx_b]
    if image_paths is not None:
        paths = np.asarray(image_paths)
        cross_label_mask &= paths[idx_a] != paths[idx_b]
    idx_a, idx_b, values = idx_a[cross_label_mask], idx_b[cross_label_mask], values[cross_label_mask]

    if values.size == 0:
        return pd.DataFrame(columns=columns)

    # Select the top-k pairs without fully sorting when possible
    if top_k >= values.size:
        order = np.argsort(-values)
    else:
        part = np.argpartition(-values, top_k - 1)[:top_k]
        order = part[np.argsort(-values[part])]

    return pd.DataFrame(
        {
            "idx_a": idx_a[order].astype(int, copy=False),
            "idx_b": idx_b[order].astype(int, copy=False),
            "similarity": values[order].astype(float, copy=False),
        }
    ).reset_index(drop=True)


def _pair_payload(metadata: pd.DataFrame, idx_a, idx_b, similarity, mega_similarity=None) -> dict:
    """Serialize two positional metadata rows for identity comparison.

    The DataFrame provides `image_path`, `class_id` and `label` per row.
    Include the supplied score and, if present, the global cosine score
    used to shortlist the pair before local matching.
    """
    payload = {"idx_a": int(idx_a), "idx_b": int(idx_b), "similarity": float(similarity)}
    for side, idx in (("a", idx_a), ("b", idx_b)):
        row = metadata.iloc[int(idx)]
        payload[f"path_{side}"] = row["image_path"]
        payload[f"class_id_{side}"] = int(row["class_id"])
        payload[f"label_{side}"] = None if pd.isna(row["label"]) else row["label"]
    if mega_similarity is not None:
        payload["mega_similarity"] = float(mega_similarity)
    return payload


@identification_worker.task(bind=True, name="detect_most_similar")
def detect_most_similar(
    self,
    organization_id: int,
    input_metadata_file: str = "",
    top_k: int = 20,
    aliked_pair_budget: int = 200,
    output_csv_file: str | None = None,
    **kwargs,
):
    """Find visually similar observations with different known identities.

    `input_metadata_file` is a JSONL snapshot of labeled observations with the
    original media path, class ID, label and global and ALIKED features.
    Global cosine similarity shortlists up to `aliked_pair_budget` pairs
    for LightGlue. Return up to `top_k` results per method and export both
    sets to `output_csv_file` or a CSV alongside the input.
    """
    try:
        # Load the labeled observation snapshot (JSONL).
        metadata = load_analysis_inputs(input_metadata_file)
        assert "image_path" in metadata
        assert "class_id" in metadata, "Identity id should be in `class_id` column"
        assert "label" in metadata, "Label should be in `label` column"

        # Drop rows with missing embeddings
        logger.info(f"Dropping rows with missing embeddings: {metadata.embedding.isna().sum()}")
        metadata = metadata.dropna(subset=["embedding"])

        # Embeddings are stored as [mega, local]
        parsed_features = [json.loads(e) for e in metadata["embedding"]]
        aliked_embeddings, mega_embeddings = prepare_feature_types(parsed_features)

        # Rank all cross-identity pairs by MegaDescriptor cosine
        mega_shortlist = _most_similar_cross_label_pairs(
            mega_embeddings,
            metadata["class_id"].to_numpy(),
            top_k=aliked_pair_budget,
            image_paths=metadata["image_path"].to_numpy(),
        )
        mega_pairs = [
            _pair_payload(metadata, row.idx_a, row.idx_b, row.similarity)
            for row in mega_shortlist.head(top_k).itertuples(index=False)
        ]

        # Score the mega shortlist with LightGlue match counts
        local_pairs = []
        lightglue_shortlist = mega_shortlist
        if not lightglue_shortlist.empty:
            logger.info("Running LightGlue on %s mega shortlist pairs.", len(lightglue_shortlist))

            # Prepare dataset and matcher pairs
            aliked_dataset = FeatureDataset(aliked_embeddings, metadata, col_label="class_id")
            matcher_pairs = list(
                zip(lightglue_shortlist["idx_a"].astype(int), lightglue_shortlist["idx_b"].astype(int))
            )

            # Score the mega shortlist with LightGlue match counts
            with log_model_operation("LightGlue model load", "ALIKED matcher"):
                matcher = MatchLightGlue(features="aliked", num_workers=2)
            score_grid = matcher(aliked_dataset, aliked_dataset, pairs=matcher_pairs)

            # Convert score grid to match counts
            match_counts = np.array([score_grid[idx_a, idx_b] for idx_a, idx_b in matcher_pairs], dtype=float)
            lightglue_shortlist["match_count"] = np.nan_to_num(match_counts, nan=0.0)
            lightglue_shortlist = lightglue_shortlist.sort_values("match_count", ascending=False).head(top_k)
            local_pairs = [
                _pair_payload(metadata, row.idx_a, row.idx_b, row.match_count, mega_similarity=row.similarity)
                for row in lightglue_shortlist.itertuples(index=False)
            ]
        else:
            logger.info("Mega shortlist is empty; skipping LightGlue.")

        # Save mega and local pairs for inspection
        pair_rows = [{**item, "source": "mega"} for item in mega_pairs]
        pair_rows.extend({**item, "source": "local"} for item in local_pairs)

        output_csv = (
            Path(output_csv_file) if output_csv_file
            else Path(input_metadata_file).with_name(f"{Path(input_metadata_file).stem}_most_similar.csv")
        )
        pd.DataFrame(pair_rows).to_csv(output_csv, index=False)
        logger.info("Saved most-similar pairs file to %s", output_csv)

    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        return {"status": "ERROR", "error": error}

    return {
        "status": "DONE",
        "message": f"Found {len(mega_pairs)} mega and {len(local_pairs)} local cross-identity pairs.",
        "organization_id": organization_id,
        "input_metadata_file": input_metadata_file,
        "mega_pairs": mega_pairs,
        "local_pairs": local_pairs,
    }
