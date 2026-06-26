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
from progress import ProgressReporter
from train_model import train_identification_model
from wildlife_tools.data import FeatureDataset
from wildlife_tools.similarity.pairwise.collectors import CollectAll
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue

from utils import config
from utils.database import get_db_connection, init_db_connection
from utils.embedding_processing import EmbeddingProcessing
from utils.inference_identification import (
    calibrate_models,
    compute_partial,
    del_models,
    encode_images,
    get_keypoints,
    identify,
    identify_from_similarity,
    init_models,
    prepare_feature_types,
)
from utils.log import setup_logging
from utils.sequence_identification import extend_df_with_datetime, extend_df_with_sequence_id

os.environ.setdefault("TZ", "Europe/Prague")
if hasattr(time, "tzset"):
    time.tzset()

setup_logging()
logger = logging.getLogger("app")

logger.debug(f"{config.RABBITMQ_URL=}")
logger.debug(f"{config.REDIS_URL=}")
logger.debug(f"{config.POSTGRES_URL=}")

identification_worker = Celery("identification_worker", broker=config.RABBITMQ_URL, backend=config.REDIS_URL)
identification_worker.conf.timezone = os.environ.get("TZ", "Europe/Prague")
identification_worker.conf.enable_utc = False
init_db_connection(db_url=config.POSTGRES_URL)


@identification_worker.task(bind=True, name="train_identification")
def train_identification(
    self,
    input_metadata_file: str,
    organization_id: int,
    identification_model: dict = None,
    **kwargs,
):
    """Process and store Reference Image records in the database."""
    logger.debug(f"{identification_model=}")
    if identification_model is None:
        identification_model = {
            "name": "derived from LynxV4-MegaDescriptor-v2-T-256",
            "source_path": "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
            "path": "/models/model1/LynxV4-MegaDescriptor-v2-T-256.pth",
        }
    # outputdir = Path(identification_model["path"]).parent
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

    return {"status" "DONE"}


@identification_worker.task(bind=True, name="init_identification")
def init(
    self,
    input_metadata_file: str,
    organization_id: int,
    identification_model: dict = None,
    **kwargs,
):
    """Process and store Reference Image records in the database."""
    logger.debug(f"{identification_model=}")
    if identification_model is None:
        identification_model = {
            "name": "",
            "path": "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
        }

    try:
        progress = ProgressReporter(self, operation="init")
        progress.stage("load_metadata", "Loading initialization metadata")
        logger.info(f"Applying init task with args: {input_metadata_file=}, {organization_id=}.")
        # log celery worker id
        logger.debug(f"celery {self.request.id=}")

        # read metadata file
        metadata = pd.read_csv(input_metadata_file)
        progress.update(1, 1)
        assert "image_path" in metadata
        assert "class_id" in metadata
        assert "label" in metadata

        # remove all unused columns
        metadata = metadata[["image_path", "class_id", "label", "detection_results"]]

        # generate embeddings
        progress.stage("prepare_database", "Preparing reference image database")
        db_connection = get_db_connection()
        database_size = db_connection.reference_image.get_reference_images_count(organization_id)
        logger.debug(f"Database size: {database_size}")
        db_connection.reference_image.del_reference_images(organization_id)
        progress.update(1, 1)

        progress.stage("init_models", "Initializing identification models")
        init_models(identification_model["path"])
        progress.update(1, 1)
        encoding_batch_size = int(os.environ["ENCODING_BATCH_SIZE"])
        target_num_splits = math.ceil(len(metadata) / encoding_batch_size)
        metadata_splits = np.array_split(metadata, target_num_splits)
        logger.info(
            f"Starting embedding images: {len(metadata)}, " f"data will be processed in {len(metadata_splits)} batches"
        )
        for i, _metadata in enumerate(metadata_splits):
            logger.debug(f"[{i + 1}/{target_num_splits}] - {len(_metadata)}")
            progress.stage(
                "encode_embeddings",
                f"Encoding reference embeddings ({i + 1}/{target_num_splits})",
            )
            progress.update(i, target_num_splits)
            _features = encode_images(_metadata, identification_model_path=identification_model["path"])
            progress.update(i + 0.5, target_num_splits)
            _features = [json.dumps(e) for e in _features]
            _metadata["embedding"] = _features

            logger.info("Storing feature vectors into the database.")
            progress.update(
                i + 0.5,
                target_num_splits,
                message=f"Storing reference embeddings ({i + 1}/{target_num_splits})",
            )
            db_connection.reference_image.create_reference_images(organization_id, _metadata)
            progress.update(i + 1, target_num_splits)
        del_models()

        progress.stage("finalize", "Finalizing identification initialization")
        database_size = db_connection.reference_image.get_reference_images_count(organization_id)
        logger.debug(f"Database size: {database_size}")
        progress.update(1, 1)

        logger.info("Finished init identification processing.")
        out = {
            "status": "DONE",
            "message": f"Identification initiated with {len(metadata['image_path'])} images.",
        }
    except Exception:
        logger.debug(f"{identification_model=}")
        err = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{err}'.")
        out = {"status": "ERROR", "error": err}
    return out


@identification_worker.task(bind=True, name="iworker_simple_log")
def iworker_simple_log(self, *args, **kwargs):
    """Simple log task."""
    logger.info(f"Applying simple log task with args: {args=}, {kwargs=}.")
    return {"status": "DONE"}


@shared_task(bind=True, name="shared_simple_log")
def shared_simple_log(self, *args, **kwargs):
    """Simple log task."""
    logger.info(f"Applying simple log task with args: {args=}, {kwargs=}.")
    return {"status": "DONE"}


def load_features(db_connection, organization_id, *, start: int = -1, end: int = -1, rows: tuple = ()):
    """Loads specific or all rows from database."""
    # logger.debug("Started loading features from database")

    if rows:
        reference_images = []
        for idx in rows:
            _reference_images = db_connection.reference_image.get_reference_images(
                organization_id, start=idx, end=idx + 1
            )
            reference_images.append(_reference_images)
        reference_images = pd.concat(reference_images)
        # reference_images = db_connection.reference_image.get_reference_images(
        #   organization_id, rows=rows)
    else:
        reference_images = db_connection.reference_image.get_reference_images(
            organization_id, start=start, end=end, rows=list(rows)
        )

    features = [json.loads(e) for e in reference_images["embedding"]]
    # logger.debug(f"Loaded features {len(reference_images)}, rows: <{start}, {end})")
    return features, reference_images


def get_priority_pairs_from_parts(priority_parts: list, image_budget: int):
    """Merge priority_parts and get pairs.

    Specific for local descriptors (matcher=MatchLightGlue(features='aliked')).
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
    db_connection: object,
    organization_id: int,
    identification_model_path,
    top_k: int = 1,
    progress: ProgressReporter | None = None,
):
    """Predict identification for all samples."""
    # load features from database
    if progress:
        progress.stage("load_references", "Loading reference embeddings")
    database_features, reference_images = load_features(db_connection, organization_id)

    # generate query embeddings
    if progress:
        progress.stage("identify", "Encoding query images")
    query_features = encode_images(metadata, identification_model_path)

    # prepare metadata for database
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

    if progress:
        progress.stage("identify", "Comparing query images with references")
        progress.update(1, 3)
    identification_output = identify(
        query_features=query_features,
        database_features=database_features,
        query_metadata=query_metadata,
        database_metadata=database_metadata,
        identification_model_path=identification_model_path,
        top_k=top_k,
        cal_images=int(os.environ["CALIBRATION_IMAGES"]),
        image_budget=int(os.environ["IMAGE_BUDGET"]),
    )
    if progress:
        progress.update(1, 1)

    id2label = dict(zip(reference_images["class_id"], reference_images["label"]))

    return identification_output, id2label


def predict_batch(
    metadata: pd.DataFrame,
    db_connection: object,
    organization_id: int,
    database_size: int,
    identification_model_path: str,
    top_k: int = 1,
    progress: ProgressReporter | None = None,
):
    """Predict identification in batches."""
    database_batch_size = int(os.environ["DATABASE_BATCH_SIZE"])
    encoding_batch_size = int(os.environ["ENCODING_BATCH_SIZE"])
    cal_images = int(os.environ["CALIBRATION_IMAGES"])
    image_budget = int(os.environ["IMAGE_BUDGET"])

    # initialize and calibrate models
    if progress:
        progress.stage("load_references", "Initializing models and calibration data")
    init_models(identification_model_path)

    # TODO: get random calibration images?
    calibration_features, reference_images = load_features(db_connection, organization_id, start=0, end=cal_images)
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

    # prepare database split indexes
    database_split_idx = np.arange(np.ceil(database_size // database_batch_size + 1)) * database_batch_size
    database_split_idx = list(database_split_idx.astype(int))
    if database_split_idx[-1] != database_size:
        database_split_idx.append(database_size)

    # iterate over query splits/batches
    identification_output = {}
    for qi, _metadata in enumerate(metadata_splits):
        progress_str = f"[{qi + 1}/{target_num_splits}] - {len(_metadata)}"
        logger.debug(f"predict_batch: {progress_str}")
        if progress:
            progress.stage("identify", f"Encoding query batch {qi + 1}/{target_num_splits}")
            progress.update(qi, target_num_splits)
        query_features = encode_images(_metadata, identification_model_path, tqdm_desc=progress_str)
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
                db_connection,
                organization_id,
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

            # accumulate priority matrix for query_features
            logger.debug("Computing priority matrix")
            _priority_matrix = compute_partial(
                query_features=query_features,
                database_features=database_features,
                query_metadata=query_metadata,
                database_metadata=database_metadata,
                identification_model_path=identification_model_path,
                target="priority",
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

        split_idx = np.arange(np.ceil(len(database_idx) // database_batch_size + 1)) * database_batch_size
        split_idx = list(split_idx.astype(int))
        if split_idx[-1] != len(database_idx):
            split_idx.append(len(database_idx))

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
            database_features, reference_images = load_features(db_connection, organization_id, rows=_database_idx)

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
            post_process=os.environ.get("POST_PROCESS", None),
        )

        # calculate keypoints
        collector = CollectAll()
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

            database_features, reference_images = load_features(db_connection, organization_id, rows=didx)
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
    **kwargs,
):
    """Process and compare input samples with Reference Image records from the database."""
    logger.debug(f"{identification_model=}")

    if identification_model is None:
        identification_model = {
            "name": "",
            "path": "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
        }

    # identification_model["name"]
    # identification_model["path"]
    try:
        progress = ProgressReporter(self, operation="identify")
        progress.stage("load_metadata", "Loading identification metadata")
        logger.info(f"Applying identify task with args: {input_metadata_file_path=}, {organization_id=}.")
        logger.debug(f"celery task id={self.request.id=}")

        # read metadata file
        metadata = pd.read_csv(input_metadata_file_path)
        progress.update(1, 1)
        if len(metadata) == 0:
            logger.info("Input data is empty. Finishing the job.")
            out = {"status": "ERROR", "error": "Input data is empty."}
        else:
            assert "image_path" in metadata
            assert "mediafile_id" in metadata
            first_image_path = metadata["image_path"].iloc[0]
            assert Path(first_image_path).exists(), f"File '{first_image_path}' does not exist."
            logger.debug(f"first image = {first_image_path}, {Path(first_image_path).exists()}")

            # fetch embeddings of reference samples from the database
            progress.stage("load_references", "Checking reference image database")
            logger.info("Loading reference feature vectors from the database.")
            db_connection = get_db_connection()
            database_size = db_connection.reference_image.get_reference_images_count(organization_id)
            progress.update(1, 1)

            if database_size == 0:
                logger.info(f"Identification worker was not initialized for {organization_id=}. " "Finishing the job.")
                out = {
                    "status": "ERROR",
                    "error": "Identification worker was not initialized.",
                }
            else:
                logger.debug(f"Starting identification with: {len(metadata)} query files.")
                # estimate sequence id
                if ("sequence_number" not in metadata) and ("locality_name" in metadata):
                    progress.stage("prepare_sequences", "Estimating sequences")
                    logger.debug("Estimating sequence number and datetime.")
                    metadata["locality"] = metadata["locality_name"]
                    metadata = extend_df_with_datetime(metadata)
                    metadata = extend_df_with_sequence_id(metadata, sequence_time)
                    metadata["sequence_number"] = np.where(
                        metadata["locality"].isna(),
                        -1,
                        metadata["sequence_number"],
                    )
                    progress.update(1, 1)
                else:
                    progress.stage("prepare_sequences", "Using existing sequences")
                    progress.update(1, 1)
                query_image_path = list(metadata.image_path)
                query_masked_path = [p.replace("/images/", "/masked_images/") for p in query_image_path]

                database_batch_size = int(os.environ["DATABASE_BATCH_SIZE"])
                encoding_batch_size = int(os.environ["ENCODING_BATCH_SIZE"])

                if (database_batch_size >= database_size) and (encoding_batch_size >= len(metadata)):
                    logger.info("Starting full identification.")
                    identification_output, id2label = predict_full(
                        metadata,
                        db_connection,
                        organization_id,
                        identification_model_path=identification_model["path"],
                        top_k=top_k,
                        progress=progress,
                    )
                else:
                    logger.info("Starting batched identification.")
                    identification_output, id2label = predict_batch(
                        metadata,
                        db_connection,
                        organization_id,
                        database_size,
                        identification_model_path=identification_model["path"],
                        top_k=top_k,
                        progress=progress,
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
    """
    Perform outlier detection and return suspicious database images and their candidate identities.

    This function analyzes the provided metadata and available reference embeddings to detect which
    database images appear suspiciously labeled or potentially mislabeled. It identifies, for each such query image:
      - the image index,
      - its current (predicted) identity (`class_id`),
      - candidate alternative identities with similarity scores,
      - the file path to at least one candidate media file, allowing the API to resolve the corresponding MediaFile id.
    """
    try:
        # read metadata file
        metadata = pd.read_csv(input_metadata_file)
        assert "image_path" in metadata
        assert "class_id" in metadata, "Identity id should be in `class_id` column"
        assert "label" in metadata, "Label should be in `label` column"

        mediafile_paths = list(metadata["image_path"])
        class_ids = list(metadata["class_id"])

        logger.info("Loading reference feature vectors from the database.")
        db_connection = get_db_connection()
        _, reference_images = load_features(db_connection, organization_id)

        logger.info(
            "Starting identification outlier detection for organization_id=%s with %s explicit paths.",
            organization_id,
            0 if mediafile_paths is None else len(mediafile_paths),
        )

        # Add embeddings from reference_images to metadata by matching image_path
        metadata["image_name"] = metadata["image_path"].apply(lambda x: os.path.basename(x))
        reference_images["image_name"] = reference_images["image_path"].apply(lambda x: os.path.basename(x))
        if "embedding" not in metadata.columns:
            path_to_embedding = dict(zip(reference_images["image_name"], reference_images["embedding"]))
            metadata["embedding"] = metadata["image_name"].map(path_to_embedding)

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
