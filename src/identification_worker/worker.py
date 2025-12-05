import json
import logging
import math
import os
import traceback
from pathlib import Path

import numpy as np

print(f"numpy version: {np.__version__}")
import pandas as pd
import torch
from celery import Celery, shared_task
from train_model import train_identification_model
from wildlife_tools.data import FeatureDataset
from wildlife_tools.similarity.pairwise.collectors import CollectAll
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue

from utils import config
from utils.database import get_db_connection, init_db_connection
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

setup_logging()
logger = logging.getLogger("app")

logger.debug(f"{config.RABBITMQ_URL=}")
logger.debug(f"{config.REDIS_URL=}")
logger.debug(f"{config.POSTGRES_URL=}")

identification_worker = Celery("identification_worker", broker=config.RABBITMQ_URL, backend=config.REDIS_URL)
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
        logger.info(f"Applying init task with args: {input_metadata_file=}, {organization_id=}.")
        # log celery worker id
        logger.debug(f"celery {self.request.id=}")

        # read metadata file
        metadata = pd.read_csv(input_metadata_file)
        assert "image_path" in metadata
        assert "class_id" in metadata
        assert "label" in metadata

        # remove all unused columns
        metadata = metadata[["image_path", "class_id", "label", "detection_results"]]

        # generate embeddings
        db_connection = get_db_connection()
        database_size = db_connection.reference_image.get_reference_images_count(organization_id)
        logger.debug(f"Database size: {database_size}")
        db_connection.reference_image.del_reference_images(organization_id)

        init_models(identification_model["path"])
        encoding_batch_size = int(os.environ["ENCODING_BATCH_SIZE"])
        target_num_splits = math.ceil(len(metadata) / encoding_batch_size)
        metadata_splits = np.array_split(metadata, target_num_splits)
        logger.info(
            f"Starting embedding images: {len(metadata)}, " f"data will be processed in {len(metadata_splits)} batches"
        )
        for i, _metadata in enumerate(metadata_splits):
            logger.debug(f"[{i + 1}/{target_num_splits}] - {len(_metadata)}")
            _features = encode_images(_metadata, identification_model_path=identification_model["path"])
            _features = [json.dumps(e) for e in _features]
            _metadata["embedding"] = _features

            logger.info("Storing feature vectors into the database.")
            db_connection.reference_image.create_reference_images(organization_id, _metadata)
        del_models()

        database_size = db_connection.reference_image.get_reference_images_count(organization_id)
        logger.debug(f"Database size: {database_size}")

        logger.info("Finished processing.")
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
):
    """Predict identification for all samples."""
    # load features from database
    database_features, reference_images = load_features(db_connection, organization_id)

    # generate query embeddings
    query_features = encode_images(metadata, identification_model_path)

    # prepare metadata for database
    query_metadata = pd.DataFrame(
        {
            "path": metadata["image_path"],
            "identity": [-1] * len(metadata["image_path"]),
            "split": ["test"] * len(metadata["image_path"]),
        }
    )
    database_metadata = pd.DataFrame(
        {
            "path": reference_images["image_path"],
            "identity": reference_images["class_id"],
            "split": ["train"] * len(reference_images["class_id"]),
        }
    )

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

    id2label = dict(zip(reference_images["class_id"], reference_images["label"]))

    return identification_output, id2label


def predict_batch(
    metadata: pd.DataFrame,
    db_connection: object,
    organization_id: int,
    database_size: int,
    identification_model_path: str,
    top_k: int = 1,
):
    """Predict identification in batches."""
    database_batch_size = int(os.environ["DATABASE_BATCH_SIZE"])
    encoding_batch_size = int(os.environ["ENCODING_BATCH_SIZE"])
    cal_images = int(os.environ["CALIBRATION_IMAGES"])
    image_budget = int(os.environ["IMAGE_BUDGET"])

    # initialize and calibrate models
    init_models(identification_model_path)
    calibration_features, reference_images = load_features(db_connection, organization_id, start=0, end=cal_images)
    calibration_metadata = pd.DataFrame(
        {
            "path": reference_images["image_path"],
            "identity": reference_images["class_id"],
            "split": ["train"] * len(reference_images["class_id"]),
        }
    )
    calibrate_models(calibration_features, calibration_metadata)

    # prepare query metadata splits
    target_num_splits = math.ceil(len(metadata) / encoding_batch_size)
    metadata_splits = np.array_split(metadata, target_num_splits)

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
        query_features = encode_images(_metadata, identification_model_path, tqdm_desc=progress_str)
        # prepare query metadata
        query_metadata = pd.DataFrame(
            {
                "path": _metadata["image_path"],
                "identity": [-1] * len(_metadata["image_path"]),
                "split": ["test"] * len(_metadata["image_path"]),
            }
        )

        logger.debug("*" * 50)
        logger.debug("STARTING PRIORITY CALCULATION")
        logger.debug("*" * 50)

        # priority calculation iterate over
        full_database_metadata = []
        priority_matrix = []
        for db_idx in range(1, len(database_split_idx)):
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
        _identification_output, result_idx = identify_from_similarity(similarity, full_database_metadata, top_k)

        # calculate keypoints
        collector = CollectAll()
        keypoint_matcher = MatchLightGlue(features="aliked", collector=collector)

        keypoints = []
        for qidx, didx in result_idx.items():
            query_aliked_features, _ = prepare_feature_types([query_features[qidx]])
            keypoint_query_features = FeatureDataset(query_aliked_features, query_metadata.iloc[[qidx]])

            database_features, reference_images = load_features(db_connection, organization_id, rows=didx)
            database_metadata = pd.DataFrame(
                {
                    "path": reference_images["image_path"],
                    "identity": reference_images["class_id"],
                    "split": ["train"] * len(reference_images["class_id"]),
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
        logger.info(f"Applying init task with args: {input_metadata_file_path=}, {organization_id=}.")
        logger.debug(f"celery {self.request.id=}")

        # read metadata file
        metadata = pd.read_csv(input_metadata_file_path)
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
            logger.info("Loading reference feature vectors from the database.")
            db_connection = get_db_connection()
            database_size = db_connection.reference_image.get_reference_images_count(organization_id)

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
                    logger.debug("Estimating sequence number and datetime.")
                    metadata["locality"] = metadata["locality_name"]
                    metadata = extend_df_with_datetime(metadata)
                    metadata = extend_df_with_sequence_id(metadata, sequence_time)
                    metadata["sequence_number"] = np.where(metadata["locality"].isna(), -1, metadata["sequence_number"])
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
                    )

                pred_labels = [[id2label[x] for x in row] for row in identification_output["pred_class_ids"]]
                identification_output["mediafile_ids"] = metadata["mediafile_id"].tolist()
                identification_output["pred_labels"] = pred_labels
                identification_output["query_image_path"] = query_image_path
                identification_output["query_masked_path"] = query_masked_path

                # save output to json
                with open(output_json_file_path, "w") as f:
                    json.dump(identification_output, f)

                logger.info("Finished processing.")
                out = {"status": "DONE", "output_json_file": output_json_file_path}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out
