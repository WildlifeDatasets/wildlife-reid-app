import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.log import setup_logging

setup_logging()
logger = logging.getLogger("app")


def split_ratio(data: pd.DataFrame, train_split: float, test_split: float):
    individuals_counts = dict(data["identity"].value_counts())

    train_counts = {k: int(np.ceil(v * train_split)) for k, v in individuals_counts.items()}
    test_counts = {k: int(np.ceil(v * test_split)) for k, v in individuals_counts.items()}

    split_counts = {k: 0 for k, v in train_counts.items()}
    test_observation_ids = []
    train_idx = []
    test_idx = []

    for row_idx, row in tqdm(data.iterrows(), desc="Split data"):
        identity = row.identity
        split = "train"
        if split_counts[identity] < test_counts[identity]:
            split = "test"
            if not isinstance(row.observation_id, str):
                test_observation_ids.append(int(row.observation_id))

        if split == "train" and row.observation_id in test_observation_ids:
            split = "test"
        split_counts[identity] += 1

        if split == "train":
            train_idx.append(row_idx)
        if split == "test":
            test_idx.append(row_idx)

    train_data = data.loc[train_idx].reset_index(drop=True)
    test_data = data.loc[test_idx].reset_index(drop=True)

    # remove individuals from test set if they are missing in train
    train_names = set(train_data.identity)
    test_data = test_data[test_data.identity.isin(train_names)].reset_index(drop=True)

    return train_data, test_data


def remove_data_tail(train_data, test_data, both=False, min_occurrence=10):
    # remove from test set individuals which are underrepresented in train set
    train_counts = dict(train_data.identity.value_counts())
    low_occurrence_names = [n for n, c in train_counts.items() if c < min_occurrence]
    test_data = test_data[~test_data.identity.isin(low_occurrence_names)]
    if both:
        train_data = train_data[~train_data.identity.isin(low_occurrence_names)]

    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


def split_data(
    metadata_file: pd.DataFrame,
    train_split_ratio: float = 0.8,
    remove_tail: int = 5,
    remove_both_tails: bool = True,
):

    train_split = train_split_ratio
    test_split = 1 - train_split_ratio

    # split data
    train_data, test_data = split_ratio(metadata_file, train_split, test_split)

    num = len(train_data) + len(test_data)
    logger.info(
        f"Test: {len(test_data)} -> {len(test_data) / num * 100:.2f}%\n"
        f"Train: {len(train_data)} -> {len(train_data) / num * 100:.2f}%\n"
    )

    if remove_tail:
        logger.info("Removing tail:")
        train_data, test_data = remove_data_tail(
            train_data, test_data, remove_both_tails, remove_tail
        )

        num = len(train_data) + len(test_data)
        logger.info(
            f"Test: {len(test_data)} -> {len(test_data) / num * 100:.2f}%\n"
            f"Train: {len(train_data)} -> {len(train_data) / num * 100:.2f}%\n"
        )

    return train_data, test_data
