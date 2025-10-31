from copy import deepcopy

import numpy as np
import torch
from wildlife_tools.features import DeepFeatures
from wildlife_tools.inference import KnnClassifier
from wildlife_tools.similarity import CosineSimilarity

from .tools import load_data, save_data


class AccuracyCallback:
    def __init__(self, train_dataset, val_dataset, log_period: int = 10):
        self.log_period = log_period
        self.dataset_database = train_dataset
        self.dataset_query = val_dataset

    def __call__(self, trainer, epoch_data: dict[str, int], **kwargs):
        if (trainer.epoch + 1) % self.log_period != 0:
            return

        # calculate and log accuracy
        extractor = DeepFeatures(
            trainer.model,
            batch_size=trainer.batch_size,
            num_workers=trainer.num_workers,
            device=trainer.device,
        )
        query, database = extractor(self.dataset_query), extractor(self.dataset_database)

        similarity = CosineSimilarity()
        sim = similarity(query, database)  # ['cosine']
        classifier = KnnClassifier(k=1, database_labels=self.dataset_database.labels_string)
        preds = classifier(sim)
        acc = sum(preds == self.dataset_query.labels_string) / len(self.dataset_query.labels_string)
        # print(f"Epoch: {trainer.epoch}, Val Accuracy: {(acc*100):.2f}%")
        epoch_data["val_accuracy_epoch"] = acc

        # add attributes to trainer for other callbacks
        trainer.epoch_similarity = sim
        trainer.dataset_database = self.dataset_database
        trainer.dataset_query = self.dataset_query


class FileEpochLog:
    def __init__(self, status_path: str = ""):
        self.status_path = status_path

    def __call__(self, trainer, epoch_data: dict[str, int], **kwargs):
        status: dict = load_data(self.status_path)

        _epoch_data = deepcopy(epoch_data)
        for k, v in _epoch_data.items():
            if isinstance(v, (np.float32, np.float16)):
                _epoch_data[k] = float(v)

        _epoch_data["epoch"] = trainer.epoch
        if "history" in status:
            status["history"].append(_epoch_data)
        else:
            status["history"] = [_epoch_data]
        save_data(self.status_path, status)


class EpochCheckpoint:
    def __init__(self, status_path: str = "", checkpoint_path: str = ""):
        self.status_path = status_path
        self.checkpoint_path = checkpoint_path

    def __call__(self, trainer, epoch_data: dict[str, int], **kwargs):
        status: dict = load_data(self.status_path)
        status["epochs_trained"] += 1

        checkpoint = {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "epoch": trainer.epoch,
        }

        torch.save(checkpoint, self.checkpoint_path)
        status["last_checkpoint_path"] = self.checkpoint_path
        save_data(self.status_path, status)
