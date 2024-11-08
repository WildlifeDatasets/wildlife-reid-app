from wildlife_tools.data.dataset import WildlifeDataset, FeatureDataset
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
import numpy as np
from typing import Union
import torch


def get_hits(dataset0, dataset1):
    '''Return grid of label correspondences given two labeled datasets.'''

    gt0 = dataset0.labels_string
    gt1 = dataset1.labels_string
    gt_grid0 = np.tile(gt0, (len(gt1), 1)).T
    gt_grid1 = np.tile(gt1, (len(gt0), 1))
    return (gt_grid0 == gt_grid1)


class SimilarityPipelineExtended(SimilarityPipeline):
    def get_feature_dataset(self, dataset: Union[WildlifeDataset, dict]) -> Union[FeatureDataset, dict]:
        if not isinstance(dataset, WildlifeDataset):
            return dataset
        if self.transform is not None:
            dataset.transform = self.transform
        if self.extractor is not None:
            return self.extractor(dataset)
        else:
            return dataset

    def fit_calibration(self, dataset0: Union[WildlifeDataset, dict], dataset1: Union[WildlifeDataset, dict]):
        super().fit_calibration(dataset0, dataset1)

    def __call__(self, dataset0: Union[WildlifeDataset, dict], dataset1: Union[WildlifeDataset, dict], pairs=None):
        return super().__call__(dataset0, dataset1, pairs)


class WildFusionExtended(WildFusion):
    def select_dataset(self, dataset, matcher):
        _dataset = dataset
        if isinstance(dataset, dict):
            _dataset = dataset[type(matcher.extractor)]
        return _dataset

    def fit_calibration(self, dataset0: Union[WildlifeDataset, dict], dataset1: Union[WildlifeDataset, dict]):
        for matcher in self.calibrated_matchers:
            _dataset0 = self.select_dataset(dataset0, matcher)
            _dataset1 = self.select_dataset(dataset1, matcher)
            matcher.fit_calibration(_dataset0, _dataset1)

        if self.priority_matcher is not None:
            _dataset0 = self.select_dataset(dataset0, self.priority_matcher)
            _dataset1 = self.select_dataset(dataset1, self.priority_matcher)
            self.priority_matcher.fit_calibration(_dataset0, _dataset1)

    def get_partial_priority(self, dataset0, dataset1):
        _dataset0 = self.select_dataset(dataset0, self.priority_matcher)
        _dataset1 = self.select_dataset(dataset1, self.priority_matcher)
        priority = self.priority_matcher(_dataset0, _dataset1)

        return priority

    def get_partial_scores(self, dataset0, dataset1, pairs):
        scores = []
        for matcher in self.calibrated_matchers:
            _dataset0 = self.select_dataset(dataset0, matcher)
            _dataset1 = self.select_dataset(dataset1, matcher)
            scores.append(matcher(_dataset0, _dataset1, pairs=pairs))
        return scores

    def get_priority_pairs(self, dataset0: WildlifeDataset, dataset1: WildlifeDataset, B):
        ''' Shortlisting strategy for selection of most relevant pairs.'''

        if self.priority_matcher is None:
            raise ValueError('Priority matcher is not assigned.')

        priority = self.priority_matcher(dataset0, dataset1)
        _, idx1 = torch.topk(torch.tensor(priority), min(B, priority.shape[1]))
        idx0 = np.indices(idx1.numpy().shape)[0]
        grid_indices = np.stack([idx0.flatten(), idx1.flatten()]).T
        return grid_indices

    def __call__(self, dataset0, dataset1, pairs=None, B=None):
        if B is not None:
            _dataset0 = self.select_dataset(dataset0, self.priority_matcher)
            _dataset1 = self.select_dataset(dataset1, self.priority_matcher)
            pairs = self.get_priority_pairs(_dataset0, _dataset1, B=B)

        scores = []
        for matcher in self.calibrated_matchers:
            _dataset0 = self.select_dataset(dataset0, matcher)
            _dataset1 = self.select_dataset(dataset1, matcher)
            scores.append(matcher(_dataset0, _dataset1, pairs=pairs))

        score_combined = np.mean(scores, axis=0)
        score_combined = np.where(np.isnan(score_combined), -np.inf, score_combined)
        return score_combined
