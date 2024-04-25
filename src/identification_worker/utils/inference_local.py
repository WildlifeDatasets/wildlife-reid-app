from torchvision import transforms as T
from .matcher_loftr import MatchLOFTR, LoFTR
import torch
from .loftr_utils import PairSubsetDataset
from wildlife_tools.data.dataset import WildlifeDataset
import logging
import numpy as np
import cv2
import pandas as pd

logger = logging.getLogger("app")
from fgvc.utils.utils import set_cuda_device

DEVICE = set_cuda_device("cuda:0" if torch.cuda.is_available() else "cpu")


class CarnivoreDataset(WildlifeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def image_paths(self):
        """Return the image paths."""
        return self.metadata["path"].astype(str).values


class WildFusionClassifier():
    def __init__(
            self,
            matcher=None,
            extractor_deep=None,
            extractor_local=None,
            transform_deep=None,
            transform_local=None,
            sim_deep=None,
            sim_local=None,
            device='cuda',
            k_range: int = 20,
            thr_range: float = 50,
    ):
        if ((extractor_local and transform_local and matcher) or sim_local) is None:
            raise ValueError(
                """You need to provide local similarity either directly or
                tools (matcher, transform and extractor) to calculate it"""
            )
        if ((extractor_deep and transform_deep) or sim_deep) is None:
            raise ValueError(
                """You need to provide deep similarity either directly or
                tools (transform and extractor) to calculate it"""
            )
        if k_range is None:
            k_range = [1, 3, 10, 50, 100]
        if thr_range is None:
            thr_range = [10, 25, 50, 75, 100]

        self.matcher = matcher
        self.extractor_deep = extractor_deep
        self.extractor_local = extractor_local
        self.transform_deep = transform_deep
        self.transform_local = transform_local
        self.device = device
        self.thr_range = thr_range
        self.k_range = k_range
        self.sim_deep = sim_deep
        self.sim_local = sim_local

    def __call__(self, query, database, flatten=False):
        self.k_range = np.min([self.k_range, self.sim_deep.shape[1]])

        if self.sim_local is None:
            print('Calculating local similarity')
            query_local = self.get_features(query, self.extractor_local, self.transform_local)
            database_local = self.get_features(database, self.extractor_local, self.transform_local)

            cosine_scores, cosine_idx = torch.tensor(self.sim_deep).topk(k=self.k_range, dim=1)
            pairs = PairSubsetDataset(query_local, database_local, subset_matrix=cosine_idx)

            self.sim_local = self.matcher(pairs=pairs)

        # preds = {}
        # for key, sim in self.sim_local.items():
        #     preds[key] = self.get_predictions(self.sim_deep, sim, database.labels_string)
        preds = []
        for key, sim in self.sim_local.items():
            preds = self.get_predictions(self.sim_deep, sim, database.labels_string)

        return preds

    def get_features(self, dataset, extractor, transform):
        """ Extract features."""

        dataset = WildlifeDataset(
            metadata=dataset.metadata,
            root=dataset.root,
            transform=transform,
            img_load=dataset.img_load,
            col_path=dataset.col_path,
            col_label=dataset.col_label,
            load_label=dataset.load_label,
        )
        return extractor(dataset)

    def get_predictions(self, sim_deep, sim_local, labels):
        """Calculates combined predictions given set deep and local similarities."""

        if not torch.is_tensor(sim_deep):
            sim_deep = torch.tensor(sim_deep)

        if not torch.is_tensor(sim_local):
            sim_local = torch.tensor(sim_local)

        if sim_deep.shape != sim_local.shape:
            raise ValueError("Deep and local similarities have different shapes.")

        if sim_deep.shape[1] != len(labels):
            raise ValueError("Labels and similarity columns have different sizes.")

        results = []
        cosine_score, cosine_label_idx = sim_deep.topk(self.k_range)
        local_filtered = torch.gather(sim_local, 1, cosine_label_idx)

        local_score, local_idx = local_filtered.float().topk(self.k_range)
        local_label_idx = torch.gather(cosine_label_idx, 1, local_idx)
        # cosine_label_idx = cosine_label_idx[:, :3]

        for ls, li, ci in zip(local_score, local_label_idx, cosine_label_idx):
            pred_hybrid = torch.where(ls > self.thr_range, li, ci)
            results.append(pred_hybrid.numpy())

        return results


def get_local_matcher(size=512):
    extractor = lambda x: x
    matcher = MatchLOFTR(
        pretrained='outdoor',
        thresholds=(0.6,),
        batch_size=2,
        device=DEVICE,
        init_threshold=0.6,
        num_workers=2,
    )
    transform = T.Compose([
        T.Resize(size=(size, size)),
        T.Grayscale(),
        T.ToTensor(),
    ])

    return extractor, matcher, transform


def get_dataset(metadata):
    dataset = CarnivoreDataset(
        metadata=metadata,
        root="",
        img_load="full",
    )
    return dataset


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_keypoints(query, database, merged_ids, num_kp=10, size=512):
    model = LoFTR(pretrained="outdoor", apply_fine=True, thr=0.05).to(DEVICE)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(size=(size, size)),
        T.Grayscale(),
        T.ToTensor(),
    ])

    keypoints = []
    for query_idx, database_idxs in enumerate(merged_ids):
        _keypoints = []
        for database_idx in database_idxs:
            query_path = query.image_paths[query_idx]
            database_path = database.image_paths[database_idx]

            query_image = load_image(query_path)
            database_image = load_image(database_path)

            query_image = cv2.resize(query_image, (size, size))
            database_image = cv2.resize(database_image, (size, size))

            image0 = transform(query_image).unsqueeze(0)
            image1 = transform(database_image).unsqueeze(0)

            data = {
                "image0": image0.to(DEVICE),
                "image1": image1.to(DEVICE),
            }

            with torch.inference_mode():
                output = model(data)

            for k in output:
                output[k] = output[k].cpu().numpy()
            idxs = np.argsort(output["confidence"])[::-1]
            idxs = idxs[:num_kp]

            kp0 = output["keypoints0"][idxs].astype(float).tolist()
            kp1 = output["keypoints1"][idxs].astype(float).tolist()
            _keypoints.append((kp0, kp1))
        keypoints.append(_keypoints)

    return keypoints

def top_identities(predictions_ids, database_names, top_k=3):
    new_predictions_ids = []
    for query_pred in predictions_ids:
        _identities = []
        _idxs = []
        for idx in query_pred:
            name = database_names[idx]
            if name not in _identities:
                _identities.append(name)
                _idxs.append(idx)

            if len(_identities) == top_k:
                break
        new_predictions_ids.append(_idxs)
    return new_predictions_ids


def get_merged_predictions(
        query_metadata: pd.DataFrame,
        database_metadata: pd.DataFrame,
        cos_similarity: np.ndarray,
        top_k=3,
        k_range: int = 50,
        num_kp: int = 10,
        identities: bool = True
) -> (list, list):
    extractor_local, matcher, transform_local = get_local_matcher()

    classifier = WildFusionClassifier(
        matcher=matcher,
        extractor_deep=None,
        extractor_local=extractor_local,
        transform_deep=None,
        transform_local=transform_local,
        sim_deep=cos_similarity,
        k_range=k_range
    )

    query = get_dataset(query_metadata)
    database = get_dataset(database_metadata)

    merged_predictions_ids = classifier(query, database)
    if identities:
        merged_predictions_ids = top_identities(merged_predictions_ids, database.labels_string, top_k)
    else:
        merged_predictions_ids = np.array(merged_predictions_ids)[:,:top_k]
    keypoints = get_keypoints(query, database, merged_predictions_ids, num_kp)

    return merged_predictions_ids, keypoints
