import logging

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T
from wildlife_tools.data.dataset import WildlifeDataset

from .loftr_utils import PairSubsetDataset
from .matcher_loftr import LoFTR, MatchLOFTR, remove_masked_keypoints

try:
    from ..infrastructure_utils import mem
except ImportError:
    from infrastructure_utils import mem

logger = logging.getLogger("app")

# __DEVICE__ = "cuda:0" if torch.cuda.is_available() else "cpu"
# logger.debug(f"{__DEVICE__=}")
# print(f"{__DEVICE__=}")
# DEVICE = set_cuda_device(__DEVICE__)
# DEVICE = set_cuda_device("0") if torch.cuda.is_available() else "cpu"
DEVICE = mem.get_torch_cuda_device_if_available(0)
LOFTR_MODEL = None


class CarnivoreDataset(WildlifeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def image_paths(self):
        """Return the image paths."""
        return self.metadata["path"].astype(str).values


def get_loftr_model():
    """Prepare the LOFTR model."""
    global LOFTR_MODEL
    logger.debug("Before LoFTR.")
    logger.debug(f"{mem.get_vram(DEVICE)}     {mem.get_ram()}")
    pretrained = "outdoor"
    apply_fine = False
    init_threshold = 0.8
    mem.wait_for_vram(1.0)
    LOFTR_MODEL = LoFTR(pretrained=pretrained, apply_fine=apply_fine, thr=init_threshold).to(DEVICE)
    logger.debug("After LoFTR.")
    logger.debug(f"{mem.get_vram(DEVICE)}     {mem.get_ram()}")


def del_loftr_model():
    """Release the LOFTR model."""
    global LOFTR_MODEL
    LOFTR_MODEL = None
    torch.cuda.empty_cache()


class WildFusionClassifier:
    def __init__(
        self,
        matcher=None,
        extractor_deep=None,
        extractor_local=None,
        transform_deep=None,
        transform_local=None,
        sim_deep=None,
        sim_local=None,
        device="cuda",
        k_range: int = 5,
        thr_range: float = 100,
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

    def __call__(self, query, database, flatten=False, remove_masked=False):
        """Calculate predictions for the given query and database."""
        self.k_range = np.min([self.k_range, self.sim_deep.shape[1], len(database.labels_map)])

        if self.sim_local is None:
            print("Calculating local similarity")
            query_local = self.get_features(query, self.extractor_local, self.transform_local)
            database_local = self.get_features(database, self.extractor_local, self.transform_local)

            cosine_scores, cosine_idx = self.top_k_ident(
                torch.tensor(self.sim_deep), database.labels_string, self.k_range
            )
            pairs = PairSubsetDataset(query_local, database_local, subset_matrix=cosine_idx)

            self.sim_local = self.matcher(pairs=pairs, remove_masked=remove_masked)

        preds = []
        for key, sim in self.sim_local.items():
            preds = self.get_predictions(self.sim_deep, sim, database.labels_string)

        return preds

    def top_k_ident(self, sim, labels, k):
        """Get top-k identities based on cosine similarity."""
        cosine_score = []
        cosine_label_idx = []

        for row in sim:
            _cosine_score = []
            _cosine_label_idx = []
            _cosine_label_identity = []

            scores, sort_idx = torch.sort(row)
            scores = np.array(scores)[::-1]
            sort_idx = np.array(sort_idx)[::-1]
            for label_idx, score in zip(sort_idx, scores):
                if labels[label_idx] not in _cosine_label_identity:
                    _cosine_label_idx.append(label_idx)
                    _cosine_score.append(score)
                    _cosine_label_identity.append(labels[label_idx])
                if len(_cosine_score) == k:
                    break

            cosine_score.append(_cosine_score)
            cosine_label_idx.append(_cosine_label_idx)
        cosine_score = torch.tensor(cosine_score)
        cosine_label_idx = torch.tensor(cosine_label_idx)

        return cosine_score, cosine_label_idx

    def get_features(self, dataset, extractor, transform):
        """Extract features."""
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
        cosine_score, cosine_label_idx = self.top_k_ident(sim_deep, labels, self.k_range)

        local_filtered = torch.gather(sim_local, 1, cosine_label_idx)
        local_score, local_idx = local_filtered.float().topk(self.k_range)
        local_label_idx = torch.gather(cosine_label_idx, 1, local_idx)

        for ls, li, ci in zip(local_score, local_label_idx, cosine_label_idx):
            pred_hybrid = torch.where(ls > self.thr_range, li, ci)
            results.append(pred_hybrid.numpy())

        return results


def get_local_matcher(size=512, threshold=0.8):
    """Prepare local LOFTR matcher."""
    # extractor = lambda x: x
    # use function instead of lambda x: x
    def extractor(x):
        return x

    matcher = MatchLOFTR(
        model=LOFTR_MODEL,
        thresholds=(threshold,),
        batch_size=2,
        device=DEVICE,
        num_workers=2,
    )
    matcher.tqdm_kwargs.update({"desc": "Match LOFTR"})
    transform = T.Compose(
        [
            T.Resize(size=(size, size)),
            T.Grayscale(),
            T.ToTensor(),
        ]
    )

    return extractor, matcher, transform


def get_dataset(metadata):
    """Get the Carnivore dataset."""
    dataset = CarnivoreDataset(
        metadata=metadata,
        root="",
        img_load="full",
    )
    return dataset


def load_image(path):
    """Load image from path with OpenCV."""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_keypoints(query, database, merged_ids, num_kp=10, size=512):
    """Get keypoints for each pair of images."""
    if LOFTR_MODEL is None:
        get_loftr_model()

    LOFTR_MODEL.apply_fine = True
    _thr = LOFTR_MODEL.coarse_matching.thr
    LOFTR_MODEL.coarse_matching.thr = 0.5

    # model = LoFTR(pretrained="outdoor", apply_fine=True, thr=0.5).to(DEVICE)

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(size=(size, size)),
            T.Grayscale(),
            T.ToTensor(),
        ]
    )

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
                output = LOFTR_MODEL(data)

            for k in output:
                output[k] = output[k].cpu().numpy()

            kp0 = output["keypoints0"]
            kp1 = output["keypoints1"]
            confidence = output["confidence"]

            confidence = remove_masked_keypoints(query_image, kp0, confidence, True)
            confidence = remove_masked_keypoints(database_image, kp1, confidence, True)

            idxs = np.argsort(confidence)[::-1]
            idxs = idxs[:num_kp]
            idxs = idxs[confidence[idxs] > 0]

            kp0 = kp0[idxs].astype(float).tolist()
            kp1 = kp1[idxs].astype(float).tolist()
            _keypoints.append((kp0, kp1))
        keypoints.append(_keypoints)

    LOFTR_MODEL.apply_fine = False
    LOFTR_MODEL.coarse_matching.thr = _thr

    return keypoints


def top_identities(predictions_ids, database_names, top_k=3):
    """Find top_k identities for each query."""
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
    k_range: int = 10,
    thr_range: int = 100,
    threshold: float = 0.8,
    num_kp: int = 10,
    identities: bool = True,
) -> (list, list):
    """Get merged predictions and keypoints."""
    get_loftr_model()
    extractor_local, matcher, transform_local = get_local_matcher(threshold=threshold)
    matcher.tqdm_kwargs.update({"desc": "Local matching"})

    classifier = WildFusionClassifier(
        matcher=matcher,
        extractor_deep=None,
        extractor_local=extractor_local,
        transform_deep=None,
        transform_local=transform_local,
        sim_deep=cos_similarity,
        k_range=k_range,
        thr_range=thr_range,
    )

    query = get_dataset(query_metadata)
    database = get_dataset(database_metadata)

    merged_predictions_ids = classifier(query, database, remove_masked=True)
    if identities:
        merged_predictions_ids = top_identities(
            merged_predictions_ids, database.labels_string, top_k
        )
    else:
        merged_predictions_ids = np.array(merged_predictions_ids)[:, :top_k]
    keypoints = get_keypoints(query, database, merged_predictions_ids, num_kp)

    del_loftr_model()

    return merged_predictions_ids, keypoints
