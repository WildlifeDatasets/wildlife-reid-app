from typing import Tuple, Union

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class PredictionDataset(Dataset):
    def __init__(self, image_paths: list, transform: Union[A.Compose, T.Compose], **kwargs):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        file_path = self.image_paths[idx]
        image = Image.open(file_path).convert("RGB")
        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=np.asarray(image))["image"]
            else:
                image = self.transform(image)
        return image, 0, file_path
