from typing import Union
import torch
import numpy as np
import torchvision.transforms as T
import json
from PIL import Image


def postprocess_image(image: torch.Tensor) -> np.ndarray:
    transform = T.Compose([
        T.Normalize(mean=[0., 0., 0.],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406],
                    std=[1., 1., 1.])
    ])

    image = transform(image)
    image = image.permute((1, 2, 0))

    return image.numpy()


def load_data(path: str) -> Union[dict, list]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_data(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_bbox(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(Image.open(path))
    img_gray = img[..., 0]

    y, x = np.nonzero(img_gray)
    bbox = np.array([np.min(x), np.min(y), np.max(x), np.max(y)])
    return bbox, img
