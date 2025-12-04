from typing import Dict, Optional, Tuple, Type, Union

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def light_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[A.Compose, A.Compose]:
    """Create light training and validation transforms."""
    train_tfms = A.Compose(
        [
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    val_tfms = A.Compose(
        [
            A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1]),
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_tfms, val_tfms


def heavy_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[A.Compose, A.Compose]:
    """Create heavy training and validation transforms."""
    train_tfms = A.Compose(
        [
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.7, 1.3)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussianBlur(blur_limit=(7, 7), p=0.5),
            A.HueSaturationValue(p=0.2),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, fill_value=128, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.25, rotate_limit=90, p=0.5),
            A.RandomGridShuffle(grid=(3, 3), p=0.1),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), elementwise=True, p=0.1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    val_tfms = A.Compose(
        [
            A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1]),
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_tfms, val_tfms


def vit_light_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[T.Compose, T.Compose]:
    """Create light training and validation transforms based on the RandAugment method."""
    train_tfms = T.Compose(
        [
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            T.RandAugment(num_ops=2, magnitude=10),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    val_tfms = T.Compose(
        [
            T.Resize(size=image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, val_tfms


def vit_medium_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[T.Compose, T.Compose]:
    """Create medium training and validation transforms based on the RandAugment method."""
    train_tfms = T.Compose(
        [
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            T.RandAugment(num_ops=2, magnitude=15),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    val_tfms = T.Compose(
        [
            T.Resize(size=image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, val_tfms


def vit_heavy_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[T.Compose, T.Compose]:
    """Create heavy training and validation transforms based on the RandAugment method."""
    train_tfms = T.Compose(
        [
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            T.RandAugment(num_ops=2, magnitude=20),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    val_tfms = T.Compose(
        [
            T.Resize(size=image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, val_tfms


def tv_light_transforms(
    *, image_size: tuple, mean: tuple = IMAGENET_MEAN, std: tuple = IMAGENET_STD, **kwargs
) -> Tuple[T.Compose, T.Compose]:
    """Create light training and validation transforms."""
    train_tfms = T.Compose(
        [
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply(T.ColorJitter(brightness=0.2, contrast=0.2), p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    val_tfms = T.Compose(
        [
            T.Resize(size=image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfms, val_tfms


default_tranforms = {
    "light": light_transforms,
    "heavy": heavy_transforms,
    "tv_light": tv_light_transforms,
    "vit_light": vit_light_transforms,
    "vit_medium": vit_medium_transforms,
    "vit_heavy": vit_heavy_transforms,
}


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Union[A.Compose, T.Compose], **kwargs):
        assert "image_path" in df
        assert "class_id" in df
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        image, file_path = self.get_image(idx)
        class_id = self.get_class_id(idx)
        image = self.apply_transforms(image)
        return image, class_id, file_path

    def get_image(self, idx: int) -> Tuple[Image.Image, str]:
        """Get i-th image and its file path in the dataset."""
        file_path = self.df["image_path"].iloc[idx]
        image_pil = Image.open(file_path).convert("RGB")
        # if len(image_pil.size) < 3:
        #     rgbimg = Image.new("RGB", image_pil.size)
        #     rgbimg.paste(image_pil)
        #     image_pil = rgbimg
        # image_np = np.asarray(image_pil)[:, :, :3]
        return image_pil, file_path

    def get_class_id(self, idx: int) -> int:
        """Get class id of i-th element in the dataset."""
        return self.df["class_id"].iloc[idx]

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        """Apply augmentation transformations on the image."""
        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=np.asarray(image))["image"]
            else:
                image = self.transform(image)
        return image


def get_dataloaders(
    train_data: Optional[Union[pd.DataFrame, list, dict]],
    val_data: Optional[Union[pd.DataFrame, list, dict]],
    augmentations: str,
    image_size: tuple,
    model_mean: tuple = IMAGENET_MEAN,
    model_std: tuple = IMAGENET_STD,
    batch_size: int = 32,
    num_workers: int = 8,
    *,
    transforms_fns: Dict[str, callable] = None,
    transforms_kws: dict = None,
    dataset_cls: Type[ImageDataset] = ImageDataset,
    dataset_kws: dict = None,
    dataloader_kws: dict = None,
) -> Tuple[DataLoader, DataLoader, tuple, tuple]:
    """Create training and validation augmentation transformations, datasets, and DataLoaders.

    The method is generic and allows to create augmentations and datasets of any given class.

    Parameters
    ----------
    train_data
        Training data of any type supported by Dataset defined using `dataset_cls`.
    val_data
        Validation data of any type supported by Dataset defined using `dataset_cls`.
    augmentations
        Name of augmentations to use (light, heavy, ...).
    image_size
        Image size used for resizing in augmentations.
    model_mean
        Model mean used for input normalization in augmentations.
    model_std
        Model mean used for input normalization in augmentations.
    batch_size
        Batch size used in DataLoader.
    num_workers
        Number of workers used in DataLoader.
    transforms_fns
        A dictionary with names of augmentations (light, heavy, ...) as keys
        and corresponding functions to create training and validation augmentations as values.
    transforms_kws
        Additional keyword arguments for the transformation function.
    dataset_cls
        Dataset class that implements `__len__` and `__getitem__` functions
        and inherits from `torch.utils.data.Dataset` PyTorch class.
    dataset_kws
        Additional keyword arguments for the Dataset class.
    dataloader_kws
        Additional keyword arguments for the DataLoader class.

    Returns
    -------
    trainloader
        Training PyTorch DataLoader.
    valloader
        Validation PyTorch DataLoader.
    (trainset, valset)
        Tuple with training and validation dataset instances.
    (train_tfm, val_tfm)
        Tuple with training and validation augmentation transformations.
    """
    transforms_fns = transforms_fns or default_tranforms
    assert len(transforms_fns) > 0
    transforms_kws = transforms_kws or {}
    dataset_kws = dataset_kws or {}
    dataloader_kws = dataloader_kws or {}

    # create training and validation augmentations
    if augmentations in transforms_fns:
        transforms_fn = transforms_fns[augmentations]
        train_tfm, val_tfm = transforms_fn(image_size=image_size, mean=model_mean, std=model_std, **transforms_kws)
    else:
        raise ValueError(
            f"Augmentation {augmentations} is not recognized. " f"Available options are {list(transforms_fns.keys())}."
        )

    # create training dataset and dataloader
    if train_data is not None:
        trainset = dataset_cls(train_data, transform=train_tfm, **dataset_kws)
        trainloader_kws = dataloader_kws.copy()
        if "shuffle" not in trainloader_kws:
            trainloader_kws["shuffle"] = True
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, **trainloader_kws)
    else:
        trainset = None
        trainloader = None

    # create validation dataset and dataloader
    if val_data is not None:
        valset = dataset_cls(val_data, transform=val_tfm, **dataset_kws)
        valloader_kws = dataloader_kws.copy()
        if "shuffle" not in valloader_kws:
            valloader_kws["shuffle"] = False
        valloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, **valloader_kws)
    else:
        valset = None
        valloader = None

    return trainloader, valloader, (trainset, valset), (train_tfm, val_tfm)

    # return None, valloader, (None, None), (None, None)
