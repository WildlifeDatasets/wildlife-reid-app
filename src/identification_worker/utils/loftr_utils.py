import numpy as np
import torch


class PairProductDataset(torch.utils.data.IterableDataset):
    """Create IterableDataset as a product of two mapping style datasets.

    Each item in dataset0 creates pair with each item in dataset1.

    Iteration returns 4-tuple (idx0, <dataset0 data at idx0>, idx1, <dataset1 data at idx1>)

    Example:
        dataset = PairProductDataset(['x', 'y'], ['a', 'b'])
        iterator = iter(dataset)
        next(iterator)
        >>> (0, 'x', 0, 'a')
        next(iterator)
        >>> (0, 'x', 1, 'b')
    """

    def __init__(self, dataset0, dataset1, load_label=False):
        super().__init__()
        self.dataset0 = dataset0
        self.dataset1 = dataset1
        self.load_label = load_label

    def __len__(self):
        return len(self.dataset0) * len(self.dataset1)

    def index_shape(self):
        """Describes max possible value of the idx0 and idx1 indexes."""
        return len(self.dataset0), len(self.dataset1)

    def __iter__(self):
        iterator = itertools.product(range(len(self.dataset0)), range(len(self.dataset1)))

        # Get Worker specific iterator
        worker = torch.utils.data.get_worker_info()
        if worker:
            iterator = itertools.islice(iterator, worker.id, None, worker.num_workers)

        for idx0, idx1 in iterator:
            if self.load_label:
                yield idx0, self.dataset0[idx0], idx1, self.dataset1[idx1]
            else:
                yield idx0, self.dataset0[idx0][0], idx1, self.dataset1[idx1][0]


import itertools

import torch
import torchvision.transforms as T


class PairSubsetDataset(torch.utils.data.IterableDataset):
    """Create iterable dataset as a product of two mapping styles.

    Create IterableDataset as a product of two mapping style datasets given
    Each item in dataset0 creates pair with subset of items in dataset1 with size defined
    by subset_matrix.

    Iteration returns 4-tuple (idx0, <dataset0 data at idx0>, idx1, <dataset1 data at idx1>)
    """

    def __init__(self, dataset0, dataset1, subset_matrix, load_label=False):
        super().__init__()
        if len(subset_matrix) != len(dataset0):
            raise ValueError("Topk matrix must have row for each dataset0 entry.")
        self.dataset0 = dataset0
        self.dataset1 = dataset1
        self.subset_matrix = subset_matrix
        self.load_label = load_label

    def __len__(self):
        if torch.is_tensor(self.subset_matrix):
            return self.subset_matrix.numel()
        else:
            return self.subset_matrix.size

    def index_shape(self):
        """Describes max possible value of the idx0 and idx1 indexes."""
        return len(self.dataset0), len(self.dataset1)

    def __iter__(self):
        iterator = np.ndenumerate(self.subset_matrix)

        # Get Worker specific iterator
        worker = torch.utils.data.get_worker_info()
        if worker:
            iterator = itertools.islice(iterator, worker.id, None, worker.num_workers)

        for (idx0, _), idx1 in iterator:
            if self.load_label:
                yield idx0, self.dataset0[idx0], idx1, self.dataset1[idx1]
            else:
                yield idx0, self.dataset0[idx0][0], idx1, self.dataset1[idx1][0]


class ResizeLonger:
    """Resizes an image to have the longer side length equal to a given size.

    Maintains aspect ratio.

    Args:
        size (int): Target size for the longer side of the image.
    """

    def __init__(self, size, **resize_kwargs):
        self.size = size
        self.resize_kwargs = resize_kwargs

    def __call__(self, img):
        """Resize the input image to have the longer side equal to the target size."""
        w, h = img.size
        if w < h:
            new_size = (int(self.size * w / h), self.size)
        else:
            new_size = (self.size, int(self.size * h / w))
        return T.Resize(new_size, **self.resize_kwargs)(img)
