from wildlife_tools.data import WildlifeDataset


class CarnivoreDataset(WildlifeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def image_paths(self):
        return self.metadata["path"].astype(str).values
