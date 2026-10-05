"""Dataset code for the Severstal notebook.

Lives in a .py file (not a notebook cell) so DataLoader workers can import it.
On Windows, workers are started with 'spawn': each one is a fresh Python process
that re-imports the Dataset class by module name. Classes defined in a Jupyter
cell live in __main__, which the workers can't see, so they die silently and
next(iter(loader)) waits forever.
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Runs in every worker process too, because each worker imports this module.
cv2.setNumThreads(0)


def rle_to_mask(rle, shape=(256, 1600)):
    """Converts a Run-Length Encoding string to a 2D mask array."""
    if pd.isna(rle):
        return np.zeros(shape, dtype=np.uint8)

    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Kaggle RLE is 1-indexed
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # Severstal RLE is column-major (top-to-bottom, then left-to-right)
    return img.reshape(shape, order="F")


def _read_rgb(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"cv2 could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class SeverstalDataset(Dataset):
    def __init__(self, df, data_folder, transforms=None, cache_in_ram=False):
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.transforms = transforms
        self.cache_in_ram = cache_in_ram
        self.image_cache = {}

        # NOTE (Windows): with 'spawn' this cache is pickled and copied into
        # EVERY worker, not shared. Only use it with num_workers=0.
        if self.cache_in_ram:
            from tqdm.auto import tqdm
            for image_id in tqdm(self.df["ImageId"].unique(), desc="Caching to RAM"):
                self.image_cache[image_id] = _read_rgb(os.path.join(self.data_folder, image_id))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["ImageId"]

        if self.cache_in_ram:
            image = self.image_cache[image_id]
        else:
            image = _read_rgb(os.path.join(self.data_folder, image_id))

        masks = np.zeros((256, 1600, 4), dtype=np.float32)
        for i in range(1, 5):
            rle = row[f"rle_{i}"]
            if pd.notna(rle):
                masks[:, :, i - 1] = rle_to_mask(rle, shape=(256, 1600))

        if self.transforms:
            augmented = self.transforms(image=image, mask=masks)
            image = augmented["image"]
            masks = augmented["mask"]

        return image, masks


class ClassifierDataset(Dataset):
    """Stage 1: image-level defect / no-defect. Skips mask decoding."""

    def __init__(self, df, data_folder, transforms=None):
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor([row["has_defect"]], dtype=torch.float32)
        image = _read_rgb(os.path.join(self.data_folder, row["ImageId"]))
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, label



class TestDataset(Dataset):
    """Inference: returns (image, image_id). Works with a DataFrame or a list of filenames."""
 
    def __init__(self, image_ids, data_folder, transforms=None):
        if isinstance(image_ids, pd.DataFrame):
            image_ids = image_ids["ImageId"].tolist()
        self.image_ids = list(image_ids)
        self.data_folder = data_folder
        self.transforms = transforms
 
    def __len__(self):
        return len(self.image_ids)
 
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = _read_rgb(os.path.join(self.data_folder, image_id))
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, image_id