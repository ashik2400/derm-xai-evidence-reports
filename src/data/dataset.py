"""Step 2.2: PyTorch dataset that loads one image type (derm or clinic) per case.

A Dataset is a list-like object: dataset[i] gives one (image tensor, case index) pair.
A DataLoader then groups these into batches for the GPU.
"""
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src import config

# ImageNet colour statistics: the pretrained encoder expects images normalised this way
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_SIZE = 224


def eval_transform():
    """No randomness: shrink short side to 224, cut the centre square, normalise."""
    return T.Compose([
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def train_transform():
    """Random changes so the model sees a slightly different image every epoch.
    Lesions have no 'up', so flips and rotations are safe."""
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class CaseImages(Dataset):
    """Images of one modality for the given cases.

    modality: "derm" or "clinic".
    Cases without a real image (has_<modality> == False) return an all-zero image
    and missing=1, following our missing-image rule.
    """

    def __init__(self, cases, modality, transform):
        self.cases = cases.reset_index(drop=True)
        self.modality = modality
        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        row = self.cases.iloc[i]
        if not row[f"has_{self.modality}"]:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), 1   # missing
        img = Image.open(config.IMAGES_DIR / row[f"{self.modality}_path"]).convert("RGB")
        return self.transform(img), 0
