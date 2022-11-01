import torch
from typing import Optional
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class SRDataset(Dataset):
    def __init__(self, root_dir, csv_path, transform=None):
        self.root_dir = root_dir
        self.csv_path = pd.read_csv(csv_path)
        self.transform = transform

        if not self.transform:
            self.transform = A.Compose([Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
                       ToTensorV2()],
                      additional_targets={'image': 'label'})

    def __len__(self):
        return len(self.csv_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = np.expand_dims(plt.imread(os.path.join(self.root_dir, 'high_res', self.csv_path.iloc[idx,0])), axis=2)
        label = np.expand_dims(plt.imread(os.path.join(self.root_dir, 'low_res/2x', self.csv_path.iloc[idx,1])), axis=2)
        transform = self.transform(image=image, label=label)
        image, label = transform['image'], transform['label']

        return image, label

class DataModule(pl.LightningDataModule):
    def __init__(self, root_dir,
                 train_path,
                 test_path,
                 split,
                 batch_size,
                 transform):
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.training = None
        self.validation = None
        self.testing = None


    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            train_dataset = SRDataset(self.hparams.root_dir, self.hparams.train_path, self.hparams.transform)
            valid_dataset = SRDataset()
            self.training, self.validation = random_split(train_dataset, [self.hparams.split, 1-self.hparams.split])

        if stage == 'test' or stage is None:
            self.testing = SRDataset(self.hparams.root_dir, self.hparams.test_path)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.training,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )