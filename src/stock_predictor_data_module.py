import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utilities import load_data
from stock_predictor_dataset import StockPredictorDataset, DatasetType
import os

class StockPredictorDataModule(pl.LightningDataModule):
    def __init__(self, base_directory, device, train_workers:int=19, val_workers:int=19, test_workers:int=1, train_batch_size: int = 512, val_batch_size:int = 64, test_batch_size:int = 32):
        super().__init__()
        self.base_directory = base_directory
        self.sequences = load_data(base_directory, 300, 0.70, 0.15, device)
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.device = device;

    def __getstate__(self):
        self.sequences = None
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        self.sequences = load_data(self.base_directory, 150, 0.70, 0.15, self.device)

    def setup(self, stage=None):
        print("Setting up data module...")
        self.train_dataset = StockPredictorDataset(self.base_directory, DatasetType.TRAIN, self.sequences, self.device)
        self.val_dataset = StockPredictorDataset(self.base_directory, DatasetType.VALIDATE, self.sequences, self.device)
        self.test_dataset = StockPredictorDataset(self.base_directory, DatasetType.TEST, self.sequences, self.device)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.train_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.test_workers, persistent_workers=True)