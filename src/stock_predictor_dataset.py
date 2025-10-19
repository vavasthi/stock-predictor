from torch.utils.data import Dataset
from datasettype import DatasetType
from preprocessed import PreProcessed
from utilities import findBucketIndex
import torch
import numpy as np


class StockPredictorDataset(Dataset) :
    def __init__(self, base_directory, dataset_type:DatasetType, sequences: list[PreProcessed],device) :
        self.base_directory = base_directory
        self.sequences = sequences
        self.dataset_type = dataset_type
        self.length = 0
        self.current_bucket_index = 0
        self.device = device
        self.current_dataset_bucket = self.sequences[self.current_bucket_index]
        self.current_bucket_data = np.load(self.current_dataset_bucket.file)
        for p in sequences:
            match self.dataset_type:
                case DatasetType.TRAIN:
                    self.length = max(p.train_index_end, self.length)
                case DatasetType.VALIDATE:
                    self.length = max(p.val_index_end, self.length)
                case DatasetType.TEST:
                    self.length = max(p.test_index_end, self.length)
        self.length += 1

    def __getstate__(self):
        self.current_dataset_bucket = None
        self.current_bucket_data = None
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        self.current_dataset_bucket = self.sequences[self.current_bucket_index]
        self.current_bucket_data = np.load(self.current_dataset_bucket.file)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        match self.dataset_type:
            case DatasetType.TRAIN:
                if idx < self.current_dataset_bucket.train_index_start or idx > self.current_dataset_bucket.train_index_end:
                    self.current_bucket_index = findBucketIndex(idx, DatasetType.TRAIN, self.sequences)
                    self.current_dataset_bucket = self.sequences[self.current_bucket_index]
                    self.current_bucket_data = np.load(self.current_dataset_bucket.file)
                index_in_bucket = idx - self.current_dataset_bucket.train_index_start;
                train_ip = torch.from_numpy(self.current_bucket_data['train_input'][index_in_bucket]).float().to(self.device)
                train_op = torch.from_numpy(self.current_bucket_data['train_output'][index_in_bucket]).float().to(self.device)
                return train_ip,train_op,
            case DatasetType.VALIDATE:
                if idx < self.current_dataset_bucket.val_index_start or idx > self.current_dataset_bucket.val_index_end:
                    self.current_bucket_index = findBucketIndex(idx, DatasetType.VALIDATE, self.sequences)
                    self.current_dataset_bucket = self.sequences[self.current_bucket_index]
                    self.current_bucket_data = np.load(self.current_dataset_bucket.file)
                index_in_bucket = idx - self.current_dataset_bucket.val_index_start;
                val_ip = torch.from_numpy(self.current_bucket_data['val_input'][index_in_bucket]).float().to(self.device)
                val_op = torch.from_numpy(self.current_bucket_data['val_output'][index_in_bucket]).float().to(self.device)
                return val_ip,val_op,
            case DatasetType.TEST:
                if idx < self.current_dataset_bucket.test_index_start or idx > self.current_dataset_bucket.test_index_end:
                    self.current_bucket_index = findBucketIndex(idx, DatasetType.TEST, self.sequences)
                    self.current_dataset_bucket = self.sequences[self.current_bucket_index]
                    self.current_bucket_data = np.load(self.current_dataset_bucket.file)
                index_in_bucket = idx - self.current_dataset_bucket.test_index_start;
                test_ip = torch.from_numpy(self.current_bucket_data['test_input'][index_in_bucket]).float().to(self.device)
                test_op =torch.from_numpy(self.current_bucket_data['test_output'][index_in_bucket]).float().to(self.device)
                return test_ip,test_op,