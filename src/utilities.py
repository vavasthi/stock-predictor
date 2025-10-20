import json

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import random_split
from tqdm import tqdm
import os
import pandas as pd

from preprocessed import PreProcessed, PreProcessedEncoder
from datasettype import DatasetType

def get_csv_directory(base_directory):
    return os.path.join(base_directory, 'outputs')

def get_cache_directory(base_directory):
    return os.path.join(base_directory, 'cache')

def convert_to_sequences(memory, days_prediction, data_sequence):
    x = []
    y = []
    for i in range(len(data_sequence) - memory - max(days_prediction)):
        intermediate_y = []
        window = data_sequence[i:i + memory]
        for k in range(memory):
            prediction = []
            for j in range(len(days_prediction)):
                after_days = days_prediction[j]
                prediction.append(data_sequence[i + k + after_days - 1, [8, 9]])
            after_window = np.hstack(prediction)
            intermediate_y.append(after_window)
        x.append(window)
        y.append(intermediate_y)
    return np.asarray(x), np.asarray(y)

def find_bucket_index(idx, dataset_type:DatasetType, sequence):
    match dataset_type:
        case DatasetType.TRAIN:
            for i in range(0, len(sequence)):
                if idx >= sequence[i].train_index_start and idx <= sequence[i].train_index_end:
                    return i
            return len(sequence) - 1
        case DatasetType.VALIDATE:
            for i in range(0, len(sequence)):
                if idx >= sequence[i].val_index_start and idx <= sequence[i].val_index_end:
                    return i
            return len(sequence) - 1
        case DatasetType.TEST:
            for i in range(0, len(sequence)):
                if idx >= sequence[i].test_index_start and idx <= sequence[i].test_index_end:
                    return i
            return len(sequence) - 1

def populate_scaler(base_directory):

    jumbo_df = pd.DataFrame()
    csv_directory = get_csv_directory(base_directory)
    cache_directory = get_cache_directory(base_directory)
    for f in tqdm(os.listdir(csv_directory)):
        file = os.path.join(csv_directory, f)
        df = pd.read_csv(file, sep=',', index_col=False)
        jumbo_df = pd.concat([jumbo_df, df])
    scaler = MinMaxScaler()
    scaler.fit(jumbo_df)
    joblib.dump(scaler, os.path.join(cache_directory, "scaler.gz"))

def load_scaler(base_directory):
    cache_directory = get_cache_directory(base_directory)
    file = os.path.join(cache_directory, "scaler.gz")
    if not os.path.exists(file):
        populate_scaler(base_directory)

    return joblib.load(file)

def load_data(base_directory, memory, train_perc, val_perc, device, forecast_days = [1, 7, 15]):
    rv = []
    cache_directory = get_cache_directory(base_directory)
    cache_json_file = os.path.join(cache_directory, "preprocessed.json")
    if os.path.exists(cache_json_file):
        with open(cache_json_file, "r") as f :
            for o in json.load(f):
                v = PreProcessed(**o)
                rv.append(PreProcessed(**o))
        return rv;
    train_input_dataset = []
    train_output_dataset = []
    val_input_dataset = []
    val_output_dataset = []
    test_input_dataset = []
    test_output_dataset = []
    count = 1
    train_index_start = 0
    val_index_start = 0
    test_index_start = 0
    scaler = load_scaler(base_directory)
    csv_directory = get_csv_directory(base_directory)
    for f in tqdm(os.listdir(csv_directory)):
        file = os.path.join(csv_directory, f)
        original_df = pd.read_csv(file, sep=',', index_col=False)
        df = pd.DataFrame(scaler.transform(original_df), index=original_df.index, columns=original_df.columns)
        input,output = convert_to_sequences(memory, forecast_days, df.to_numpy())
        train_size = int(len(input) * train_perc);
        val_size = int(len(input) * val_perc);
        test_size = int(len(input) - train_size - val_size)
        train_input_dataset_single, val_input_dataset_single, test_input_dataset_single = random_split(input, [train_size, val_size, test_size])
        train_output_dataset_single, val_output_dataset_single, test_output_dataset_single = random_split(output, [train_size, val_size, test_size])
        train_input_dataset = np.vstack([train_input_dataset, train_input_dataset_single]) if (len(train_input_dataset) != 0) else train_input_dataset_single
        train_output_dataset = np.vstack([train_output_dataset, train_output_dataset_single]) if (len(train_output_dataset) != 0) else train_output_dataset_single
        val_input_dataset = np.vstack([val_input_dataset, val_input_dataset_single]) if (len(val_input_dataset) != 0) else val_input_dataset_single
        val_output_dataset = np.vstack([val_output_dataset, val_output_dataset_single]) if (len(val_output_dataset) != 0) else val_output_dataset_single
        test_input_dataset = np.vstack([test_input_dataset, test_input_dataset_single]) if (len(test_input_dataset) != 0) else test_input_dataset_single
        test_output_dataset = np.vstack([test_output_dataset, test_output_dataset_single]) if (len(test_output_dataset) != 0) else test_output_dataset_single
        train_data_size = len(train_input_dataset)
        val_data_size = len(val_input_dataset)
        test_data_size = len(test_input_dataset)
        if (train_data_size > 20000):
            outfile = os.path.join(cache_directory, 'preprocessed-{}.npz'.format(count))
            np.savez_compressed(outfile, train_input=np.asarray(train_input_dataset), train_output=np.asarray(train_output_dataset), val_input=np.asarray(val_input_dataset), val_output=np.asarray(val_output_dataset), test_input=np.asarray(test_input_dataset), test_output=np.asarray(test_output_dataset))
            rv.append(PreProcessed(train_index_start, val_index_start, test_index_start, train_index_start + train_data_size - 1, val_index_start + val_data_size - 1, test_index_start + test_data_size - 1, train_data_size, val_data_size, test_data_size, outfile))
            train_index_start = train_index_start + train_data_size
            val_index_start = val_index_start + val_data_size
            test_index_start = test_index_start + test_data_size
            count = count + 1
            train_input_dataset = []
            train_output_dataset = []
            val_input_dataset = []
            val_output_dataset = []
            test_input_dataset = []
            test_output_dataset = []
    with open(cache_json_file, "w") as f :
        json.dump(rv, f,cls=PreProcessedEncoder)
    return rv