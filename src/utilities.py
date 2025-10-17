import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

def convert_to_sequences(memory, days_prediction, data_sequence):
    x = []
    y = []
    for i in range(len(data_sequence) - memory - max(days_prediction)):
        window = data_sequence[i:i+memory]
        prediction = []
        for j in range(len(days_prediction)):
            after_days = days_prediction[j]
            prediction.append(data_sequence[i+memory + after_days - 1,[8,9]])
        after_window = np.hstack(prediction)
        x.append(window)
        y.append(after_window)
    return torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y))

def load_data(directory, cache_directory, memory, train_perc, val_perc, device, forecast_days = [1, 7, 15]):
    rv = []
    train_input_dataset = []
    train_output_dataset = []
    val_input_dataset = []
    val_output_dataset = []
    test_input_dataset = []
    test_output_dataset = []
    count = 1
    train_data_size = 0
    val_data_size = 0
    test_data_size = 0
    train_index_start = 0
    val_index_start = 0
    test_index_start = 0
    for f in tqdm(os.listdir(directory)):
        file = os.path.join(directory, f)
        df = pd.read_csv(file, sep=',', index_col=False)
        input,output = convert_to_sequences(memory, forecast_days, newdf.to_numpy())
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
        train_data_size = train_data_size + len(train_input_dataset)
        val_data_size = val_data_size + len(val_input_dataset)
        test_data_size = test_data_size + len(test_input_dataset)
        if (train_data_size > 20000):
            print("Writing file of size ", train_data_size)
            outfile = os.path.join(cache_directory, 'preprocessed-{}.npz'.format(count))
            np.savez_compressed(outfile, train_input=np.asarray(train_input_dataset), train_output=np.asarray(train_output_dataset), val_input=np.asarray(val_input_dataset), val_output=np.asarray(val_output_dataset), test_input=np.asarray(test_input_dataset), test_output=np.asarray(test_output_dataset)) 
            rv.append(PreProcessed(train_index_start, val_index_start, test_index_start, train_index_start + train_data_size - 1, val_index_start + val_data_size - 1, test_index_start + test_data_size - 1, train_data_size, val_data_size, test_data_size, outfile))
            train_index_start = train_index_start + train_data_size
            val_index_start = val_index_start + val_data_size
            test_index_start = test_index_start + test_data_size
            train_data_size = 0
            val_data_size = 0
            test_data_size = 0
            count = count + 1
            train_input_dataset = []
            train_output_dataset = []
            val_input_dataset = []
            val_output_dataset = []
            test_input_dataset = []
            test_output_dataset = []
    return rv