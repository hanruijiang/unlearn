import os
import numpy as np
import pandas as pd

import torch


def load_dataset(path):
    
    df = pd.read_csv(path)
    
    x = df.drop(labels=['label'], axis=1).values
    y = df['label'].values
    
    return torch.utils.data.TensorDataset(torch.tensor(x).float().view(-1, 1, 28, 28), torch.tensor(y).long())

def load_original_dataset(data_dir):
    
    train_set = load_dataset(os.path.join(data_dir, 'Original/train_set.csv'))
    test_set = load_dataset(os.path.join(data_dir, 'Original/test_set.csv'))
    
    return train_set, test_set

def load_deleted_dataset(data_dir, delete_precent):
    
    dir_name = f'{delete_precent}perDeleted'
    
    retain_set = load_dataset(os.path.join(data_dir, dir_name, 'retain_set.csv'))
    test_set = load_dataset(os.path.join(data_dir, dir_name, 'test_set.csv'))
    forget_set = load_dataset(os.path.join(data_dir, dir_name, 'forget_set.csv'))
    
    return retain_set, test_set, forget_set