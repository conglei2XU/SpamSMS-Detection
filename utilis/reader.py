import numpy as np
import pandas as pd


def csv_reader(file_path, keys=('text', 'label', 'spans')):
    """
    generate data as described in keys
    """
    data = pd.read_csv(file_path)
    if len(keys) > 2:
        data[keys[1]] = data[keys[1]].apply(eval)
        data[keys[2]] = data[keys[2]].apply(eval)
    data_ = []
    for key in keys:
        if key in data.columns:
            data_.append(list(data[key]))
        else:
            raise KeyError(f"{key} doesn't exist in source csv")
    return data_


if __name__ == "__main__":
    path = '../dataset/THUCnews/train.csv'
    csv_reader(path)