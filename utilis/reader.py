import numpy as np
import pandas as pd

from utilis.constants import KEYS


def csv_reader(file_path, keys=KEYS):
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


def read_vector(word_vector_source, skip_head=False, vector_dim=100) -> dict:
    """

    :param word_vector_source: path of word2vector file
    :param skip_head: (bool)
    :param vector_dim: dimension of vector
    :return: (dict), key word, value vector
    """
    word_vector = {}
    with open(word_vector_source, 'r', encoding='utf-8') as f:
        if skip_head:
            f.readline()
        line = f.readline()
        assert len(line.split()) == vector_dim + 1
        while line:
            word_vector_list = line.split()
            word, vector = word_vector_list[0], word_vector_list[1:]
            if len(vector) == vector_dim:
                vector = [float(num) for num in vector]
                word_vector[word] = vector
            line = f.readline()
        return word_vector


if __name__ == "__main__":
    path = '../dataset/THUCnews/train.csv'
    csv_reader(path)
