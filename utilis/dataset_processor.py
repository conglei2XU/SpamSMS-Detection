import os

import json
import pandas as pd


def split_dataset(data, train_size=0.7, dev_size=0.1, test_size=0.2):
    """
    split dataset in a dataframe foram and saved in csv format.
    :param data:
    :param train_size:
    :param dev_size:
    :param test_size:
    :return:
    """
    data = data.sample(frac=1).reset_index(drop=True)
    num_train = int(len(data) * train_size)
    num_dev = int(len(data) * dev_size)
    train = data[:num_train]
    dev = data[num_train:num_train+num_dev]
    test = data[num_train+num_dev:]
    return train, dev, test


def _to_df(data_dir):
    # df = pd.DataFrame(columns=['content', 'label'])
    all_samples = []
    files_ = os.listdir(data_dir)
    label_mapping = {}
    for file in files_:
        label_name = file.strip()
        label_id = len(label_mapping)
        label_mapping[label_name] = label_id
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            for sent in f:
                all_samples.append([sent, file])
    data_df = pd.DataFrame(all_samples, columns=['content', 'label'])
    json.dump(label_mapping, open('../label_mapping.json', 'w', encoding='utf-8'))
    return data_df


def main():
    dataset_save = 'spam'
    dataset_dir = 'spam_data'
    all_data = _to_df(dataset_dir)
    if not os.path.exists(dataset_save):
        os.makedirs(dataset_save)
    train, dev, test = split_dataset(all_data)
    train.to_csv(os.path.join(dataset_save, 'train.csv',), index=False)
    dev.to_csv(os.path.join(dataset_save, 'val.csv'), index=False)
    test.to_csv(os.path.join(dataset_save, 'test.csv'), index=False)


if __name__ == "__main__":
    main()