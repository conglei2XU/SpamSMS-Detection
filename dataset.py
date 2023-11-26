from abc import ABC

from torch.utils.data import Dataset


class DatasetSpam(Dataset, ABC):
    def __init__(self, path, data_reader, label_mapping=None, key_pairs=['content', 'label']):
        self.all_data = data_reader(path)
        self.label_mapping = label_mapping
        self.key_pairs = key_pairs

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        this_pair = self.all_data.loc[item]
        if self.key_pairs:
            sent, label = this_pair[self.key_pairs[0]], this_pair[self.key_pairs[1]]
        else:
            sent, label = this_pair[0], this_pair[1]
        if self.label_mapping:
            try:
                label = self.label_mapping[label]
            except KeyError:
                raise KeyError(f"{label} doesn't exists in label_mapping" )
        return sent, label

