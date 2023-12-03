from collections import Iterable

from torch.utils.data import Dataset


class SentDataset(Dataset):
    def __init__(self, dataset_path, reader, label2idx=None):
        self.all_samples, self.labels = reader(dataset_path)
        self.label2idx = {}
        if label2idx is None:
            for label in self.labels:
                if isinstance(label, Iterable):
                    for inner_label in label:
                        if inner_label not in self.label2idx:
                            self.label2idx[inner_label] = len(self.label2idx)
                else:
                    if label not in self.label2idx:
                        self.label2idx[label] = len(self.label2idx)
        else:
            self.label2idx = label2idx

    def __getitem__(self, item):
        label_cur = self.labels[item]
        if isinstance(label_cur, Iterable):
            label_idx_cur = [self.label2idx.get(i, None) for i in label_cur]
            for i in label_idx_cur:
                if i is None:
                    raise KeyError(f"found unexisted key in {label_cur}")
        else:
            label_idx_cur = self.label2idx.get(label_cur, None)
            if not label_idx_cur:
                raise KeyError(f"{label_cur} doesn't exist in label list")
        return self.all_samples[item], self.labels[item]

    def __len__(self):
        return len(self.all_samples)
