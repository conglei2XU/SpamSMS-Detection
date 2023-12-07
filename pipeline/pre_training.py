import os
import json

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from TransformerBased import PretrainSentModel, LongFormer
from pipeline.Dataset import SentDataset, DocDataset
from utilis.reader import csv_reader
from utilis.constants import *

DATACLASS = {
    'sent': SentDataset,
    'doc-span': DocDataset
}

READER = {
    'csv': csv_reader
}

MODELS = {
    'sent': PretrainSentModel,
    'doc-span': LongFormer
}


class PreTraining:
    def __init__(self,
                 train_arguments,
                 model_arguments):
        # model config
        self.hidden_dim = model_arguments.hidden_dim
        self.batch_size = train_arguments.batch_size
        self.model_path = model_arguments.model_path
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.save_to = train_arguments.save_to

        # config for setting pipeline
        self.dataset_path = train_arguments.dataset_path
        self.reader = READER[train_arguments.reader]
        self.Dataset = DATACLASS[train_arguments.task_type]
        self.method = train_arguments.method
        self.task_type = train_arguments.task_type

        # training arguments
        self.lr = train_arguments.learning_rate
        self.optimizer = optim.AdamW
        self.adam_eps = train_arguments.adam_eps
        self.warmup_step = train_arguments.warmup_step
        self.epoch = train_arguments.epoch
        self.patience = train_arguments.early_stop

        # post-processing after initialization
        self.train, self.val, self.test = self.init_dataset()
        if os.path.exists(train_arguments.label_mapping):
            self.label2idx = json.load(open(train_arguments.label_mapping, 'r', encoding='utf-8'))
            self.train.label2idx = self.label2idx
        else:
            self.label2idx = self.train.label2idx
            json.dump(self.label2idx, open('label_mapping.json', 'w', encoding='utf-8'))
        self.val.label2idx = self.label2idx
        self.test.label2idx = self.label2idx

        self.idx2label = {}
        for key, value in self.label2idx.items():
            self.idx2label[value] = key

    def create_loader(self, collate_fn=None, data_sampler=None):
        train_loader = DataLoader(self.train, batch_size=self.batch_size, collate_fn=collate_fn,
                                  sampler=data_sampler)
        val_loader = DataLoader(self.val, batch_size=self.batch_size, collate_fn=collate_fn)
        test_loader = DataLoader(self.test, batch_size=self.batch_size, collate_fn=collate_fn)
        return train_loader, val_loader, test_loader

    def init_dataset(self):
        train_dataset = self.Dataset(os.path.join(self.dataset_path, 'train.csv'), self.reader)
        val_dataset = self.Dataset(os.path.join(self.dataset_path, 'val.csv'), self.reader)
        test_dataset = self.Dataset(os.path.join(self.dataset_path, 'test.csv'), self.reader)
        return train_dataset, val_dataset, test_dataset

    def prepare_model(self):
        if self.method == 'long-attention':
            tokenizer = BertTokenizer.from_pretrained(self.model_path)
            try:
                model_class = MODELS[self.task_type]
                model = model_class(self.model_path, self.hidden_dim, len(self.label2idx))
            except KeyError:
                raise f"{self.task_type} doesn't have a corresponding model "
            return tokenizer, model

    def prepare_optimizer(self, model, data_loader):
        all_steps = self.epoch * len(data_loader)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, eps=self.adam_eps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.warmup_step,
                                                    num_training_steps=all_steps)
        loss_fn = nn.CrossEntropyLoss(ignore_index=LABEL_PAD)
        return optimizer, scheduler, loss_fn
