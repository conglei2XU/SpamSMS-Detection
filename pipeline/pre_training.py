import os
import json
import pickle

import numpy as np
from collections import Counter
import torch
import wordninja
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from numpy.random import default_rng

from LightModels import RNNNet
from TransformerBased import PretrainSentModel, LongFormer
from pipeline.Dataset import SentDataset, DocDataset
from utilis.reader import csv_reader, read_vector
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
        # self.tokenizer = AlbertTokenizer.from_pretrained(self.model_path)
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
        #     json.dump(self.label2idx, open('label_mapping.json', 'w', encoding='utf-8'))
        self.val.label2idx = self.label2idx
        self.test.label2idx = self.label2idx

        self.idx2label = {}
        for key, value in self.label2idx.items():
            self.idx2label[value] = key

    def create_loader(self, collate_fn=None, data_sampler=None):
        train_loader = DataLoader(self.train, batch_size=self.batch_size, collate_fn=collate_fn,
                                  sampler=data_sampler, shuffle=True)
        val_loader = DataLoader(self.val, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)
        test_loader = DataLoader(self.test, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)
        return train_loader, val_loader, test_loader

    def init_dataset(self):
        train_dataset = self.Dataset(os.path.join(self.dataset_path, 'val.csv'), self.reader)
        val_dataset = self.Dataset(os.path.join(self.dataset_path, 'val.csv'), self.reader)
        test_dataset = self.Dataset(os.path.join(self.dataset_path, 'val.csv'), self.reader)
        return train_dataset, val_dataset, test_dataset

    def prepare_model(self):
        if 'albert' in self.model_path or 'Albert' in self.model_path:
            tokenizer = AlbertTokenizer.from_pretrained(self.model_path)
        else:
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
        # optimizer = optim.SGD(model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.warmup_step,
                                                    num_training_steps=all_steps)
        loss_fn = nn.CrossEntropyLoss(ignore_index=LABEL_PAD)
        return optimizer, scheduler, loss_fn


class PreTrainingLight(PreTraining):
    def __init__(self, train_arguments, model_arguments):
        self.pretrained_tokenizer = model_arguments.pretrained_tokenizer
        self.cached_tokenizer = train_arguments.cached_tokenizer
        self.char_hidden_dim = model_arguments.char_hidden_dim
        self.input_dim = model_arguments.input_dim
        self.num_layers = model_arguments.num_layers
        self.word_vector = model_arguments.word_vector
        self.vector_dim = model_arguments.vector_dim

        super(PreTrainingLight, self).__init__(train_arguments, model_arguments)

    def prepare_model(self, vocabulary=None):

        if self.pretrained_tokenizer:
            if 'albert' in self.model_path or 'Albert' in self.model_path:
                basic_tokenizer = AlbertTokenizer.from_pretrained(self.model_path)
            else:
                basic_tokenizer = BertTokenizer.from_pretrained(self.model_path)
        else:
            basic_tokenizer = None
        try:
            print(f'try to load cached tokenizer from {self.cached_tokenizer}')
            tokenizer = pickle.load(open(self.cached_tokenizer, 'rb'))
            # tokenizer = pickle.load(open(self.cached_tokenizer, 'rb'), word2vector=word2vector)
        except FileNotFoundError:
            print(f"could not find existed cached tokenizer")
            print(f"initializing new tokenizer")
            if os.path.exists(self.word_vector):
                print(f'loading word2vector file from {self.word_vector}')
                word2vector = read_vector(self.word_vector, vector_dim=self.vector_dim)
            else:
                print(f"{self.word_vector} doesn't exist in project, use random initialized vector instead")
                word2vector = None
            tokenizer = TokenizerLight(self.train, basic_tokenizer, word2vector=word2vector)
            base_dir = os.path.dirname(self.cached_tokenizer)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            pickle.dump(tokenizer, open(self.cached_tokenizer, 'wb'))
        pretrained_vector = build_matrix(tokenizer.token2idx, tokenizer.word2vector)
        model = RNNNet(
            vocabulary_size=len(tokenizer.token2idx),
            char_alphabet_size=None,
            char_hidden_dim=self.char_hidden_dim,
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            word_vector_size=self.input_dim,
            num_layers=self.num_layers,
            num_labels=len(self.label2idx),
            pretrained_vector=pretrained_vector
        )
        return tokenizer, model


class TokenizerLight:
    def __init__(self, dataset, tokenizer=None, vocabulary=None, sep='.', word2vector=None):
        self.sep = sep
        self.counter = Counter()
        self.word2vector = word2vector
        if tokenizer:
            self.basic_tokenizer = tokenizer.tokenize
        else:
            self.basic_tokenizer = self._default_tokenize
        self.token2idx = {'PAD': LABEL_PAD_LIGHT,
                          'UNK': UNK}
        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key
        if vocabulary:
            for vocab in vocabulary:
                if vocab not in self.token2idx:
                    self.token2idx[vocab] = len(self.token2idx)
        # else:
        #     self._build_vocab(dataset)
        if word2vector:
            self._merge_word2vector()

    def __call__(self, batch_text):
        batch_token_len = []
        if isinstance(batch_text, str):
            batch_token = self.basic_tokenizer(batch_text)
        else:
            batch_token = [self.basic_tokenizer(text_sample) for text_sample in batch_text]
            batch_token_len = []
            for token_sample in batch_token:
                if len(token_sample) > MAX_SENT_LENGTH:
                    batch_token_len.append(MAX_SENT_LENGTH)
                else:
                    batch_token_len.append(len(token_sample))

        batch_token_idx = self._post_processing(batch_token)
        return batch_token_idx, batch_token_len

    def _post_processing(self, batch_token):
        max_batch_len = max(map(len, batch_token))
        if max_batch_len > MAX_SENT_LENGTH:
            max_batch_len = MAX_SENT_LENGTH
        batch_token_idx = []
        for token_sample in batch_token:
            token_idx_sample = [self.token2idx.get(token, self.token2idx['UNK']) for token in token_sample]
            if len(token_sample) > MAX_SENT_LENGTH:
                token_sample = token_sample[:MAX_SENT_LENGTH]
                token_idx_sample = token_idx_sample[:MAX_SENT_LENGTH]
            padding_num = max_batch_len - len(token_sample)
            if padding_num > 0:
                token_idx_sample.extend([self.token2idx['PAD']] * padding_num)
            batch_token_idx.append(token_idx_sample)
        return batch_token_idx

    def _build_vocab(self, dataset):
        """
        build word counter and then construct token2idx from word counter
        """

        for data_idx in range(len(dataset)):
            text_sample = dataset[data_idx][0]
            tokens = self.basic_tokenizer(text_sample)
            self.counter.update(tokens)
            for token, freq in self.counter.items():
                if freq > THRESHOLD:
                    self.token2idx[token] = len(self.token2idx)
                    self.idx2token[len(self.idx2token)] = token

    def _default_tokenize(self, batch_text):
        if isinstance(batch_text, str):
            if self.sep is None:
                tokens = [token for token in batch_text]
            else:
                tokens = wordninja.split(batch_text)
        else:
            tokens = []
            for text_sample in batch_text:
                if self.sep is None:
                    token_ = [token for token in text_sample]
                else:
                    token_ = wordninja.split(text_sample)
                tokens.append(token_)

        return tokens

    def _merge_word2vector(self):
        for word in self.word2vector.keys():
            if word not in self.token2idx:
                self.token2idx[word] = len(self.token2idx)
                self.idx2token[len(self.idx2token)] = word


def build_matrix(token_alphabet, word_vector, word_dim=100) -> torch.Tensor:
    random_generator = default_rng()
    total_word = len(token_alphabet)
    out_vocabulary = 0
    vector_matrix = random_generator.standard_normal((total_word, word_dim))
    # vector_matrix = np.zeros((total_word, word_dim))
    for word in token_alphabet:
        if word in word_vector:
            idx = token_alphabet[word]
            vector_matrix[idx, :] = word_vector[word]
        else:
            out_vocabulary += 1
    if word_vector:
        print(f'out of vocabulary number: {out_vocabulary}/{len(token_alphabet)}')
    return torch.tensor(vector_matrix, dtype=torch.float)
