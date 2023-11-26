import json
import os
import argparse
import multiprocessing as mp
from multiprocessing import Value, Lock

import pickle
import logging
import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import PreTrainedTokenizer, AutoTokenizer, PretrainedConfig, AutoConfig
from torch.utils.data import DataLoader

from tools import log_wrapper
from dataset import DatasetSpam
from DataReader import csv_reader
from model.TransformerBased import PretrainedModels

Reader = {
    'csv': csv_reader
}


# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1080'
# os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:1080'


class CollateFnSeq:
    def __init__(self, tokenizer=None, is_split_into_words=False, seq_task=False, label2idx=None, idx2label=None):
        self.tokenizer = tokenizer
        self.is_split_into_words = is_split_into_words
        self.seq_task = seq_task
        self.label2idx = label2idx
        self.idx2label = idx2label
        self.batch_texts = []
        self.batch_labels = []

    def __call__(self, batch):
        texts = []
        labels = []
        padded_labels = []

        for text, label in batch:
            texts.append(text)
            labels.append(label)
        self.batch_labels = labels
        self.batch_texts = texts
        batchify_input = self.tokenizer(texts, padding='longest', is_split_into_words=self.is_split_into_words,
                                        truncation=True, return_tensors='pt')
        if self.seq_task:
            for idx, label in enumerate(labels):
                label_id = []
                pre_word = None
                for word_idx in batchify_input.word_ids(batch_index=idx):
                    if word_idx is None:
                        label_id.append(-100)
                    elif word_idx != pre_word:
                        label_id.append(self.label2idx[label[word_idx]])
                        pre_word = word_idx
                    else:
                        label_id.append(-100)
                padded_labels.append(label_id)
            batchify_input['labels'] = torch.tensor(padded_labels)
        else:
            batchify_input['labels'] = torch.tensor(labels)
        return batchify_input


def init_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        '--cache_dir',
        type=str,
        default='cache/',
        help='path to save checkpoints of model, pretrained model file etc.'
    )
    args.add_argument('--dataset_path', type=str, default='spam/', help='location of dataset for training, testing '
                                                                        'and validating')
    args.add_argument('--dataset_type', type=str, default='csv', help='file type of data')
    args.add_argument('--label_mapping', type=str, default='label_mapping.json', help='mapping label to corresponding '
                                                                                      'id by this json file')
    args.add_argument('--model_save_dir', type=str, default='final_models/', help='location for saving the best model')
    args.add_argument('--transformer_model', type=str, default='bert_case_chinese', help='the name of the transformer '
                                                                                         'model, using in the '
                                                                                         'neuralnetwork')
    args.add_argument('--lstm_size', type=int, default=300)
    args.add_argument('--transformer_size', type=int, default=768)
    args.add_argument('--epoch', type=int, default=30)
    args.add_argument('--lr', type=float, default=1e-5, help='learning rate for neural network')
    args.add_argument('--no_improve', type=int, default=5)

    args.add_argument('--device', type=str, default='cpu')
    args.add_argument('--world_size', type=int, default=1)

    args_parse = args.parse_args()
    return args_parse


def main(rank, args):
    # args = init_args()
    dist.init_process_group(backend='nccl', rank=rank, world_size=args.world_)
    data_dir = args.dataset_path
    file_type = args.dataset_type
    label_mapping = json.load(open(args.label_mapping, 'r'))
    data_reader = Reader[file_type]
    # proxies = {'https': '127.0.0.1:1080', 'http': '127.0.0.1:1080'}
    train_path = os.path.join(data_dir, 'train.' + file_type)
    val_path = os.path.join(data_dir, 'val.' + file_type)
    test_path = os.path.join(data_dir, 'test.' + file_type)
    train_set = DatasetSpam(train_path, data_reader, label_mapping=label_mapping)
    val_set = DatasetSpam(val_path, data_reader, label_mapping=label_mapping)
    test_set = DatasetSpam(test_path, data_reader, label_mapping=label_mapping)
    local_model_dir = os.path.join(args.cache_dir, args.transformer_model)
    config_ = AutoConfig.from_pretrained(local_model_dir)
    # config_ = PretrainedConfig.from_dict(json.load(open(local_config, 'r')))
    transformer_dim = config_.hidden_size
    tokenizer_ = AutoTokenizer.from_pretrained(os.path.join(args.cache_dir, args.transformer_model))
    transformer_model = PretrainedModels(local_model_dir, len(label_mapping), transformer_dim,
                                         pretrain_model_config=config_)
    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
    train_loader = DataLoader(train_set)


def trainer_transformer(
        model,
        train_loader,
        val_loader,
        args,
        num_labels=None,
        loss_fn=None,
        warmup_strategy=None,
        lr_decay_scheduler=None,
        idx2label=None
):
    best_model_path = ''
    return best_model_path


if __name__ == "__main__":
    args_ = init_args()
    if args_.world_size > 1:
        mp.spawn(main, nprocs=args_.world_size, args=(args_, ))
    else:
        main(-1, args_)
    main()
