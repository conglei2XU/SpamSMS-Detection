import os
import time
import logging
import random
from collections import Counter

import pickle
import torch
import tqdm
import numpy as np
import multiprocessing as mp
from multiprocessing import Value, Lock
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
from transformers import HfArgumentParser

from utilis.mixTool import log_wrapper, to_device
from pipeline.BatchEditor import CollateFn, CollateFnLight
from pipeline.pre_training import PreTraining, PreTrainingLight
from pipeline.arguments import TrainArguments, LongFormerArguments
from pipeline.evaluate import evaluate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '9091'

best_f1 = Value('d', -1.0)
best_f1_lock = Lock()
stop_flag = Value('i', 0)


def init_args():
    parser = HfArgumentParser((TrainArguments, LongFormerArguments))
    train_arguments, model_argument = parser.parse_args_into_dataclasses()
    return train_arguments, model_argument


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False


def trainer(
        model,
        loaders,
        optimizer,
        scheduler,
        loss_fn,
        train_config,
        device,
        rank
):
    train_loader, val_loader, test_loader = loaders
    all_steps = len(train_loader) * train_config.epoch
    train_loss_trend = []
    global_step = 1
    eval_f1_trend = []
    patience_counter = 0
    previous_model, best_model_path = None, None
    for epoch_idx in range(train_config.epoch):
        p_bar = tqdm.tqdm(train_loader)
        epoch_step = len(train_loader)
        local_step = 0
        all_num, correct_num = 0, 0
        model.train()
        for batch_data in p_bar:
            to_device(batch_data, device)
            # convert label ( batch_size, num_span) to
            labels = batch_data['label']
            logit = model(**batch_data)
            pred_ = torch.argmax(logit, dim=-1)
            if train_config.task_type == 'doc-span':
                labels = labels.view(-1)  # change labels from (batch_size, num_span) t0 (bach_size)
            correct_num += torch.sum(labels == pred_).item()
            all_num += pred_.size(0)
            loss = loss_fn(logit, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if global_step % 10 == 0 and (rank == -1 or rank == 0):
                train_loss_trend.append(loss.item())
            global_step += 1
            local_step += 1
            p_bar.set_description(f'Epoch: {epoch_idx + 1},'
                                  f' Percentage: {local_step}/{epoch_step},'
                                  f' Loss: {round(loss.item(), 2)},'
                                  f' Accuracy: {round(correct_num / all_num, 2)}')
        print(f'number of correct: {correct_num}; number of samples: {all_num}')
        model.eval()
        pred_report = evaluate(model=model,
                               loader=val_loader,
                               device=device,
                               idx2label=train_config.idx2label
                               )
        over_acc = pred_report['overall accuracy']
        if rank == -1 or rank == 0:
            logger.info('-' * 50)
            logger.info(f'Epoch: {epoch_idx + 1}/{train_config.epoch} \t'
                        f' Global step: {global_step}/{all_steps}')
            logger.info(f"Overall accuracy: {round(over_acc, 2)}")
            logger.info(f"accuracy for each class: ")
            class_report = pred_report['inner_report']
            for label_name, value in class_report.items():
                logger.info(f"class: {label_name} accuracy: {round(value, 2)}")
            with best_f1_lock:
                if over_acc > best_f1.value:
                    logger.info(
                        f'Epoch {epoch_idx + 1} has better accuracy {over_acc} than previous best {best_f1.value}')
                    model_name = train_config.model_path.split('/')[-1]
                    model_save_name = f'{model_name}_{epoch_idx}.pth'
                    best_model_path = os.path.join(train_config.save_to, model_save_name)
                    if not os.path.exists(train_config.save_to):
                        os.makedirs(train_config.save_to)
                    torch.save(model, best_model_path)
                    if previous_model:
                        os.remove(previous_model)
                        previous_model = best_model_path
                    best_f1.value = over_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    model_name_ = train_config.model_path.split('/')[-1]
                    if patience_counter > train_config.patience:
                        train_loss_file = os.path.join(train_config.save_to, f'{model_name_}.bin')
                        pickle.dump(train_loss_trend, open(train_loss_file, 'wb'))
                        stop_flag.value = 1
        if stop_flag.value == 1:
            break
    return best_model_path


def train(rank, train_argument, model_argument, logger_):
    fix_seed(train_argument.seed)
    if train_argument.is_light:
        prepare = PreTrainingLight(train_argument, model_argument)
    else:
        prepare = PreTraining(train_argument, model_argument)

    if rank != -1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=train_argument.nproc)
        device = torch.device('cuda')
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset=prepare.train,
                                                                       num_replicas=train_argument.nproc,
                                                                       rank=rank)
    else:
        device = torch.device(train_argument.device)
        data_sampler = None
    tokenizer, model = prepare.prepare_model()
    if train_argument.is_light:
        collate_fn = CollateFnLight(tokenizer, prepare.label2idx, task_type=prepare.task_type)
    else:
        collate_fn = CollateFn(tokenizer, prepare.label2idx, task_type=prepare.task_type)
    train_loader, val_loader, test_loader = prepare.create_loader(collate_fn=collate_fn,
                                                                  data_sampler=data_sampler)
    model.to(device=device)
    optimizer, scheduler, loss_fn = prepare.prepare_optimizer(model, train_loader)

    if train_argument.given_best_model:
        best_model_path = train_argument.best_model_path
    else:
        best_model_path = trainer(model=model,
                                  loaders=(train_loader, val_loader, test_loader),
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  loss_fn=loss_fn,
                                  train_config=prepare,
                                  device=device,
                                  rank=rank
                                  )

    if rank == 0 or rank == -1:
        time_start = time.time()
        logger.info(f'loading model from best model path: {best_model_path}')
        best_model = torch.load(best_model_path)
        # best_model = torch.quantization.quantize_dynamic(best_model, {torch.nn.Linear}, dtype=torch.qint8)
        best_model.eval()
        test_report = evaluate(
            model=best_model,
            loader=test_loader,
            idx2label=prepare.idx2label,
            device=device,
            task_type=train_argument.task_type
        )
        time_eps = time.time() - time_start
        logger_.info(f'Inference time: {time_eps}s')
        over_acc = test_report['overall accuracy']
        logger.info(f"Overall accuracy: {round(over_acc, 4)}")
        logger.info(f"accuracy for each class: ")
        class_report = test_report['inner_report']
        for label_name, value in class_report.items():
            logger.info(f"class: {label_name} accuracy: {round(value, 4)}")


def main():
    train_argument, model_argument = init_args()
    log_wrapper(logger, base_dir=train_argument.log_dir)
    if torch.cuda.is_available() and train_argument.nproc > 1:
        mp.spawn(train,
                 args=(train_argument, model_argument, logger,),
                 nprocs=train_argument.nproc)
    else:
        train(-1, train_argument, model_argument, logger)


if __name__ == "__main__":
    main()
