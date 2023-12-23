import pdb
from collections import Counter

import torch
import numpy as np
from sklearn.metrics import classification_report

from utilis.mixTool import to_device


def evaluate(model, loader, device, task_type=None, idx2label=None):
    all_pred, all_target = [], []
    report = {}
    correct = 0
    for batch_data in loader:
        target = batch_data['label']  # (batch_size)
        if task_type == 'doc':
            target = target.view(
                -1)  # the shape of target is (batch_size, num_span) when the task is doc classification
        # pdb.set_trace()
        pred_score = model(**batch_data)  # (batch_size, num_categories)
        pred_class = torch.argmax(pred_score, dim=-1)  # (batch_size)
        correct += torch.sum(pred_class == target).item()
        assert target.size(0) == pred_class.size(0)

        all_pred.append(pred_class)
        all_target.append(target)
    all_pred, all_target = torch.cat(all_pred, dim=0), torch.cat(all_target, dim=0)
    if all_pred.is_cuda:
        all_pred, all_target = all_pred.cpu().numpy(), all_target.cpu().numpy()
    else:
        all_pred, all_target = all_pred.numpy(), all_target.numpy()
    all_num = all_pred.shape[0]
    # pdb.set_trace()
    indies = all_pred == all_target
    correct_array = all_pred[indies]
    correct_num = np.sum(indies)
    acc = correct_num / all_num
    report['overall accuracy'] = acc
    print(f'number of correct: {correct}')
    print(f'number of correct: {correct_num}; number of all samples: {all_num} ')
    inner_report = {}
    if idx2label:
        correct_class = list(map(lambda x: idx2label[x], correct_array))
        all_class = list(map(lambda x: idx2label[x], all_target))
        correct_counter, all_counter = Counter(correct_class), Counter(all_class)
        for key, value in all_counter.items():
            if key in correct_counter:
                inner_report[f'{key} accuracy'] = correct_counter[key] / value
            else:
                print(f'all predictions of {key} class are wrong')
    report['inner_report'] = inner_report

    return report
