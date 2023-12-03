import torch
from sklearn.metrics import classification_report

from utilis.mixTool import to_device


def evaluate(model, loader, device, task_type=None, label2idx=None):
    if label2idx:
        target_names, labels = [], []
        for key, value in label2idx.items():
            labels.append(value)
            target_names.append(key)
    else:
        target_names, labels = None, None
    start_flag = True
    all_pred, all_target = [], []
    for batch_data in loader:
        to_device(batch_data, device)
        target = batch_data['label'].view(-1)  # (batch_size * num_span)

        pred_score = model(**batch_data)  # (batch_size, num_span, num_categories)
        pred_class = torch.argmax(pred_score, dim=-1).view(-1)  # (batch_size * num_span)
        indices = target != -100
        target_ = target[indices]
        pred = pred_class[indices]
        assert target_.size(1) == pred.size(1)
        all_pred.append(pred)
        all_target.append(target_)
    all_pred, all_target = torch.cat(all_pred, dim=0), torch.cat(all_target, dim=0)
    pred_result = classification_report(y_true=all_target,
                                        y_pred=all_pred,
                                        return_dict=True,
                                        target_names=target_names,
                                        labels=labels
                                        )

    return pred_result
