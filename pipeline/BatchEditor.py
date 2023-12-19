import torch
import pandas as pd

from utilis.constants import SPAN_PAD, LABEL_PAD, MAX_SENT_LENGTH


class CollateFn:
    def __init__(self,
                 tokenizer,
                 label2idx,
                 idx2label=None,
                 is_split=False,
                 task_type='doc'
                 ):
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        if idx2label:
            self.idx2label = idx2label
        else:
            self.idx2label = {}
            for label, idx in label2idx.items():
                self.idx2label[idx] = label
        self.is_split = is_split
        self.task_type = task_type

    def __call__(self, batch_data):
        batchfy_input, batch_data_sep = self.processing(batch_data)
        batchfy_input = self.post_process(batchfy_input, batch_data_sep)
        return batchfy_input

    def processing(self, batch_data):
        # (all_text, all_label, entity_spans[optional])
        batch_data_sep = _pre_processing(batch_data, task_type=self.task_type)
        batchfy_input = self.tokenizer(batch_data_sep[0],
                                       is_split_into_words=self.is_split,
                                       truncation=True,
                                       padding=True,
                                       return_tensors='pt',
                                       max_length=MAX_SENT_LENGTH
                                       )
        return batchfy_input, batch_data_sep

    def post_process(self, batchfy_input, batch_data_sep):
        if self.task_type == 'doc-span':

            # pad_labels, pad_spans = _padding_token(all_labels, all_spans)
            pad_labels, pad_spans = _padding_entity(batchfy_input[1], batchfy_input[2])
            # pad_spans_df = _span_to_csv(pad_spans)
            batchfy_input['label'] = torch.tensor(pad_labels, dtype=torch.long)
            batchfy_input['spans'] = pad_spans
            return batchfy_input
        else:
            batchfy_input['label'] = torch.tensor(batch_data_sep[1], dtype=torch.long)

        return batchfy_input


class CollateFnLight(CollateFn):
    def __init__(self,
                 tokenizer,
                 label2idx,
                 idx2label=None,
                 is_split=False,
                 task_type='doc-span'
                 ):
        super(CollateFnLight, self).__init__(tokenizer, label2idx, idx2label, is_split, task_type)

    def processing(self, batch_data):
        batch_data_sep = _pre_processing(batch_data, task_type=self.task_type)
        batch_token_idx, batch_token_len = self.tokenizer(batch_data_sep[0])
        batchfy_input = {'input_ids': torch.tensor(batch_token_idx, dtype=torch.long),
                         'input_lengths': batch_token_len}
        return batchfy_input, batch_data_sep

    def post_process(self, batchfy_input, batch_data_sep):
        if self.task_type == 'doc-span':
            pass
        else:
            batchfy_input['label'] = torch.tensor(batch_data_sep[1], dtype=torch.long)
        return batchfy_input


def _span_to_csv(all_spans):
    span_df = pd.DataFrame(all_spans, columns=[str(i) for i in range(len(all_spans[0]))])
    return span_df


def _pre_processing(batch_data, task_type):
    """
    processing batch data from dataloader before feeding into model depends on task type
    return:
    all_text: list[str]
    all_label: list[int] (see this task as a token classification tasks)
    entity_spans: pd.Dataframe (in this way, model can select list of indexes effciently
    """
    all_text, all_label = [], []
    if task_type == 'doc-span':
        entity_spans = []
    else:
        entity_spans = None
    for zip_sample in batch_data:
        text, label = zip_sample[0], zip_sample[1]
        all_text.append(text)
        all_label.append(label)
        if entity_spans is not None:
            entity_spans.append(zip_sample[2])
    # spans_csv = pd.DataFrame(entity_spans, columns=[for ])
    return (all_text, all_label, entity_spans) if entity_spans else (all_text, all_label)


def _padding_token(labels, spans):
    """
    padding labels into same length of spans and covert label format to token classification format
    """
    max_len = max(map(len, labels))
    pad_spans = []
    label_in_token = []
    for label_item, span_item in zip(labels, spans):
        cur_label_token = []
        for label_inner, span_inner in zip(label_item, span_item):
            cur_label_token += [label_inner] * (span_inner[1] - span_inner[0])
        label_in_token.append(cur_label_token)
        pad_num = (max_len - len(label_item))
        pad_spans.append(span_item + [SPAN_PAD] * pad_num)
    max_len_label = max(map(len, label_in_token))
    pad_label_token = []
    for label_token_item in label_in_token:
        pad_num = max_len_label - len(label_token_item)
        pad_label_token.append(label_token_item + [LABEL_PAD] * pad_num)
    return pad_label_token, pad_spans


def _padding_entity(labels, spans):
    """
    padding labels and span's entity in a classification format, using average
    entity representation to do prediction
    """
    max_len = max(map(len, labels))
    pad_labels, pad_spans = [], []
    for label_item, span_item in zip(labels, spans):
        pad_num = max_len - len(label_item)
        pad_labels.append(label_item + [LABEL_PAD] * pad_num)
        pad_spans.append(span_item + [SPAN_PAD] * pad_num)
    return pad_labels, pad_spans
