import sys
import torch
import json
import zipfile
import logging

import TransformerBased
from transformers import BertTokenizer
from ts.torch_handler.base_handler import BaseHandler

SPAN_PAD = 1
LABEL_PAD = -100
MAX_SENT_LENGTH = 4000
IDX_PATH = 'label_mapping.json'
MODEL = 'albert-chinese'
LOCAL_PATH = 'albert-chinese_10.pth'
EXCEPT_KEYS = ['spans']


class DocHandler(BaseHandler):
    def __init__(self):
        super(DocHandler, self).__init__()
        self.label2idx = json.load(open(IDX_PATH, 'r'))
        with zipfile.ZipFile(MODEL + '.zip', 'r') as zip:
            zip.extractall()
        self.tokenizer = BertTokenizer.from_pretrained(MODEL)
        # self.tokenizer.save_pretrained('tmp')
        self.model = torch.load(LOCAL_PATH)
        self.idx2label = {}
        for label, idx in self.label2idx.items():
            self.idx2label[idx] = label
        self.device = None
        self.is_initialize = False

    def initialize(self, context):
        properties = context.system_properties
        device_name = 'cuda:' + str(properties.get('gpu_id')) if torch.cuda.is_available() and properties.get('gpu_id') is not None else 'cpu'
        self.device = torch.device(device_name)
        self.is_initialize = True
        # self.mainfest = context.mainfest

    def inference(self, model_input):
        return self.model(**model_input)

    def preprocess(self, requests):
        logging.info(f'request object: {requests[0]}')
        input_data = requests[0].get('data')
        if input_data is None:
            input_data = requests[0].get('body')
        string_value = input_data.decode('utf-8')
        input_json = json.loads(string_value)
        texts = input_json['data']
        logging.info(fr'input text: {texts}')
        batchfy_input = self.tokenizer(
            texts,
            padding='longest',
            max_length=MAX_SENT_LENGTH,
            return_tensors='pt'
        )
        if self.device:
            to_device(batchfy_input, device=self.device)

        return batchfy_input
        # requests list of test [str,]

    def postprocess(self, output):
        prob_ = torch.nn.functional.softmax(output, dim=-1)
        pred_ = torch.argmax(prob_, dim=-1).view(-1)
        if pred_.is_cuda:
            pred_ = pred_.cpu()
        pred_ = pred_.numpy()
        pred_list = list(map(lambda x: self.idx2label.get(x, 'Missing'), pred_))
        return [pred_list]

def to_device(batch_data, device, except_keys=EXCEPT_KEYS):
    for key, value in batch_data.items():
        if key not in except_keys:
            batch_data[key] = value.to(device)


_service = DocHandler()


def handle(data, context):
    try:
        if data is None:
            return None
        if not _service.is_initialize:
            _service.initialize(context)

        input_data = _service.preprocess(data)
        pred_result = _service.inference(input_data)
        result = _service.postprocess(pred_result)
        return result
    except Exception as e:
        raise e


if __name__ == "__main__":
    handler = DocHandler()
    text = ['年后我要去欧洲读博士', '你真的好美丽']
    batch_input = handler.preprocess(text)
    output_ = torch.randint(5, 10, size=(5, 10), dtype=torch.float32)
    pred_list = handler.postprocess(output_)
