import json
import requests

from transformers import LukeModel, BertTokenizer
from tokenizers import (normalizers, )
sample = {'data': ['女孩', 'ndndd']}
result = requests.post('http://localhost:8080/predictions/albert', data=json.dumps(sample))
print(result.json())
tokenizer = BertTokenizer.from_pretrained()
tokenizer.add_special_tokens()