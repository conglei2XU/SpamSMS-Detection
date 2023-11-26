import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification
)


class PretrainedModels(nn.Module):
    def __init__(self,
                 local_model,
                 num_labels,
                 pretrain_size,
                 lstm_size=300,
                 batch_first=True,
                 pretrain_model_config=None):
        super(PretrainedModels, self).__init__()
        self.transformer_base = AutoModel.from_pretrained(local_model, config=pretrain_model_config)
        self.lstm = nn.LSTM(pretrain_size, lstm_size, batch_first=batch_first)
        self.full_connected = nn.Linear(lstm_size, num_labels)

    def forward(self,
                input_ids=None,
                attention_masks=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attention=None,
                output_hidden_states=None,
                return_dict=None
                ):
        # outputs[0] last_hidden_state, outputs[1] pooler_output.
        outputs = self.transformer_base(input_ids=input_ids, attention_masks=attention_masks, token_type_ids=token_type_ids)
        last_hidden_state = outputs[0]
        lstm_output, _ = self.lstm(last_hidden_state)
        # (batch_size, sent_len, lstm_size)
        sentence_rep = lstm_output[:, -1, :]
        res = self.full_connected(sentence_rep)
        return res

