import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    BigBirdModel
)


class PretrainSentModel(nn.Module):
    def __init__(self, model_path, hidden_dim, num_labels):
        super(PretrainSentModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = AutoModel.from_pretrained(model_path)
        # self.encoder = BigBirdModel.from_pretrained(model_path)
        self.linear_layer = nn.Linear(hidden_dim, num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                label=None,
                spans=None,
                output_attentions=None,
                return_dict=None
                ):
        # (loss[optional], logit, hidden_states[optional], output_attentions[optional]
        output = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              return_dict=return_dict)
        sequence_output = output[0]  # (batch_size, seq_len, hidden_dim)
        last_hidden_state = sequence_output[:, -1, :]  # (batch_size, hidden_dim)
        # batch_size, num_span = sequence_output.size(0), len(spans[0])

        pred_ = self.linear_layer(last_hidden_state)  # (batch_size, num_categories)
        return pred_

    def dynamic_quantization(self):
        quantized_model = torch.quantization.quantize_dynamic(self.encoder, {torch.nn.Linear}, dtype=torch.qint8)
        setattr(self, 'encoder', quantized_model)


class LongFormer(nn.Module):
    def __init__(self, model_path, hidden_dim, num_labels):
        super(LongFormer, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = BigBirdModel.from_pretrained(model_path)
        self.linear_layer = nn.Linear(hidden_dim, num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                label=None,
                spans=None,
                output_attentions=None,
                return_dict=None
                ):
        # (loss[optional], logit, hidden_states[optional], output_attentions[optional]
        output = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              return_dict=return_dict)
        sequence_output = output[0]
        batch_size, num_span = sequence_output.size(0), len(spans[0])
        entity_embedding = torch.rand(batch_size, num_span, self.hidden_dim)
        for idx, span_items in enumerate(spans):
            for idx_span, span_item in enumerate(span_items):
                entity_rep = sequence_output[idx, span_item[0]:span_item[1]]
                entity_embedding[idx, idx_span, :] = torch.mean(entity_rep, dim=0)
        # (batch_size, num_spans, hidden_dim) -> (batch_size, num_spans, num_categories)
        pred_ = self.linear_layer(entity_embedding)
        pred_ = pred_.view(batch_size * num_span, -1)  # (batch_size, number of categories)
        return pred_
