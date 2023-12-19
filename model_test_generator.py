import torch

from transformers import BartForCausalLM


def gen_model(batch_size, num_labels):
    pred = torch.arange(0, num_labels, dtype=torch.long)
    return pred
