import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

RNNMapping = {
    'LSTM': nn.LSTM,
    'GRU': nn.GRU
}


class RNNNet(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 char_alphabet_size,
                 input_size,
                 hidden_size,
                 word_vector_size,
                 num_layers,
                 num_labels,
                 pretrained_vector=None,
                 rnn_type='LSTM',
                 use_char=False,
                 char_embedding_dim=None,
                 char_hidden_dim=None,
                 kernel_size=None):
        super(RNNNet, self).__init__()
        self.use_char = use_char
        if pretrained_vector is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(pretrained_vector, freeze=False)
        else:
            self.embedding_layer = nn.Embedding(vocabulary_size, word_vector_size)
        if use_char:
            self.char_cnn = CharCNN(char_alphabet_size, char_embedding_dim, char_hidden_dim, kernel_size)
        self.rnn = RNNMapping[rnn_type](input_size=input_size,
                                        hidden_size=hidden_size // 2,
                                        num_layers=num_layers,
                                        bidirectional=True,
                                        )

        self.classify_layer = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(p=0.1)

    def _build_features(self,
                        input_ids=None,
                        input_lengths=None,
                        extra_word_feature=None,
                        char_ids=None,
                        extra_char_feature=None,
                        word_length=None
                        ):
        """
        word_ids: (batch_size, max_sent_length)
        char_ids: (batch_size, max_sent_length, max_word_length)
        """
        masks = input_ids.gt(0)
        word_embed = self.embedding_layer(input_ids)

        if self.use_char:
            batch_size = char_ids.size(0)
            sent_len = char_ids.size(1)
            char_len = char_ids.size(-1)
            char_ids = char_ids.reshape(-1, char_len)
            char_embed = self.char_cnn(char_ids)
            char_embed = char_embed.reshape(batch_size, sent_len, -1)
            input_ = torch.cat((word_embed, char_embed, extra_word_feature, extra_char_feature), dim=-1)
        else:
            if extra_char_feature and extra_char_feature:
                input_ = torch.cat((word_embed, extra_word_feature, extra_char_feature), dim=-1)
            else:
                input_ = word_embed
        # print(input_lengths)
        input_ = pack_padded_sequence(input_, lengths=input_lengths, enforce_sorted=False, batch_first=True)
        rnn_out, _ = self.rnn(input_)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        return rnn_out, masks

    def forward(self,
                input_ids=None,
                extra_word_feature=None,
                input_lengths=None,
                char_ids=None,
                extra_char_feature=None,
                masks=None,
                label=None
                ):
        rnn_out, masks_ = self._build_features(input_ids=input_ids, input_lengths=input_lengths)

        tag_seq = self.classify_layer(rnn_out[:, 0, :])
        return tag_seq

    def loss(self,
             sentence_ids=None,
             extra_word_feature=None,
             sentence_length=None,
             char_ids=None,
             extra_char_feature=None,
             masks=None,
             label_ids=None,
             label_ids_original=None):

        rnn_out, masks_ = self._build_features(input_ids=sentence_ids,
                                               extra_word_feature=extra_word_feature,
                                               input_lengths=sentence_length,
                                               char_ids=char_ids,
                                               extra_char_feature=extra_char_feature)
        loss = self.crf.loss(rnn_out, label_ids, masks_)
        return loss


class CharCNN(nn.Module):
    """

    """

    def __init__(self, alphabet_size, embedding_dim, char_dim, kernerl_size=3):
        super(CharCNN, self).__init__()
        # kernel_w char dim || kernel_h windows size
        self.embedding_layer = nn.Embedding(alphabet_size, embedding_dim)
        self.char_drop = nn.Dropout()
        self.char_cnn = nn.Conv1d(embedding_dim, char_dim, kernerl_size, padding=1)

    def forward(self, char_input):
        # char_input (batch_size, max_word_length) -> (batch_size, max_word_length, embed_dim)
        char_input = self.embedding_layer(char_input)
        # -> (batch_size, embed_dim, max_word_length)
        char_input_embedding = char_input.transpose(2, 1).contiguous()
        char_input_embedding = self.char_drop(char_input_embedding)
        # -> (batch_size, hidden_dim, length)
        char_cnn_out = self.char_cnn(char_input_embedding)
        # -> (batch_size, hidden_dim, max_word_length)
        char_cnn_out = self.char_drop(char_cnn_out)
        char_cnn_out = f.max_pool1d(char_cnn_out, char_cnn_out.size(2))
        # -> (batch_size, max_word_length, hidden_dim)
        # char_cnn_out = f.max_pool1d(char_cnn_out, char_cnn_out.size(2))
        return char_cnn_out
