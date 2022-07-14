import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import base_model
from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .utils import sort_batch_by_length, init_lstm, init_linear

class Bert_LSTM_model(base_model):
    def __init__(self, config):
        super(Bert_LSTM_model, self).__init__()
        self.config = config
        self.name = 'BertLSTM'
        self.encoder = Bert_Encoder(config=config)
        self.classifier = Softmax_Layer(input_size=config.lstm_hidden_size, num_class=config.num_of_relation)

    def forward(self, inputs, mask):
        return self.classifier(self.encoder(inputs, mask))

class Bert_Encoder(base_model):
    def __init__(self, config):
        super(Bert_Encoder, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path)
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # the dimension of output
        self.output_size = config.encoder_output_size

        self.drop = nn.Dropout(0.1)

        # which encoding is used
        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding method!')
        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size+config.marker_size)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size*2, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)
        self.layer_normalization = nn.LayerNorm([self.output_size])
        # self.convs = nn.ModuleList([nn.Conv2d(1, config.cnn_hidden_size, (k, self.encoder.output_size)) for k in [2, 3, 4]])
        self.lstm_base = nn.LSTM(input_size=self.output_size, hidden_size=config.lstm_hidden_size,
                                 bidirectional=True, batch_first=True, dropout=0.1)
        self.project = nn.Linear(config.lstm_hidden_size * 4, config.lstm_hidden_size)
        # self.apply(self.weight_init)
    #
    # def weight_init(self, m):
    #     classname = m.__class__.__name__
    #     if classname.find('Linear') != -1:
    #         init_linear(m)
    #     elif classname.find('LSTM') != -1:
    #         init_lstm(m)
    #
    def max_mean_agg(self, inputs, mask):
        max_input, _ = torch.max(inputs, 1)
        mean_input = torch.sum(inputs, 1) / torch.sum(mask, 1).unsqueeze(-1)

        output = torch.cat([max_input, mean_input], 1)
        return self.project(output)

    def get_output_size(self):
        return self.output_size

    def forward(self, inputs, mask):
        output = self.encoder(inputs)[0]
        output, _ = self.lstm_base(output)
        # output = self.project(self.max_mean_agg(output, mask))

        output = self.max_mean_agg(output, mask)
        # output = self.project(self.drop(output)[:, -1, :])
        # lengths = mask.long().sum(1)
        # sorted_inputs, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(output,
        #                                                                                       lengths)
        # packed_sequence_input = pack_padded_sequence(sorted_inputs,
        #                                              sorted_sequence_lengths.cpu(),
        #                                              batch_first=True)
        # output, _ = self.lstm_base(packed_sequence_input)
        # unpacked_sequence_tensor, _ = pad_packed_sequence(output, batch_first=True)
        # unpacked_sequence_tensor = unpacked_sequence_tensor.index_select(0, restoration_indices)
        # output = self.max_mean_agg(unpacked_sequence_tensor, mask)
        return output

class Softmax_Layer(base_model):
    def __init__(self, input_size, num_class):
        super(Softmax_Layer, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=True)

    def forward(self, input):
        logits = self.fc(input)
        return logits