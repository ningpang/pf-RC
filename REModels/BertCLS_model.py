import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import base_model
from transformers import BertModel, BertConfig

class Bert_CLS_model(base_model):
    def __init__(self, config):
        super(Bert_CLS_model, self).__init__()
        self.config = config
        self.name = 'BertCLS'
        self.encoder = Bert_Encoder(config=config)
        self.classifier = Softmax_Layer(input_size=self.encoder.output_size, num_class=config.num_of_relation)

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

        self.drop = nn.Dropout(config.drop_out)

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

    def get_output_size(self):
        return self.output_size

    def forward(self, inputs, mask):
        output = self.encoder(inputs, attention_mask=mask)[1]
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