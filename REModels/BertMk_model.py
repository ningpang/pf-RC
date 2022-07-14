import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import base_model
from transformers import BertModel, BertConfig

class Bert_Mark_model(base_model):
    def __init__(self, config):
        super(Bert_Mark_model, self).__init__()
        self.config = config
        self.name = 'BertMk'
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
        if self.pattern == 'standard':
            output = self.encoder(inputs, attention_mask=mask)[1]
        else:
            e11 = []
            e21 = []
            for i in range(inputs.size()[0]):
                tokens = inputs[i].cpu().numpy()
                e11.append(np.argwhere(tokens==30522)[0][0])
                e21.append(np.argwhere(tokens==30524)[0][0])
            tokens_output = self.encoder(inputs, attention_mask=mask)[0]
            output = []
            for i in range(len(e11)):
                instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
                instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
                output.append(instance_output)

            output = torch.cat(output, dim=0)
            output = output.view(output.size()[0], -1)
            output = self.drop(output)
            output = self.linear_transform(output)
            output = F.gelu(output)
            output = self.layer_normalization(output)
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