import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import base_model
from transformers import BertModel, BertConfig

class Bert_EntCLS_model(base_model):
    def __init__(self, config):
        super(Bert_EntCLS_model, self).__init__()
        self.encoder = Bert_Encoder(config=config)
        # self.classifier = Softmax_Layer(input_size=self.encoder.output_size, num_class=config.num_of_relation)
        self.ent_classifier = Softmax_Layer(input_size=self.encoder.output_size, num_class=config.num_of_entity)
        self.rel_classifier = Softmax_Layer(input_size=self.encoder.output_size, num_class=config.num_of_relation)

    def forward(self, inputs, mask):
        reps, head, tail = self.encoder(inputs, mask)
        rel_logits = self.rel_classifier(reps)
        head_logits = self.ent_classifier(head)
        tail_logits = self.ent_classifier(tail)
        return rel_logits, head_logits, tail_logits


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
        self.entity_map = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)
        self.layer_normalization = nn.LayerNorm([self.output_size])

    def get_output_size(self):
        return self.output_size

    def forward(self, inputs, mask):
        bert_output = self.encoder(inputs, attention_mask=mask)
        tokens_output, cls_tokens = bert_output[0], bert_output[1]

        e11 = []
        e12 = []
        e21 = []
        e22 = []
        for i in range(inputs.size()[0]):
            tokens = inputs[i].cpu().numpy()
            e11.append(np.argwhere(tokens == 30522)[0][0])
            e12.append(np.argwhere(tokens == 30523)[0][0])
            e21.append(np.argwhere(tokens == 30524)[0][0])
            e22.append(np.argwhere(tokens == 30525)[0][0])
        head = []
        tail = []
        for i in range(len(e11)):
            instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
            # print(e11[i], e12[i], [index for index in range(e11[i]+1, e12[i])])
            instance_head = torch.index_select(instance_output, 1, torch.tensor([index for index in range(e11[i]+1, e12[i])]).cuda()).mean(1)
            instance_tail = torch.index_select(instance_output, 1, torch.tensor([index for index in range(e21[i]+1, e22[i])]).cuda()).mean(1)
            head.append(instance_head)
            tail.append(instance_tail)

        head = torch.cat(head, dim=0)
        tail = torch.cat(tail, dim=0)
        head = head.view(head.size()[0], -1)
        tail = tail.view(tail.size()[0], -1)
        head = self.drop(head)
        tail = self.drop(tail)
        head = self.entity_map(head)
        tail = self.entity_map(tail)
        head = F.gelu(head)
        tail = F.gelu(tail)
        head = self.layer_normalization(head)
        tail = self.layer_normalization(tail)

        return cls_tokens, head, tail

class Softmax_Layer(base_model):
    def __init__(self, input_size, num_class):
        super(Softmax_Layer, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=True)

    def forward(self, input):
        logits = self.fc(input)
        return logits