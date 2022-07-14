import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import base_model
class Embedding(nn.Module):
    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim

        word_vec_mat = torch.from_numpy(word_vec_mat)
        unk = torch.zeros(1, word_embedding_dim)
        pad = torch.zeros(1, word_embedding_dim)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0]+2, self.word_embedding_dim,
                                           padding_idx=word_vec_mat.shape[0] +1)
        self.word_embedding.weight.data.copy_(torch.cat((word_vec_mat, unk, pad), 0))
        # self.word_embedding.weight.data.copy_(word_vec_mat)

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)

    def forward(self, word, pos1, pos2):

        x = torch.cat([self.word_embedding(word),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)], 2)
        return x


class Encoder(base_model):
    def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
        super(Encoder, self).__init__()

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        # self.conv = nn.ModuleList(nn.Conv2d(1, self.hidden_size, (k, 60)) for k in [2, 3, 4])
        self.pool = nn.MaxPool1d(max_length)

        # For PCNN
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100

    def forward(self, inputs):
        return self.cnn(inputs)

    def cnn(self, inputs):
        x = self.conv(inputs.transpose(1, 2))
        x = F.relu(x)
        x = self.pool(x)
        return x.squeeze(2) # n x hidden_size

    # def pcnn(self, inputs, mask):
    #     x = self.conv(inputs.transpose(1, 2)) # n x hidden x length
    #     mask = 1 - self.mask_embedding(mask).transpose(1, 2) # n x 3 x length
    #     pool1 = self.pool(F.relu(x + self._minus * mask[:, 0:1, :]))
    #     pool2 = self.pool(F.relu(x + self._minus * mask[:, 1:2, :]))
    #     pool3 = self.pool(F.relu(x + self._minus * mask[:, 2:3, :]))
    #     x = torch.cat([pool1, pool2, pool3], 1)
    #     x = x.squeeze(2) # n x (hidden_size * 3)
class Softmax_Layer(nn.Module):
    def __init__(self, input_size, num_class):
        super(Softmax_Layer, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=True)

    def forward(self, input):
        logits = self.fc(input)
        return logits

class CNN_model(base_model):
    def __init__(self, config, word_vec_mat, word2id):
        nn.Module.__init__(self)
        self.hidden_size = config.cnn_hidden_size
        self.max_length = config.max_length
        self.embedding = Embedding(word_vec_mat, config.max_length,
                config.word_embedding_dim, config.pos_embedding_dim)
        self.encoder = Encoder(config.max_length, config.word_embedding_dim,
                config.pos_embedding_dim, config.cnn_hidden_size)
        self.classifier = Softmax_Layer(input_size=config.cnn_hidden_size, num_class=config.num_of_relation)

    def forward(self, word, pos1, pos2, mask):
        x = self.embedding(word, pos1, pos2)
        x = self.encoder(x)

        logits = self.classifier(x)
        return logits


