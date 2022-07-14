import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .utils import sort_batch_by_length, init_lstm, init_linear
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


class Encoder(nn.Module):
    def __init__(self, word_embedding_dim=50, pos_embedding_dim=5, hidden_dim=200):
        super(Encoder, self).__init__()
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.lstm_base = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_dim,
                                 bidirectional=True, batch_first=True)
        self.project = nn.Linear(hidden_dim * 4, hidden_dim)
        self.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            init_linear(m)
        elif classname.find('LSTM') != -1:
            init_lstm(m)

    def max_mean_agg(self, inputs, mask):
        max_input, _ = torch.max(inputs, 1)
        mean_input = torch.sum(inputs, 1)/torch.sum(mask, 1).unsqueeze(-1)

        output = torch.cat([max_input, mean_input], 1)
        return self.project(output)

    def forward(self, inputs, lengths, mask):
        sorted_inputs, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(inputs,
                                                                                              lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths.cpu(),
                                                     batch_first=True)
        output, _ = self.lstm_base(packed_sequence_input)
        unpacked_sequence_tensor, _ = pad_packed_sequence(output, batch_first=True)
        unpacked_sequence_tensor = unpacked_sequence_tensor.index_select(0, restoration_indices)

        output = self.max_mean_agg(unpacked_sequence_tensor, mask)
        return output

class Softmax_Layer(nn.Module):
    def __init__(self, input_size, num_class):
        super(Softmax_Layer, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=True)

    def forward(self, input):
        logits = self.fc(input)
        return logits

class LSTM_model(nn.Module):
    def __init__(self, config, word_vec_mat, word2id):
        super(LSTM_model, self).__init__()
        self.config = config
        self.hidden_dim = config.lstm_hidden_size

        self.embedding = Embedding(word_vec_mat, config.max_length,
                                   config.word_embedding_dim, config.pos_embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = Encoder(config.word_embedding_dim,
                config.pos_embedding_dim, config.lstm_hidden_size)
        self.classifier = Softmax_Layer(input_size=config.lstm_hidden_size, num_class=config.num_of_relation)


    def forward(self, word, pos1, pos2, mask):

        mask = (mask != 0).float()
        max_length = mask.long().sum(1).max().item()
        input_mask = mask[:, :max_length].contiguous()
        sequence_lengths = input_mask.long().sum(1)

        embedding = self.embedding(word, pos1, pos2)
        embedding_ = self.dropout(embedding[:, :max_length]).contiguous()

        output = self.encoder(embedding_, sequence_lengths, mask)
        logits = self.classifier(output)

        return logits



