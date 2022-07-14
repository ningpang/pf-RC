import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import base_model
class Attention_Memory(base_model):
    def __init__(self, mem_slots, input_size, output_size, key_size, head_size, num_head=4):
        super(Attention_Memory, self).__init__()
        self.mem_slots = mem_slots
        self.mem_size = input_size
        self.input_size = input_size
        self.output_size = output_size

        self.head_szie = head_size
        self.num_head = num_head

        # query-key-value
        self.query_size = key_size
        self.key_size = key_size
        self.value_size = key_size

        self.q_projector = nn.Linear(self.mem_size, self.num_head*self.query_size)
        self.q_layernorm = nn.LayerNorm([self.num_head, self.query_size])

        self.kv_projector = nn.Linear(self.mem_size, self.num_head*(self.key_size+self.value_size))
        self.k_layernorm = nn.LayerNorm([self.num_head, self.query_size])
        self.v_layernorm = nn.LayerNorm([self.num_head, self.query_size])

        self.concatenate_mlp = nn.Linear(self.num_head*self.value_size, self.mem_size)
        self.concatenate_layernorm = nn.LayerNorm([self.mem_size])
        self.attention_ouput_layernorm = nn.LayerNorm([self.mem_size])

        self.output_mlp = nn.Linear(self.mem_size, self.output_size)
        self.output_layernorm = nn.LayerNorm([self.output_size])

    def multihead_attention(self, input):

        q = self.q_projector(input) # (num, cur_rel+1, head*size)

        q_reshape = q.view(q.shape[0], q.shape[1], self.num_head, self.query_size) # (num, cur_rel+1, head, size)


        q_reshape = self.q_layernorm(q_reshape)
        q_transpose = q_reshape.permute(0, 2, 1, 3) # (num, head, cur_rel+1, size)

        kv = self.kv_projector(input)
        kv_reshape = kv.view(kv.shape[0], kv.shape[1], self.num_head, (self.key_size+self.value_size)) # (num, cur_rel+1, head, 2*size)
        k_reshape, v_reshape = torch.split(kv_reshape, [self.key_size, self.value_size], dim=-1)
        k_reshape = self.k_layernorm(k_reshape)
        v_reshape = self.v_layernorm(v_reshape)
        k_transpose = k_reshape.permute(0, 2, 1, 3)
        v_transpose = v_reshape.permute(0, 2, 1, 3)

        q_transpose *= (self.key_size ** -0.5)
        dot_product = torch.matmul(q_transpose, k_transpose.permute(0, 1, 3, 2))

        weights = F.softmax(dot_product, dim=-1)
        weight_output = torch.matmul(weights, v_transpose)

        output_transpose = weight_output.permute(0, 2, 1, 3).contiguous()
        output_transpose = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))


        output_transpose = self.concatenate_mlp(output_transpose)
        output_transpose = self.concatenate_layernorm(output_transpose)

        return output_transpose

    def attention_over_memory(self, input, memory):
        input_reshape = input.unsqueeze(dim=1)
        memory_plus_input = torch.cat([memory, input_reshape], dim=1)
        attention_output = self.multihead_attention(memory_plus_input)
        attention_output = self.attention_ouput_layernorm(attention_output+memory_plus_input)

        output = self.output_mlp(attention_output)
        output = F.gelu(output)
        output = self.output_layernorm(output+attention_output)
        return output

    def forward(self, input, memeory):
        output = self.attention_over_memory(input, memeory)

        output = output[:, -1, :] # take the index of instance

        return output
