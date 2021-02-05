import torch.nn as nn
from attention import MultiHeadAttention
from utils import ResidualNorm


class Decoder(nn.Module):
    def __init__(self, embedding_space, qkv_dim, heads, sequence_length):
        super(Decoder).__init__()
        self.multi_attention = MultiHeadAttention(embedding_space=embedding_space, heads=heads, qkv_dim=qkv_dim)
        self.residual = ResidualNorm()
        self.linear = nn.Linear([sequence_length, qkv_dim], [sequence_length, qkv_dim])
        self.relu = nn.ReLU()

    def forward(self, inputs):
        sequence_input, encoder_output = inputs
        query_output = self.residual(inputs=[self.multi_attention([sequence_input,
                                                                   sequence_input, True]), sequence_input])
        attention_output = self.residual(inputs=[self.multi_attention([sequence_input,
                                                                       query_output, False]), query_output])

        forward_output = self.residual(inputs=[self.linear(attention_output), attention_output])
        return forward_output
