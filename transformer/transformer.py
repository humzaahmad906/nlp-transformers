import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn):
    def __init__(self, n_encoders, n_decoders, embedding_space, qkv_dim, heads, sequence_length):
        super(Transformer).__init__()
        self.n_encoders = n_encoders
        self.n_decoders = n_decoders
        self.encoder = Encoder(embedding_space, qkv_dim, heads, sequence_length)
        self.decoder = Decoder(embedding_space, qkv_dim, heads, sequence_length)
        self.linear = nn.Linear([sequence_length, qkv_dim], [sequence_length, embedding_space])

    def forward(self, inputs):
        encoder_x, decoder_x = inputs
        for i in range(self.n_encoders):
            encoder_x = self.encoder(encoder_x)
        for i in range(self.n_decoders):
            decoder_x = self.decoder([decoder_x, encoder_x])
        self.linear(decoder_x)
