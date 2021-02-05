import torch.nn as nn


class ResidualNorm(nn.Module):
    def __init__(self):
        super(ResidualNorm).__init__()
    #     use norm = nn.LayerNorm(attention_output.size[1:]) if doesn't work at the forward

    def forward(self, inputs):
        attention_output, embedding_input = inputs
        norm = nn.LayerNorm(attention_output.size[1:])
        output = norm(embedding_input)+norm(attention_output)
        return output
