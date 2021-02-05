import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embedding_space, qkv_dim):
        super(SelfAttention).__init__()
        query_dim = qkv_dim
        key_dim = qkv_dim
        value_dim = qkv_dim
        self.query_weight = nn.Parameter(torch.tensor(torch.rand(embedding_space, query_dim)), requires_grad=True)
        self.key_weight = nn.Parameter(torch.tensor(embedding_space, key_dim), requires_grad=True)
        self.value_weight = nn.Parameter(torch.tensor(embedding_space, value_dim), requires_grad=True)

    def forward(self, inputs):
        """

        :param inputs consist of input_embedding of the total word that has the size of (length * embedding space),
         mask and query_embedding
        :return: attention output
        """
        input_embedding, query_embedding, include_mask = inputs
        query = torch.matmul(query_embedding, self.query_weight)
        key = torch.matmul(input_embedding, self.key_weight)
        value = torch.matmul(input_embedding, self.value_weight)
        mul_term = query@torch.transpose(key, 0, 1)/10
        if include_mask:
            mask = torch.zeros_like(mul_term)
        else:
            mask = torch.triu(-torch.ones_like(mul_term) * float('inf'), 1)
        attention = nn.Softmax(mul_term + mask) * value
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_space, heads, qkv_dim):
        super(SelfAttention).__init__()
        self.multi_head_attention = nn.ModuleList([SelfAttention(embedding_space=embedding_space, qkv_dim=qkv_dim) for _ in range(heads)])
        # return dimensions is length * value dim
        self.concatenated_weights = nn.Parameter(torch.tensor(torch.randn(heads*qkv_dim, qkv_dim)))

    def forward(self, inputs):
        """

        :param inputs consist of input_embedding of the total word that has the size of (length * embedding space),
         mask and query_embedding
        :return: attention output
        """
        input_embedding, query_embedding, include_mask = inputs
        concat_attentions = torch.cat([attention([input_embedding, query_embedding, include_mask])
                                       for attention in self.multi_head_attention], dim=-1)
        output_attention = concat_attentions@self.concatenated_weights
        return output_attention
