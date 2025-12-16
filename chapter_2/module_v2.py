import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, input_embeddings):
        # Compute queries, keys, and values
        queries = self.W_query(input_embeddings)
        keys = self.W_key(input_embeddings)
        values = self.W_value(input_embeddings)

        # Compute attention scores
        attn_scores = queries @ keys.T

        # Scale scores
        d_k = keys.shape[-1]
        attn_scores = attn_scores / (d_k ** 0.5)

        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        output = attn_weights @ values
        return output

# Example usage
d_in = 3
d_out = 2
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with  
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] # step
    )

if __name__ == "__main__":
    torch.manual_seed(789)
    sa_v1 = SelfAttention_v2(d_in, d_out)
    print(sa_v1(inputs))