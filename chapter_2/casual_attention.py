import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.context_length = context_length
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, input_embeddings):
        b, num_tokens, dim_in = input_embeddings.shape
        keys = self.W_key(input_embeddings)      # (b, num_tokens, d_out)
        queries = self.W_query(input_embeddings)  # (b, num_tokens, d_out)
        values = self.W_value(input_embeddings)   # (b, num_tokens, d_out

        attn_scores = queries @ keys.transpose(1, 2)  # (b, num_tokens, num_tokens)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens].bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  # (b, num_tokens, num_tokens)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values  # (b, num_tokens, d_out)
        return context_vec


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
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)