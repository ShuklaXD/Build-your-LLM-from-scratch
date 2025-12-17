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
        print("\nAttention Scores:\n", attn_scores)

        # Create a simple lower triangular mask to avoid attending to future tokens
        context_length = input_embeddings.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        print("\nMasked Attention Scores:\n", masked)

        # Apply mask (if needed)
        attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
        print("\nMasked Attention Weights:\n", attn_weights)

        # Apply dropout for regularization
        torch.manual_seed(123)
        dropout = torch.nn.Dropout(0.5)
        print(dropout(attn_weights))

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