import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        
        # Dimension per head
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, input_embeddings):
        b, num_tokens, dim_in = input_embeddings.shape
        
        # Compute queries, keys, and values
        queries = self.W_query(input_embeddings)  # (b, num_tokens, d_out)
        keys = self.W_key(input_embeddings)       # (b, num_tokens, d_out)
        values = self.W_value(input_embeddings)   # (b, num_tokens, d_out)
        
        # Reshape for multi-head attention
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)  # (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)        # (b, num_tokens, num_heads, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)    # (b, num_tokens, num_heads, head_dim)
        
        # Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)        # (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)    # (b, num_heads, num_tokens, head_dim)
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)  # (b, num_heads, num_tokens, num_tokens)

        # Apply causal mask
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens].bool(), -torch.inf)
        
        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  # (b, num_heads, num_tokens, num_tokens)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        context_vec = attn_weights @ values  # (b, num_heads, num_tokens, head_dim)
        
        # Concatenate heads and project
        context_vec = context_vec.transpose(1, 2) # (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # (b, num_tokens, d_out)
        
        # Final linear projection (optional)
        output = self.out_proj(context_vec)  # (b, num_tokens, d_out)
        
        return output
    
if __name__ == "__main__":
    d_in = 3
    d_out = 2
    num_heads = 2
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with  
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55]] # step
        )
    
    batch = torch.stack((inputs, inputs), dim=0)  # Create a batch of size 2
    print(batch.shape)
    
    torch.manual_seed(123)
    context_length = batch.shape[1]
    mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=num_heads)
    output = mha(batch)
    print(output)
    print("output.shape:", output.shape)
