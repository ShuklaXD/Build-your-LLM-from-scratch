import torch

input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=output_dim)
print("Embedding weights:\n", embedding_layer.weight)
token_id = 3
print(f"\nEmbedding for token id {token_id}:\n", embedding_layer(torch.tensor([token_id])))

print(embedding_layer(input_ids))