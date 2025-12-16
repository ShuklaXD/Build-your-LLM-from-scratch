import torch

input_ids = torch.tensor([2, 3, 5, 1])

class Embedding():
    def __init__(self, num_embeddings, embedding_dim):
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    
    def forward(self, input_ids):
        return self.embedding(input_ids)

if __name__ == "__main__":
    vocab_size = 50257
    output_dim = 256

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=output_dim)
    print("Embedding weights:\n", embedding_layer.weight)
    token_id = 3
    print(f"\nEmbedding for token id {token_id}:\n", embedding_layer(torch.tensor([token_id])))

    print(embedding_layer(input_ids))