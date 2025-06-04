from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
import math
from sentence_transformers import SentenceTransformer
encoder_model_name="all-MiniLM-L6-v2"
encoder = SentenceTransformer(encoder_model_name)

def load_stsb_data(batch_size=32, alpha=0.0):
    dataset = load_dataset("glue", "stsb")

    def normalize(example):
        x = example["label"] / 5.0
        example["label"] = (math.tanh(4 * (x - 0.5)) + 1) / 2
        return example

    dataset = dataset.map(normalize)

    def add_neftune_noise(embedding):
        d = embedding.shape[-1]
        noise = torch.rand_like(embedding) * 2 - 1  # Uniform(-1, 1)
        scale = alpha / (d ** 0.5)
        return embedding + noise * scale

    # Convert to tensors
    train_emb1 = add_neftune_noise(encoder.encode(dataset["train"]["sentence1"], convert_to_tensor="pt"))
    train_emb2 = add_neftune_noise(encoder.encode(dataset["train"]["sentence2"], convert_to_tensor="pt"))
    train_labels = torch.tensor(dataset["train"]["label"])
    
    # I had to use validation because test does not have labels because its used for leaderboard submissions.
    test_emb1 = add_neftune_noise(encoder.encode(dataset["validation"]["sentence1"], convert_to_tensor="pt"))
    test_emb2 = add_neftune_noise(encoder.encode(dataset["validation"]["sentence2"], convert_to_tensor="pt"))
    test_labels = torch.tensor(dataset["validation"]["label"])

    train_dataset = TensorDataset(train_emb1, train_emb2, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_emb1, test_emb2, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
