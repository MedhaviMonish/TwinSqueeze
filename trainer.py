import torch
import torch.nn as nn
from typing import Callable

class TwinSqueezeTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: Callable, device: str = "cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.loss_log = []

    def add_neftune_noise(self, embedding, alpha = 0.0):
        d = embedding.shape[-1]
        noise = torch.rand_like(embedding) * 2 - 1  # Uniform(-1, 1)
        scale = alpha / (d ** 0.5)
        return embedding + noise * scale
    
    def train(self, dataloader, epochs: int = 50, alpha: float=0.0):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for emb1, emb2, labels in dataloader:
                emb1, emb2, labels = emb1.to(self.device), emb2.to(self.device), labels.to(self.device)
                emb1, emb2 = self.add_neftune_noise(emb1, alpha), self.add_neftune_noise(emb2, alpha)
                self.optimizer.zero_grad()
                preds = self.model(emb1, emb2)
                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.loss_log.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def get_loss_log(self):
        return self.loss_log