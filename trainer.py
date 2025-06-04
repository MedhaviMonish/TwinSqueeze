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

    def train(self, dataloader, epochs: int = 50):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for emb1, emb2, labels in dataloader:
                emb1, emb2, labels = emb1.to(self.device), emb2.to(self.device), labels.to(self.device)
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