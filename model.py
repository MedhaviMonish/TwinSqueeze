import torch
import torch.nn as nn
import torch.nn.functional as F

class TwinSqueeze(nn.Module):
    def __init__(self, input_dim=384, compressed_dim=32):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, compressed_dim),
        )

    def forward(self, emb1, emb2):
        z1 = self.compressor(emb1)
        z2 = self.compressor(emb2)

        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        cosine_sim = torch.sum(z1 * z2, dim=1)
        return cosine_sim
