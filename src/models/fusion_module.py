import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, vision_dim, llm_dim, num_hidden_layers=1, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = vision_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, llm_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = llm_dim

    def forward(self, vision_pooled):
        # vision_pooled: [B, vision_dim]
        return self.projector(vision_pooled) # [B, llm_dim]
