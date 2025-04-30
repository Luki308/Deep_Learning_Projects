import torch
import torch.nn as nn


class CNNTransformer(nn.Module):
    def __init__(self, num_classes=12, time_steps=500, freq_bins=64, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()

        # --- CNN Feature Extractor ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),  # Reduce both time and freq
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        # ---Transformer Encoder ---
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim))  # Max 1000 sequence length

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---Classification Head ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        Input: x shape [B, T, F]
        """
        B, T, F = x.shape

        # 1. CNN feature extraction
        x = x.unsqueeze(1)  # add channel dimension: [B, 1, T, F]
        x = self.cnn(x)  # out: [B, embed_dim, T', F']

        # reshape
        B, C, T_prime, F_prime = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, T', F', C]
        x = x.reshape(B, T_prime * F_prime, C)  # [B, seq_len, embed_dim]

        # 2. Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]

        # 3. Transformer
        x = self.transformer_encoder(x)  # [B, seq_len, embed_dim]

        # 4. Mean over sequence
        x = x.mean(dim=1)  # global avrg pooling over time

        # 5. Classify
        out = self.classifier(x)  # [B, num_classes]
        return out
