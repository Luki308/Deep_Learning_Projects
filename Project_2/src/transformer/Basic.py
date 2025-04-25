import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, dim]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class SpectrogramTransformerTSTF(nn.Module):
    def __init__(self, time_steps=500, freq_bins=128, dim=768, num_heads=12, num_layers=6, num_classes=50):
        super().__init__()

        # --- layers doing embedding ---
        self.temporal_embed = nn.Linear(freq_bins, dim)     # [B, T, 128] -> [B, T, dim]
        self.freq_embed = nn.Linear(time_steps, dim)        # [B, F, T] -> [B, F, dim]

        self.cls_token_t = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_f = nn.Parameter(torch.randn(1, 1, dim))

        self.pos_enc_t = PositionalEncoding(dim, max_len=time_steps + 1)
        self.pos_enc_f = PositionalEncoding(dim, max_len=freq_bins + 1)

        # --- transformer encoders for temporal and frequency parts ---
        encoder_layer = lambda: nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)

        self.temporal_encoder = nn.TransformerEncoder(encoder_layer(), num_layers=num_layers)
        self.freq_encoder = nn.TransformerEncoder(encoder_layer(), num_layers=num_layers)

        # --- MLP head ---
        self.mlp_head = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        """
        x: [B, T, F] — a log-mel spectrogram
        """
        B, T, F = x.size()

        # --- stream for temporal part ---
        x_t = self.temporal_embed(x)  # [B, T, dim]
        cls_t = self.cls_token_t.expand(B, -1, -1)
        x_t = torch.cat((cls_t, x_t), dim=1)  # [B, T+1, dim]
        x_t = self.pos_enc_t(x_t)
        x_t = self.temporal_encoder(x_t)  # [B, T+1, dim]
        cls_out_t = x_t[:, 0]  # [B, dim]

        # --- stream for frequency ---
        x_f = x.transpose(1, 2)  # [B, F, T]
        x_f = self.freq_embed(x_f)  # [B, F, dim]
        cls_f = self.cls_token_f.expand(B, -1, -1)
        x_f = torch.cat((cls_f, x_f), dim=1)  # [B, F+1, dim]
        x_f = self.pos_enc_f(x_f)
        x_f = self.freq_encoder(x_f)  # [B, F+1, dim]
        cls_out_f = x_f[:, 0]  # [B, dim]

        # --- combine and classify ---
        combined = torch.cat((cls_out_t, cls_out_f), dim=-1)  # [B, 2*dim]
        return self.mlp_head(combined)  # [B, num_classes]
