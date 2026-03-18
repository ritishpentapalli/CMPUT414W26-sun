import torch
import torch.nn as nn


class ViTRestorer(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input size ({h}, {w}) must be divisible by patch size {self.patch_size}"
            )

        patches = self.patch_embed(x)
        hp, wp = patches.shape[2], patches.shape[3]

        tokens = patches.flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.num_patches:
            raise ValueError(
                "Input patch count differs from configured image_size. "
                "Set --patch_size/--image_size consistently with --patch_size argument."
            )

        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)

        feat = tokens.transpose(1, 2).reshape(b, -1, hp, wp)
        residual = self.decode(feat)
        return torch.clamp(x + residual, 0.0, 1.0)
