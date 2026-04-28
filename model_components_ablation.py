
# model_components_ablation.py
# Branch on/off (time / time-frequency / spatial) without modifying the original model_components.py

from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the original building blocks (keeps behavior consistent)
from model_components import (
    EEGTransformerEncoder,
    UltraFastSharedSwinTransformer,
    SimpleSpatialCNN,
)


class AblationConcatFusion(nn.Module):
    """
    Concat-based fusion that supports enabling/disabling modalities.
    - Uses the same overall strategy as SimpleConcatFusion in model_components.py
    - But builds the fusion network for the active modality set only.
    """
    def __init__(
        self,
        modality_dims: Dict[str, int],
        output_dim: int,
        dropout: float = 0.2,
        modality_order: Optional[List[str]] = None,
    ):
        super().__init__()
        self.output_dim = int(output_dim)

        if modality_order is None:
            modality_order = ["raw", "stft", "spatial"]
        self.modality_order = [m for m in modality_order if m in modality_dims and modality_dims[m] > 0]
        if len(self.modality_order) == 0:
            raise ValueError("At least one modality must be enabled.")

        # Per-modality projection to a common dimension
        self.proj = nn.ModuleDict({
            m: nn.Linear(int(modality_dims[m]), self.output_dim) for m in self.modality_order
        })

        # Learnable modality weights (softmaxed)
        self.modality_weights = nn.Parameter(torch.ones(len(self.modality_order)) / len(self.modality_order))

        # Fusion MLP (same spirit as original SimpleConcatFusion)
        concat_dim = self.output_dim * len(self.modality_order)
        hidden1 = max(8, concat_dim // 2)

        self.fusion_net = nn.Sequential(
            nn.Linear(concat_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        feats: dict with keys in {"raw","stft","spatial"} and tensors shaped (B, dim)
        """
        # Align and weight
        weights = F.softmax(self.modality_weights, dim=0)
        aligned = []
        for i, m in enumerate(self.modality_order):
            x = feats[m]
            x = self.proj[m](x)  # (B, output_dim)
            x = x * weights[i]
            aligned.append(x)

        fused = torch.cat(aligned, dim=1)  # (B, output_dim * num_mod)
        out = self.fusion_net(fused)

        # Residual: average of aligned modality vectors
        residual = torch.stack(aligned, dim=0).mean(dim=0)
        out = out + 0.3 * residual
        return out


class TBEEGNetAblation(nn.Module):
    """
    TBEEGNet-compatible model with modality (branch) toggles.
    Input signature is identical: forward(x_raw, x_stft, x_spatial_temporal=None)

    - use_raw: time-domain branch (EEGTransformerEncoder)
    - use_stft: time-frequency branch (UltraFastSharedSwinTransformer)
    - use_spatial: spatial branch (SimpleSpatialCNN)

    This class does NOT modify model_components.py.
    """
    def __init__(
        self,
        n_channels: int,
        seq_length: int,
        hidden_dim: int = 64,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.3,
        n_classes: int = 2,
        fs: int = 128,
        embed_dim: int = 64,
        window_sizes: Optional[List[int]] = None,
        local_window: int = 32,

        # Branch toggles
        use_raw: bool = True,
        use_stft: bool = True,
        use_spatial: bool = True,

        # Compatibility-only (ignored)
        add_spatial_temporal: bool = True,
        electrode_positions=None,
        **kwargs,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.seq_length = int(seq_length)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.n_classes = int(n_classes)

        self.use_raw = bool(use_raw)
        self.use_stft = bool(use_stft)
        self.use_spatial = bool(use_spatial)

        if not (self.use_raw or self.use_stft or self.use_spatial):
            raise ValueError("At least one branch must be enabled (use_raw/use_stft/use_spatial).")

        if window_sizes is None:
            window_sizes = [4, 7, 14]

        # 1) Raw EEG (time-domain) branch
        if self.use_raw:
            self.raw_norm = nn.BatchNorm1d(self.n_channels)
            self.raw_transformer = EEGTransformerEncoder(
                input_dim=self.n_channels,
                hidden_dim=self.hidden_dim,
                n_heads=transformer_heads,
                n_layers=transformer_layers,
                local_window=local_window,
                conv_kernel_size=3,
                dropout=dropout * 0.5,
            )

        # 2) STFT branch (time-frequency)
        if self.use_stft:
            if self.seq_length <= 500:
                nperseg, noverlap = 64, 32
            elif self.seq_length <= 1000:
                nperseg, noverlap = 128, 64
            else:
                nperseg, noverlap = 256, 128

            freq_bins = nperseg // 2 + 1
            time_bins = max(1, (self.seq_length - noverlap) // (nperseg - noverlap))

            self.stft_processor = UltraFastSharedSwinTransformer(
                n_channels=self.n_channels,
                freq_bins=freq_bins,
                time_bins=time_bins,
                embed_dim=self.embed_dim,
                window_sizes=window_sizes,
                num_heads=2,
                num_layers=1,
                mlp_ratio=2.0,
                drop_rate=dropout * 0.5,
            )

        # 3) Spatial branch
        self.spatial_dim = self.hidden_dim // 4 if self.use_spatial else 0
        if self.use_spatial:
            self.spatial_temporal_branch = SimpleSpatialCNN(
                grid_shape=(5, 6),
                out_dim=self.spatial_dim,
            )

        # 4) Fusion
        modality_dims = {}
        if self.use_raw:
            modality_dims["raw"] = self.hidden_dim
        if self.use_stft:
            modality_dims["stft"] = self.embed_dim
        if self.use_spatial:
            modality_dims["spatial"] = self.spatial_dim

        self.fusion = AblationConcatFusion(
            modality_dims=modality_dims,
            output_dim=self.hidden_dim,
            dropout=dropout,
            modality_order=["raw", "stft", "spatial"],
        )

        # 5) Classifier (same style as original)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_raw: torch.Tensor, x_stft: torch.Tensor, x_spatial_temporal: Optional[torch.Tensor] = None):
        """
        x_raw: (B, channels, time)
        x_stft: (B, channels, freq, time)
        x_spatial_temporal: (B, H, W) or (B, H, W, T)
        """
        feats = {}

        if self.use_raw:
            xr = self.raw_norm(x_raw)
            xr = xr.transpose(1, 2)  # (B, time, channels)
            feats["raw"] = self.raw_transformer(xr)  # (B, hidden_dim)

        if self.use_stft:
            feats["stft"] = self.stft_processor(x_stft)  # (B, embed_dim)

        if self.use_spatial:
            if x_spatial_temporal is None:
                # Keep deterministic behavior: spatial disabled by input -> use zeros
                B = x_raw.size(0)
                feats["spatial"] = torch.zeros((B, self.spatial_dim), device=x_raw.device, dtype=x_raw.dtype)
            else:
                feats["spatial"] = self.spatial_temporal_branch(x_spatial_temporal)  # (B, spatial_dim)

        fused = self.fusion(feats)
        logits = self.classifier(fused)

        # Temperature scaling (keep original behavior)
        temperature = 1.05
        logits = logits / temperature
        return logits, fused
