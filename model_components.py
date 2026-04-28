import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ========== Utility Functions ==========
def to_2tuple(x):
    """Convert to 2-tuple"""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.")

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


# ========== Drop Path ==========
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ========== MLP ==========
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ========== EEG 특화 Transformer Encoder ==========
class EEGTransformerEncoder(nn.Module):
    """
    EEG 신호에 특화된 Transformer Encoder
    - 시간적 지역성을 고려한 local attention
    - 1D convolution과 결합한 hybrid approach
    - EEG 신호의 특성에 맞는 positional encoding
    """
    def __init__(self, input_dim, hidden_dim, n_heads=2, n_layers=1, 
                 local_window=32, conv_kernel_size=3, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.local_window = local_window
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # EEG-specific positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 5000, hidden_dim) * 0.02)
        
        # Hybrid layers: Conv + Local Attention
        self.layers = nn.ModuleList([
            EEGHybridLayer(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                local_window=local_window,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        print(f"🧠 EEG-Specialized Transformer:")
        print(f"   • Local attention window: {local_window}")
        print(f"   • Conv kernel size: {conv_kernel_size}")
        print(f"   • Layers: {n_layers}, Heads: {n_heads}")
        
    def forward(self, x):
        """
        x: [B, seq_len, channels]
        """
        B, seq_len, C = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [B, seq_len, hidden_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply hybrid layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Global average pooling
        output = x.mean(dim=1)  # [B, hidden_dim]
        
        return output


class EEGHybridLayer(nn.Module):
    """
    EEG용 Hybrid Layer: 1D Conv + Local Attention
    """
    def __init__(self, hidden_dim, n_heads, local_window, conv_kernel_size, dropout):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.local_window = local_window
        
        # 1D Convolution for local temporal patterns
        self.conv1d = nn.Conv1d(
            hidden_dim, hidden_dim, 
            kernel_size=conv_kernel_size, 
            padding=conv_kernel_size//2,
            groups=hidden_dim//4  # Depthwise-like convolution
        )
        
        # Local attention
        self.local_attention = LocalAttention(
            hidden_dim, n_heads, local_window, dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        x: [B, seq_len, hidden_dim]
        """
        # 1) 1D Convolution for local temporal patterns
        residual = x
        x = self.norm1(x)
        x_conv = x.transpose(1, 2)  # [B, hidden_dim, seq_len]
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, seq_len, hidden_dim]
        x = residual + x_conv
        
        # 2) Local attention
        residual = x
        x = self.norm2(x)
        x = self.local_attention(x)
        x = residual + x
        
        # 3) Feed-forward network
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class LocalAttention(nn.Module):
    """
    Local Attention: 지정된 윈도우 내에서만 attention 계산
    """
    def __init__(self, hidden_dim, n_heads, local_window, dropout):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.local_window = local_window
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [B, seq_len, hidden_dim]
        """
        B, seq_len, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Local attention computation
        if seq_len <= self.local_window:
            # If sequence is shorter than window, use global attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
        else:
            # Use local attention
            out = self._local_attention(q, k, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, seq_len, C)
        out = self.proj(out)
        
        return out
    
    def _local_attention(self, q, k, v):
        """
        Local attention implementation
        """
        B, n_heads, seq_len, head_dim = q.shape
        window_size = self.local_window
        
        # Pad sequence to make it divisible by window size
        pad_len = (window_size - seq_len % window_size) % window_size
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        
        padded_len = q.size(2)
        n_windows = padded_len // window_size
        
        # Reshape to windows
        q = q.view(B, n_heads, n_windows, window_size, head_dim)
        k = k.view(B, n_heads, n_windows, window_size, head_dim)
        v = v.view(B, n_heads, n_windows, window_size, head_dim)
        
        # Compute attention within each window
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        
        # Reshape back
        out = out.view(B, n_heads, padded_len, head_dim)
        
        # Remove padding
        if pad_len > 0:
            out = out[:, :, :seq_len, :]
        
        return out


# ========== 진짜 빠른 공유 CNN + 초경량 Swin (기존 코드 유지) ==========
class UltraFastSharedSwinTransformer(nn.Module):
    """
    🚀 진짜 공유 CNN + 경량 Multi-scale Swin Transformer
    
    핵심 최적화:
    1. 채널별 루프 완전 제거 → 진짜 공유 처리
    2. 경량 Multi-scale 윈도우 (성능 vs 속도 균형)
    3. 어텐션 헤드 수 최소화
    4. 레이어 수 최소화
    """
    
    def __init__(self, n_channels, freq_bins, time_bins, embed_dim=64, 
                 window_sizes=[2, 3, 4], num_heads=2, num_layers=1, 
                 mlp_ratio=2., drop_rate=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.window_sizes = window_sizes  # Multi-scale 지원!
        self.num_heads = num_heads
        
        print(f"🚀 초고속 공유 CNN + 경량 Multi-scale Swin:")
        print(f"   Input: ({n_channels}, {freq_bins}, {time_bins}) → {embed_dim}")
        print(f"   Windows: {window_sizes}, Heads: {num_heads}, Layers: {num_layers}")
        
        # 1. 진짜 공유 CNN (모든 채널 한 번에 처리!)
        self.shared_cnn = nn.Sequential(
            # 한 번에 모든 채널 처리
            nn.Conv2d(n_channels, embed_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        
        # 2. 간단한 패치 임베딩 (1x1 conv)
        self.patch_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        # 3. 초경량 위치 인코딩 (학습 가능한 간단한 버전)
        self.H, self.W = freq_bins, time_bins
        num_patches = self.H * self.W
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
        # 4. 경량 Multi-scale Swin 블록!
        self.swin_block = LightMultiScaleSwinBlock(
            dim=embed_dim,
            window_sizes=window_sizes,  # Multi-scale 사용!
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate
        )
        
        # 5. 출력 정규화
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        x: (B, n_channels, freq, time)
        
        🚀 초고속 처리:
        1. 모든 채널을 한 번에 공유 CNN 처리
        2. 단일 경량 Swin 블록 적용
        3. 글로벌 평균 풀링으로 마무리
        """
        B, C, H, W = x.shape
        
        # Step 1: 진짜 공유 CNN (모든 채널 동시 처리!)
        features = self.shared_cnn(x)  # (B, embed_dim, H, W)
        
        # Step 2: 패치 임베딩
        patches = self.patch_embed(features)  # (B, embed_dim, H, W)
        patches = patches.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        
        # Step 3: 위치 인코딩 (간단한 방식으로 수정)
        num_patches = patches.size(1)
        if num_patches <= self.pos_embed.size(1):
            patches = patches + self.pos_embed[:, :num_patches, :]
        else:
            # 크기가 다를 때는 간단히 반복하거나 잘라서 사용
            if num_patches > self.pos_embed.size(1):
                # 더 클 때는 반복해서 사용
                repeat_times = (num_patches // self.pos_embed.size(1)) + 1
                pos_embed = self.pos_embed.repeat(1, repeat_times, 1)[:, :num_patches, :]
            else:
                # 더 작을 때는 잘라서 사용
                pos_embed = self.pos_embed[:, :num_patches, :]
            patches = patches + pos_embed
        
        # Step 4: 경량 Multi-scale Swin 블록!
        patches = self.swin_block(patches, H, W)  # H, W 전달
        
        # Step 5: 정규화 + 글로벌 풀링
        patches = self.norm(patches)  # (B, H*W, embed_dim)
        output = patches.mean(dim=1)  # (B, embed_dim) - 간단한 평균 풀링
        
        return output


# ========== 경량 Multi-Scale Swin 블록 (기존 코드 유지) ==========
class LightMultiScaleSwinBlock(nn.Module):
    """
    경량 Multi-Scale Swin 블록:
    - 여러 윈도우 크기 사용하되 경량화
    - 각 스케일별 가중치 학습
    - 최소한의 어텐션 헤드
    """
    
    def __init__(self, dim, window_sizes=[2, 3, 4], num_heads=2, mlp_ratio=2., drop=0.1):
        super().__init__()
        self.dim = dim
        self.window_sizes = window_sizes
        self.num_heads = num_heads
        
        # 정규화
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 각 윈도우 크기별 어텐션 (경량화)
        self.window_attentions = nn.ModuleList([
            SimpleWindowAttention(
                dim=dim,
                window_size=ws,
                num_heads=num_heads,
                drop=drop
            ) for ws in window_sizes
        ])
        
        # 윈도우별 가중치 학습 (간단한 버전)
        self.scale_weights = nn.Sequential(
            nn.Linear(dim, len(window_sizes)),
            nn.Softmax(dim=-1)
        )
        
        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden,
            drop=drop
        )
        
        print(f"🎯 Multi-scale Windows: {window_sizes} (경량화 버전)")
        
    def forward(self, x, H, W):
        """
        x: (B, H*W, C)
        """
        B, N, C = x.shape
        
        # 글로벌 특징으로 스케일 가중치 계산
        global_feat = x.mean(dim=1)  # (B, C)
        scale_weights = self.scale_weights(global_feat)  # (B, num_scales)
        
        # Self-attention (Multi-scale)
        shortcut = x
        x = self.norm1(x)
        
        # 각 윈도우 크기별 어텐션 계산
        scale_outputs = []
        for i, window_attn in enumerate(self.window_attentions):
            scale_out = window_attn(x, H, W)  # (B, N, C)
            scale_outputs.append(scale_out)
        
        # 가중 결합
        final_output = torch.zeros_like(x)
        for i, scale_out in enumerate(scale_outputs):
            weight = scale_weights[:, i].view(B, 1, 1)  # (B, 1, 1)
            final_output += scale_out * weight
        
        x = shortcut + final_output
        
        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        
        return x


# ========== 초간단 윈도우 어텐션 (기존 코드 유지) ==========
class SimpleWindowAttention(nn.Module):
    """
    초간단 윈도우 어텐션:
    - relative position bias 생략
    - 복잡한 윈도우 파티셔닝 간소화
    - 최소한의 연산으로 윈도우 효과만 구현
    """
    
    def __init__(self, dim, window_size=7, num_heads=2, drop=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x, H, W):
        """
        x: (B, H*W, C)
        """
        B, N, C = x.shape
        
        # 윈도우 크기가 전체보다 크면 글로벌 어텐션
        if self.window_size >= min(H, W):
            return self._global_attention(x)
        else:
            return self._simple_window_attention(x, H, W)
    
    def _global_attention(self, x):
        """글로벌 어텐션 (빠른 버전)"""
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
    
    def _simple_window_attention(self, x, H, W):
        """간소화된 윈도우 어텐션"""
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        
        # 간단한 윈도우 파티셔닝 (패딩 최소화)
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = H + pad_h, W + pad_w
        
        # 윈도우로 분할
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, ws * ws, C)  # (B*num_windows, ws*ws, C)
        
        # 윈도우별 어텐션
        qkv = self.qkv(x).reshape(-1, ws * ws, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        x = self.proj(x)
        
        # 윈도우 합치기
        x = x.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)
        
        # 패딩 제거
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        return x.view(B, H * W, C)


# ========== 초경량 2D CNN (기존 코드 유지) ==========
class SimpleSpatialCNN(nn.Module):
    """초경량 2D 공간 처리 (3D CNN 제거)"""
    def __init__(self, grid_shape=(5, 6), out_dim=32):  # 출력 차원도 축소
        super().__init__()
        
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # 바로 1x1로 축소
            nn.Flatten(),
        )
        
        self.output_proj = nn.Linear(16, out_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x: (B, H, W) 또는 (B, H, W, T)
        if x.dim() == 4:  # (B, H, W, T) → 시간축 평균화
            x = x.mean(dim=-1)  # (B, H, W)
        
        if x.dim() == 3:  # (B, H, W)
            x = x.unsqueeze(1)  # (B, 1, H, W)
        
        features = self.spatial_cnn(x)  # (B, 16)
        features = self.dropout(features)
        output = self.output_proj(features)  # (B, out_dim)
        
        return output


# ========== 🎯 새로운 간단한 Concat 기반 Fusion ==========
class SimpleConcatFusion(nn.Module):
    """
    🎯 Cross Attention 대신 단순한 Concat 기반 Fusion
    - 훨씬 빠르고 간단함
    - 안정적인 성능
    - 메모리 효율적
    """
    def __init__(self, raw_dim, stft_dim, spatial_dim, output_dim, dropout=0.2):
        super().__init__()
        
        self.raw_dim = raw_dim
        self.stft_dim = stft_dim
        self.spatial_dim = spatial_dim
        self.output_dim = output_dim
        
        # 차원 정렬을 위한 projection layers
        self.raw_proj = nn.Linear(raw_dim, output_dim)
        self.stft_proj = nn.Linear(stft_dim, output_dim)
        self.spatial_proj = nn.Linear(spatial_dim, output_dim) if spatial_dim > 0 else None
        
        # Concat된 features의 총 차원
        concat_dim = output_dim * (2 + (1 if spatial_dim > 0 else 0))
        
        # 🎯 Simple fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.LayerNorm(concat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(concat_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(output_dim, output_dim)
        )
        
        # 🎯 Modality importance weights (learnable)
        num_modalities = 2 + (1 if spatial_dim > 0 else 0)
        self.modality_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        
        print("🎯 Simple Concat Fusion 사용!")
        print(f"   • Raw dim: {raw_dim} → {output_dim}")
        print(f"   • STFT dim: {stft_dim} → {output_dim}")
        if spatial_dim > 0:
            print(f"   • Spatial dim: {spatial_dim} → {output_dim}")
        print(f"   • Final concat dim: {concat_dim} → {output_dim}")
        
    def forward(self, raw_feat, stft_feat, spatial_feat=None):
        """
        raw_feat: (B, raw_dim)
        stft_feat: (B, stft_dim)
        spatial_feat: (B, spatial_dim) or None
        """
        # 1. 차원 정렬
        raw_aligned = self.raw_proj(raw_feat)      # (B, output_dim)
        stft_aligned = self.stft_proj(stft_feat)   # (B, output_dim)
        
        # 2. Modality-wise weighting
        features = [raw_aligned, stft_aligned]
        weights = F.softmax(self.modality_weights, dim=0)
        
        # Apply weights
        raw_aligned = raw_aligned * weights[0]
        stft_aligned = stft_aligned * weights[1]
        
        if spatial_feat is not None and self.spatial_proj is not None:
            spatial_aligned = self.spatial_proj(spatial_feat)  # (B, output_dim)
            spatial_aligned = spatial_aligned * weights[2]
            features = [raw_aligned, stft_aligned, spatial_aligned]
        else:
            features = [raw_aligned, stft_aligned]
        
        # 3. 🎯 Simple concatenation
        fused_features = torch.cat(features, dim=1)  # (B, concat_dim)
        
        # 4. Fusion network
        output = self.fusion_net(fused_features)  # (B, output_dim)
        
        # 5. 🎯 Residual connection (simple average of aligned features)
        residual = torch.stack(features, dim=0).mean(dim=0)  # (B, output_dim)
        output = output + 0.3 * residual  # 약간의 residual
        
        return output


# ========== 🎯 개선된 최종 EEG 모델 ==========
class TBEEGNet(nn.Module):
    """
    🎯 개선된 Hybrid EEG 모델:
    1. EEG 특화 Transformer Encoder
    2. 단순한 Concat 기반 Fusion
    3. 더 안정적이고 빠른 성능
    """
    def __init__(self, n_channels, seq_length, hidden_dim=64, transformer_layers=2,
                 transformer_heads=4, dropout=0.3, n_classes=2, fs=128,
                 embed_dim=64, window_sizes=[4, 7, 14], local_window=32,
                 add_spatial_temporal=True, electrode_positions=None, **kwargs):
        super().__init__()
        
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.add_spatial_temporal = add_spatial_temporal
        
        print(f"🎯 개선된 EEG 모델:")
        print(f"   • Raw EEG: EEG 특화 Transformer (local window: {local_window})")
        print(f"   • STFT: 공유 CNN + Multi-scale Swin")
        print(f"   • Spatial: 2D CNN")
        print(f"   • Fusion: 🎯 Simple Concat (Cross Attention 제거)")
        
        # 1. Raw EEG: 🎯 EEG 특화 Transformer
        self.raw_norm = nn.BatchNorm1d(n_channels)
        self.raw_transformer = EEGTransformerEncoder(
            input_dim=n_channels,
            hidden_dim=hidden_dim,
            n_heads=transformer_heads,
            n_layers=transformer_layers,
            local_window=local_window,
            conv_kernel_size=3,
            dropout=dropout * 0.5  # Transformer에는 낮은 dropout
        )
        
        # 2. STFT: 초고속 공유 CNN + 경량 Swin
        if seq_length <= 500:
            nperseg, noverlap = 64, 32
        elif seq_length <= 1000:
            nperseg, noverlap = 128, 64
        else:
            nperseg, noverlap = 256, 128
        
        freq_bins = nperseg // 2 + 1
        time_bins = max(1, (seq_length - noverlap) // (nperseg - noverlap))
        
        self.stft_processor = UltraFastSharedSwinTransformer(
            n_channels=n_channels,
            freq_bins=freq_bins,
            time_bins=time_bins,
            embed_dim=embed_dim,
            window_sizes=window_sizes,
            num_heads=2,  # 헤드 수 축소
            num_layers=1,  # 레이어 수 축소
            mlp_ratio=2.,  # MLP 비율 축소
            drop_rate=dropout * 0.5
        )
        
        # 3. Spatial: 초경량 2D CNN
        spatial_dim = hidden_dim // 4 if add_spatial_temporal else 0
        if add_spatial_temporal:
            self.spatial_temporal_branch = SimpleSpatialCNN(
                grid_shape=(5, 6),
                out_dim=spatial_dim
            )
        
        # 4. 🎯 Simple Concat Fusion (Cross Attention 제거!)
        self.fusion = SimpleConcatFusion(
            raw_dim=hidden_dim,
            stft_dim=embed_dim,
            spatial_dim=spatial_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # 5. 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
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

    def forward(self, x_raw, x_stft, x_spatial_temporal=None):
        """
        x_raw: (B, channels, time)
        x_stft: (B, channels, freq, time)  
        x_spatial_temporal: (B, H, W) or (B, H, W, T)
        """
        # 1. Raw EEG 처리 (EEG 특화 Transformer)
        x_raw = self.raw_norm(x_raw)
        x_raw_t = x_raw.transpose(1, 2)  # (B, time, channels)
        raw_features = self.raw_transformer(x_raw_t)
        
        # 2. STFT 처리 (공유 CNN + 경량 Swin)
        stft_features = self.stft_processor(x_stft)
        
        # 3. Spatial 처리 (초경량 2D CNN)
        spatial_features = None
        if self.add_spatial_temporal and x_spatial_temporal is not None:
            spatial_features = self.spatial_temporal_branch(x_spatial_temporal)
        
        # 4. 🎯 Simple Concat Fusion
        fused_features = self.fusion(raw_features, stft_features, spatial_features)
        
        # 5. 분류
        logits = self.classifier(fused_features)
        
        # Temperature scaling
        temperature = 1.05  # 더 낮게 조정
        logits = logits / temperature
        
        return logits, fused_features


# ========== 호환성을 위한 래퍼들 ==========
class TBEEGNet(TBEEGNet):
    """호환성을 위한 래퍼"""
    pass


class BalancedOptimizedEEGNet(TBEEGNet):
    """호환성을 위한 래퍼"""
    pass


# 기타 호환성 클래스들
class LightweightTransformer(EEGTransformerEncoder):
    """호환성을 위한 래퍼"""
    def __init__(self, input_dim, hidden_dim, n_heads=4, n_layers=2, max_len=1000):
        super().__init__(input_dim, hidden_dim, n_heads, n_layers, 
                        local_window=32, conv_kernel_size=3, dropout=0.1)


class EfficientCrossModalFusion(SimpleConcatFusion):
    """호환성을 위한 래퍼"""
    def __init__(self, raw_dim, stft_dim, spatial_dim, output_dim, n_heads=4):
        super().__init__(raw_dim, stft_dim, spatial_dim, output_dim, dropout=0.2)


class EnhancedCrossModalFusion(SimpleConcatFusion):
    """호환성을 위한 래퍼"""
    def __init__(self, dim1, dim2, fusion_dim, dropout=0.3):
        super().__init__(dim1, dim2, 0, fusion_dim, dropout)


class Simple2DCNN(SimpleSpatialCNN):
    """호환성을 위한 래퍼"""
    def __init__(self, grid_shape=(5, 6), out_dim=64):
        super().__init__(grid_shape, out_dim=out_dim)


class SimpleRawEEGTransformer(EEGTransformerEncoder):
    """호환성을 위한 래퍼"""
    def __init__(self, input_dim, n_channels, hidden_dim, seq_length=1000,
                 n_heads=4, n_layers=2, dropout=0.1, fs=128):
        super().__init__(input_dim, hidden_dim, n_heads, n_layers, 
                        local_window=32, conv_kernel_size=3, dropout=dropout)