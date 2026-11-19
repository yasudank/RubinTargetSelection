"""
Astronomical Image Augmentations and Preprocessing
Provides physically meaningful data transformations and robust astronomical normalization.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def crop_or_pad_stamp(arr: np.ndarray, target_size: int = 30) -> np.ndarray:
    """
    32x32 や 31x31 など中心ピクセル周辺のカットアウトを中心に揃えて (target_size, target_size) に整形。
    """
    if arr is None or arr.ndim != 2:
        return np.zeros((target_size, target_size), dtype=np.float32)
    H, W = arr.shape
    if H == target_size and W == target_size:
        return arr.astype(np.float32)

    cy, cx = H // 2, W // 2
    half = target_size // 2

    y1 = max(0, cy - half)
    y2 = min(H, cy + half)
    x1 = max(0, cx - half)
    x2 = min(W, cx + half)

    sub = arr[y1:y2, x1:x2]
    out = np.zeros((target_size, target_size), dtype=np.float32)
    sy = (target_size - sub.shape[0]) // 2
    sx = (target_size - sub.shape[1]) // 2
    out[sy:sy + sub.shape[0], sx:sx + sub.shape[1]] = sub
    return out

def normalize_stamp(arr: np.ndarray, clip_sigma: float = 10.0) -> np.ndarray:
    """
    天文学的 FITS 配列のロバスト正規化 (asinh ストレッチ & スケーリング)
    NaN / Inf の補正、中央値減算、IQR/MAD に基づく分散正規化を実施。
    """
    # NaN / Inf を有限値に置換
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2次元または3次元配列
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    scale = mad * 1.4826 if mad > 1e-6 else (np.std(arr) + 1e-6)
    
    norm = (arr - med) / scale
    
    # asinh (Inverse Hyperbolic Sine) ストレッチで高輝度中心と淡い広がりを両立
    norm = np.arcsinh(norm / 3.0)
    
    if clip_sigma is not None:
        norm = np.clip(norm, -clip_sigma, clip_sigma)
        
    return norm.astype(np.float32)

class AstronomicalAugmentation(nn.Module):
    """
    天体画像向け PyTorch データ拡張モジュール
    - D4 群（0, 90, 180, 270° 回転 + 上下左右反転）: 天文学的に完全な対称性
    - サブピクセル・シフト / 微小平行移動 (±2 ピクセル)
    - 観測シーイング変動を模した微小ガウシアンブラー
    - スカイノイズゆらぎを模した微小ガウスノイズ
    """
    def __init__(self, p_flip: float = 0.5, p_shift: float = 0.5, p_noise: float = 0.5, p_blur: float = 0.3):
        super().__init__()
        self.p_flip = p_flip
        self.p_shift = p_shift
        self.p_noise = p_noise
        self.p_blur = p_blur

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W) or (C, H, W)
        """
        is_batched = (x.ndim == 4)
        if not is_batched:
            x = x.unsqueeze(0)
            
        B, C, H, W = x.shape
        out = x.clone()
        
        # 1. ランダム回転 (0, 90, 180, 270 度)
        k = random.randint(0, 3)
        if k > 0:
            out = torch.rot90(out, k, dims=[-2, -1])
            
        # 2. ランダム水平反転
        if random.random() < self.p_flip:
            out = torch.flip(out, dims=[-1])
            
        # 3. ランダム垂直反転
        if random.random() < self.p_flip:
            out = torch.flip(out, dims=[-2])
            
        # 4. 微小平行移動 (Shift [-2, 2] px)
        if random.random() < self.p_shift:
            shift_x = random.randint(-2, 2)
            shift_y = random.randint(-2, 2)
            if shift_x != 0 or shift_y != 0:
                out = torch.roll(out, shifts=(shift_y, shift_x), dims=[-2, -1])
                
        # 5. スカイノイズ注入
        if random.random() < self.p_noise:
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(out) * noise_std
            out = out + noise

        # 6. 微小ガウシアンブラー (シーイング変動の模擬)
        if random.random() < self.p_blur:
            sigma = random.uniform(0.3, 0.8)
            kernel_size = 3
            radius = kernel_size // 2
            x_coord = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
            k1d = torch.exp(-0.5 * (x_coord / sigma) ** 2)
            k1d = k1d / k1d.sum()
            k2d = torch.outer(k1d, k1d).expand(C, 1, kernel_size, kernel_size)
            out = F.conv2d(out, k2d, padding=radius, groups=C)
            
        if not is_batched:
            out = out.squeeze(0)
            
        return out
