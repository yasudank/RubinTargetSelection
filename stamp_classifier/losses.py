"""
Contrastive Loss Functions for Self-Supervised and Supervised Learning
Includes NT-Xent (SimCLR) and Supervised Contrastive Loss (SupCon).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR)
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, D) L2-normalized projections for view 1
            z2: (B, D) L2-normalized projections for view 2
        """
        B = z1.shape[0]
        device = z1.device
        
        # 2B 個の表現を結合
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        
        # コサイン類似度行列 (2B, 2B)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # 自己類似度を除外するためのマスク (対角成分を極小値に設定)
        mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)
        
        # 正例ペアのターゲットインデックス:
        # z1[i] の正例は z2[i] (index: i + B)
        # z2[i] の正例は z1[i] (index: i)
        targets = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B, device=device)
        ], dim=0)
        
        loss = F.cross_entropy(sim_matrix, targets)
        return loss

class SemiSupConLoss(nn.Module):
    """
    半教師あり対照損失 (未ラベル + ラベル付きを同一バッチで扱う)
    - 全サンプル: 同一画像の augmentation ペアを正例 (未ラベルでは NT-Xent と等価)
    - ラベル付き同士: 同一クラスの全サンプル (両ビュー) も正例
    - 分母には未ラベルを含むバッチ全体が入るため、ラベル付き集合だけでクラスが一点に潰れる自明解を取りにくい
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            z1, z2: (N, D) 2 つのビューの射影
            labels: (N,) クラス ID, 未ラベルは -1
        Returns:
            per_anchor: (2N,) アンカーごとの損失
            labeled: (2N,) ラベル付きアンカーのマスク
        """
        with torch.autocast(device_type=z1.device.type, enabled=False):
            return self._forward_fp32(z1.float(), z2.float(), labels)

    def _forward_fp32(self, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor):
        N = z1.shape[0]
        device = z1.device
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
        lab = torch.cat([labels, labels], dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        self_mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, -1e9)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        idx = torch.arange(N, device=device)
        pos = torch.zeros(2 * N, 2 * N, dtype=torch.bool, device=device)
        pos[idx, idx + N] = True
        pos[idx + N, idx] = True
        labeled = lab >= 0
        same = (lab.unsqueeze(0) == lab.unsqueeze(1)) & labeled.unsqueeze(0) & labeled.unsqueeze(1)
        pos = (pos | same) & ~self_mask

        pos_f = pos.float()
        per_anchor = -(log_prob * pos_f).sum(1) / pos_f.sum(1)
        return per_anchor, labeled

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
    目視判定ラベルを持つデータに対して、同一クラスの潜在表現を近づけ、
    異なるクラスの潜在表現を遠ざける微調整用損失。
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (N, D) L2-normalized projections
            labels: (N,) Class labels (0, 1, ..., C-1), or -1 for unlabelled
        """
        device = z.device
        valid_mask = (labels >= 0)
        
        # ラベル付きサンプルのみ抽出
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        z = z[valid_mask]
        labels = labels[valid_mask]
        N = z.shape[0]
        
        # 同一クラスの正例マスク (N, N)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 自己ペアを除外
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(N).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 類似度行列
        sim = torch.matmul(z, z.T) / self.temperature
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        
        # 分母: 自己以外の全サンプルに対する exp(sim) の和
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # 分子: 正例ペアの平均対数確率
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss
