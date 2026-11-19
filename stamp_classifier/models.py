"""
Neural Network Architectures for 30x30 Postage Stamp SimCLR
Includes compact Residual Encoder tailored for astronomy cutouts and SimCLR Projection Head.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)

class StampEncoder(nn.Module):
    """
    30x30 天体画像に特化した軽量 ResNet エンコーダ
    ImageNet用ResNetと異なり、入力初期の急激なダウンサンプリングを行わず、
    30x30 の微細な天体プロファイル（PSF、中心輝度、淡い銀河構造）を確実に捉えます。
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 128):
        super().__init__()
        # 30x30 を維持する初期畳み込み
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 30x30 -> 30x30
        self.layer1 = ResBlock(32, 64, stride=1)
        # 30x30 -> 15x15
        self.layer2 = ResBlock(64, 128, stride=2)
        # 15x15 -> 8x8
        self.layer3 = ResBlock(128, 256, stride=2)
        
        # 8x8 -> 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.init_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        latent = self.fc(out)
        return latent

class ProjectionHead(nn.Module):
    """
    SimCLR 射影ヘッド (MLP)
    潜在空間表現 h (128次元) を対照学習空間 z (64次元) に射影
    """
    def __init__(self, in_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        return F.normalize(z, dim=-1)

class SimCLRModel(nn.Module):
    """
    SimCLR 統合モデル (エンコーダ + 射影ヘッド)
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.encoder = StampEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.projector = ProjectionHead(in_dim=latent_dim, out_dim=proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """対照学習時のフォワード (L2正規化された z を返す)"""
        h = self.encoder(x)
        z = self.projector(h)
        return z

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """下流タスク (Random Forest) 用の潜在表現 h (128次元) を抽出"""
        return self.encoder(x)

class StampClassifier(nn.Module):
    """
    30x30 天体画像用 教師あり分類器 (StampEncoder + MLP 分類ヘッド + 射影ヘッド)
    手動ラベリングデータを用いたフル学習 (End-to-End Supervised Training) に使用。
    Cross-Entropy による直接クラス分類と、SupCon による表現学習の双方に対応。
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 128, num_classes: int = 3, proj_dim: int = 64):
        super().__init__()
        self.encoder = StampEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.projector = ProjectionHead(in_dim=latent_dim, out_dim=proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """標準的な分類フォワード (logits を出力)"""
        h = self.encoder(x)
        logits = self.classifier(h)
        return logits

    def forward_with_projections(self, x: torch.Tensor):
        """SupCon + CrossEntropy 同時学習 (Joint Training) 用"""
        h = self.encoder(x)
        logits = self.classifier(h)
        z = self.projector(h)
        return logits, z, h

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """潜在表現 h (128次元) を抽出"""
        return self.encoder(x)

    def load_from_simclr_checkpoint(self, ckpt_path: str, device: torch.device):
        """SimCLR の事前学習済み重みをロード"""
        if not ckpt_path or not os.path.exists(ckpt_path):
            return
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "encoder_state_dict" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
            print(f"[+] Loaded encoder weights from {ckpt_path}")
        elif "model_state_dict" in ckpt:
            enc_keys = {k.replace("encoder.", ""): v for k, v in ckpt["model_state_dict"].items() if k.startswith("encoder.")}
            if enc_keys:
                self.encoder.load_state_dict(enc_keys, strict=False)
                print(f"[+] Loaded encoder weights from {ckpt_path}")
        if "model_state_dict" in ckpt:
            proj_keys = {k.replace("projector.", ""): v for k, v in ckpt["model_state_dict"].items() if k.startswith("projector.")}
            if proj_keys:
                try:
                    self.projector.load_state_dict(proj_keys, strict=False)
                except Exception:
                    pass

