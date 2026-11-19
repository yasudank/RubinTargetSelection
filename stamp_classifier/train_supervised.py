"""
Supervised Full Training Module for Postage Stamp Classifier
Enables end-to-end full training (from scratch or fine-tuned from SimCLR weights)
using manual human labels (human_labels_{step}.csv).
Supports Cross-Entropy classification, Supervised Contrastive (SupCon), or Joint training.
"""

import os
import time
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .config import (
    STEP1_CONFIG, STEP2_CONFIG, SupervisedTrainConfig,
    DBConfig, FITS_ROOT, MODELS_DIR
)
from .dataset import load_catalog_from_db, PostageStampDataset
from .models import StampClassifier
from .losses import SupConLoss
from .labeling_viewer import load_human_labels

def plot_supervised_history(history_df: pd.DataFrame, out_png: str, step_name: str):
    """教師あり学習の損失・精度推移プロットを保存"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Loss (Train vs Val)
    ax1 = axes[0]
    ax1.plot(history_df['epoch'], history_df['train_loss'], marker='o', label='Train Loss', color='#1f77b4', linewidth=2)
    if 'val_loss' in history_df.columns:
        ax1.plot(history_df['epoch'], history_df['val_loss'], marker='s', label='Val Loss', color='#d62728', linewidth=2)
    min_val_row = history_df.loc[history_df['val_loss'].idxmin()] if 'val_loss' in history_df.columns else None
    if min_val_row is not None:
        ax1.scatter([min_val_row['epoch']], [min_val_row['val_loss']], color='gold', s=150, zorder=5, edgecolor='black',
                    label=f"Best Val: {min_val_row['val_loss']:.4f} (Ep {int(min_val_row['epoch'])})")
    ax1.set_title(f'Loss Curve ({step_name})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 2. Accuracy & Macro F1 (Val)
    ax2 = axes[1]
    if 'train_acc' in history_df.columns:
        ax2.plot(history_df['epoch'], history_df['train_acc'], marker='o', label='Train Acc', color='#1f77b4', alpha=0.6)
    if 'val_acc' in history_df.columns:
        ax2.plot(history_df['epoch'], history_df['val_acc'], marker='s', label='Val Acc', color='#2ca02c', linewidth=2)
    if 'val_f1' in history_df.columns:
        ax2.plot(history_df['epoch'], history_df['val_f1'], marker='^', label='Val Macro F1', color='#ff7f0e', linewidth=2)
    ax2.set_title('Validation Performance', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[+] Saved supervised learning curves to: {out_png}")

def plot_confusion_matrix(cm: np.ndarray, classes: list, out_png: str):
    """混同行列のヒートマップ描画"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Validation Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[+] Saved confusion matrix to: {out_png}")

def train_supervised(
    step: int = 1,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    from_scratch: bool = False,
    loss_type: str = "joint",
    val_split: float = 0.2
) -> Dict:
    """
    手動ラベリングデータ (human_labels_{step}.csv) を用いたフル学習を実行。

    Args:
        step: 1 (difference_real_bogus) または 2 (host_classification)
        epochs: フル学習エポック数 (デフォルト 50)
        batch_size: ミニバッチサイズ (デフォルト 64)
        lr: 初期学習率 (デフォルト 1e-3)
        from_scratch: True の場合は重みをスクラッチ初期化、False の場合は事前学習 SimCLR 重みから全層学習
        loss_type: 'joint' (CrossEntropy + SupCon 同時学習), 'ce' (CrossEntropy のみ), 'supcon' (SupCon のみ)
        val_split: 検証データの割合 (デフォルト 0.2)
    """
    step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  Supervised Full Training: Step {step} ({step_config.name})")
    print(f"  Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"  Mode: {'From Scratch' if from_scratch else 'From Pre-trained SimCLR Backbone'}")
    print(f"  Loss Type: {loss_type.upper()}")
    print(f"=======================================================\n")

    # 1. 手動ラベルのロード
    df_labels = load_human_labels(step_config.labels_file)
    if df_labels.empty or len(df_labels) < 10:
        raise ValueError(
            f"Insufficient human labels in {step_config.labels_file} (found {len(df_labels)} labels, need at least 10)."
        )

    # クラス名 -> インデックスの整合性チェック
    class_to_idx = {c: i for i, c in enumerate(step_config.classes)}
    valid_labels = df_labels[df_labels["class_name"].isin(step_config.classes)].copy()
    if len(valid_labels) < len(df_labels):
        print(f"[*] Filtered {len(df_labels) - len(valid_labels)} labels not in {step_config.classes}")

    if len(valid_labels["class_name"].unique()) < 2:
        raise ValueError(
            f"At least 2 distinct classes are required for training. Found: {valid_labels['class_name'].unique().tolist()}"
        )

    print(f"Loaded {len(valid_labels)} valid labeled stamps:")
    print(valid_labels["class_name"].value_counts().to_string())

    # 2. カタログから FITS パス等メタデータを取得
    db_cfg = DBConfig()
    print("Loading catalog records for labeled stamps...")
    df_cat = load_catalog_from_db(db_cfg, dedup_daily_band=False)
    labeled_cat = df_cat[df_cat["diaSourceId"].isin(valid_labels["diaSourceId"])].copy()

    # ラベル情報のマージ
    label_map = dict(zip(valid_labels["diaSourceId"], valid_labels["class_name"].map(class_to_idx)))
    labeled_cat["label_idx"] = labeled_cat["diaSourceId"].map(label_map)
    labeled_cat = labeled_cat.dropna(subset=["label_idx"])
    labeled_cat["label_idx"] = labeled_cat["label_idx"].astype(int)

    print(f"Matched {len(labeled_cat)} stamps in database catalog.")

    # 3. Stratified Train / Validation Split
    train_cat, val_cat = train_test_split(
        labeled_cat,
        test_size=val_split,
        random_state=42,
        stratify=labeled_cat["label_idx"]
    )
    print(f"Dataset split: Train = {len(train_cat)} stamps | Validation = {len(val_cat)} stamps")

    # クラス不均衡対策の重み算出 (Inverse Frequency)
    counts = train_cat["label_idx"].value_counts().sort_index()
    total_train = len(train_cat)
    class_weights = np.ones(len(step_config.classes), dtype=np.float32)
    for c_idx in range(len(step_config.classes)):
        if c_idx in counts:
            class_weights[c_idx] = total_train / (len(step_config.classes) * counts[c_idx])
    class_weights_tensor = torch.from_numpy(class_weights).float().to(device)
    print(f"Class Weights for Loss: {np.round(class_weights, 3)}")

    # 4. PyTorch Dataset & DataLoader
    # 訓練用: データ拡張 ON (joint 学習時は 2 つの拡張ビューを生成)
    train_mode = "labeled_joint" if loss_type == "joint" else "labeled"
    train_dataset = PostageStampDataset(
        df_records=train_cat,
        step_config=step_config,
        fits_root=FITS_ROOT,
        mode=train_mode,
        labeled_data=label_map,
        cache_in_ram=False,
        augment_labeled=True
    )
    # 検証用: データ拡張 OFF (未加工正規化画像で安定評価)
    val_dataset = PostageStampDataset(
        df_records=val_cat,
        step_config=step_config,
        fits_root=FITS_ROOT,
        mode="labeled",
        labeled_data=label_map,
        cache_in_ram=False,
        augment_labeled=False
    )

    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False, drop_last=False)

    # 5. モデル構築 & 事前学習重みロード (必要に応じて)
    model = StampClassifier(
        in_channels=step_config.in_channels,
        latent_dim=128,
        num_classes=len(step_config.classes),
        proj_dim=64
    ).to(device)

    ckpt_simclr = os.path.join(step_config.checkpoint_dir, "best_simclr_encoder.pt")
    if not from_scratch and os.path.exists(ckpt_simclr):
        model.load_from_simclr_checkpoint(ckpt_simclr, device)
    elif from_scratch:
        print("[*] Initializing model parameters from scratch (no pre-trained weights).")
    else:
        print(f"[*] Pre-trained checkpoint {ckpt_simclr} not found. Training from scratch.")

    # 6. オプティマイザ & 損失関数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    supcon_loss_fn = SupConLoss(temperature=0.07)

    # 7. 学習ループ
    history = []
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(step_config.checkpoint_dir, "best_supervised_model.pt")
    os.makedirs(step_config.checkpoint_dir, exist_ok=True)

    print(f"\nStarting {epochs} epochs of supervised training...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if loss_type == "joint":
                x1, x2, y, _ = batch
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                
                logits1, z1, _ = model.forward_with_projections(x1)
                logits2, z2, _ = model.forward_with_projections(x2)
                
                loss_ce = 0.5 * (ce_loss_fn(logits1, y) + ce_loss_fn(logits2, y))
                z_all = torch.cat([z1, z2], dim=0)
                y_all = torch.cat([y, y], dim=0)
                loss_supcon = supcon_loss_fn(z_all, y_all)
                loss = loss_ce + 0.5 * loss_supcon

                preds = torch.argmax(logits1, dim=1)
                train_correct += (preds == y).sum().item()
                train_samples += y.size(0)

            elif loss_type == "ce":
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = ce_loss_fn(logits, y)

                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == y).sum().item()
                train_samples += y.size(0)

            elif loss_type == "supcon":
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                logits, z, _ = model.forward_with_projections(x)
                loss = supcon_loss_fn(z, y)

                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == y).sum().item()
                train_samples += y.size(0)

            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * y.size(0)

        scheduler.step()
        train_loss = train_loss_total / max(train_samples, 1)
        train_acc = train_correct / max(train_samples, 1)

        # 検証ループ
        model.eval()
        val_loss_total = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = ce_loss_fn(logits, y)
                val_loss_total += loss.item() * y.size(0)

                preds = torch.argmax(logits, dim=1)
                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(y.cpu().numpy())

        val_loss = val_loss_total / max(len(val_cat), 1)
        val_acc = np.mean(np.array(val_preds_list) == np.array(val_targets_list))
        val_f1 = f1_score(val_targets_list, val_preds_list, average='macro', zero_division=0)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} Macro-F1: {val_f1:.3f}")

        # ベストモデルの保存 (Val Macro F1 基準)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "classifier_state_dict": model.classifier.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "classes": step_config.classes,
                "is_supervised": True
            }, best_checkpoint_path)

            # 下流特徴量抽出・Random Forest 用に best_simclr_encoder.pt も同期更新
            torch.save({
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "finetuned": True,
                "supervised_trained": True
            }, ckpt_simclr)

    elapsed = time.time() - t_start
    print(f"\n[+] Supervised training finished in {elapsed:.1f}s.")
    print(f"[+] Best Validation Macro F1: {best_val_f1:.4f} (Saved to {best_checkpoint_path})")
    print(f"[+] Encoder checkpoint synchronized at: {ckpt_simclr}")

    # 8. 学習履歴とプロット保存
    df_history = pd.DataFrame(history)
    csv_history = os.path.join(step_config.checkpoint_dir, "supervised_training_history.csv")
    df_history.to_csv(csv_history, index=False)

    plot_png = os.path.join(step_config.checkpoint_dir, "supervised_training_curves.png")
    plot_supervised_history(df_history, plot_png, step_config.name)

    # 9. ベストモデルでの最終検証レポート & 混同行列
    best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    final_preds, final_targets = [], []
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device)
            logits = model(x)
            final_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            final_targets.extend(y.numpy())

    print("\n=== Validation Set Final Classification Report ===")
    print(classification_report(
        final_targets,
        final_preds,
        target_names=step_config.classes,
        zero_division=0
    ))

    cm = confusion_matrix(final_targets, final_preds, labels=list(range(len(step_config.classes))))
    cm_png = os.path.join(step_config.checkpoint_dir, "confusion_matrix_supervised.png")
    plot_confusion_matrix(cm, step_config.classes, cm_png)

    return {
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "history": df_history,
        "checkpoint": best_checkpoint_path
    }
