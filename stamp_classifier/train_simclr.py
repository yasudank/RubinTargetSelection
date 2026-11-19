"""
SimCLR Pre-training & Semi-Supervised Contrastive Learning Script
Supports both:
1. Pure self-supervised SimCLR (NT-Xent loss) on unlabeled catalog (~628k stamps)
2. Semi-Supervised Simultaneous Contrastive Learning (SimCLR on unlabeled + SupCon on human labels)
   aligning the 128-d latent manifold into distinct orthogonal axes for each class.
"""

import os
import time
import argparse
import hashlib
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import DBConfig, STEP1_CONFIG, STEP2_CONFIG, SimCLRConfig, FITS_ROOT, MODELS_DIR
from .dataset import load_catalog_from_db, load_step_catalog, PostageStampDataset
from .models import SimCLRModel
from .losses import NTXentLoss, SupConLoss, SemiSupConLoss
from .labeling_viewer import load_human_labels

def is_holdout_object(dia_object_id: int, frac: float) -> bool:
    """diaObjectId のハッシュで決まる安定な holdout 判定 (ラベルが増えても既存の割り当ては変わらない)"""
    if frac <= 0:
        return False
    h = int(hashlib.md5(str(int(dia_object_id)).encode()).hexdigest()[:8], 16)
    return (h % 1000) < int(frac * 1000)

def load_stamp_array(dataset: PostageStampDataset) -> np.ndarray:
    """Dataset の全レコードの正規化済み画像を (N, C, 30, 30) で読み込む (ラベル付き少数用)"""
    with ThreadPoolExecutor(max_workers=16) as executor:
        imgs = list(executor.map(lambda r: dataset._read_image(r["paths"]), dataset.records))
    return np.stack(imgs, axis=0).astype(np.float32)

@torch.no_grad()
def embed_array(model, images: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    feats = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i:i + batch_size]).to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            feats.append(model.get_latent(x).float().cpu().numpy())
    return np.concatenate(feats, axis=0)

def evaluate_holdout(model, X_tr: np.ndarray, y_tr: np.ndarray, X_ho: np.ndarray, y_ho: np.ndarray, device) -> Dict[str, float]:
    """
    エンコーダ出力 h (128次元) で学習用ラベルに kNN / RF を当てはめ、holdout ラベルで精度を測る。
    クラス不均衡のため balanced accuracy も記録。
    """
    was_training = model.training
    model.eval()
    h_tr = embed_array(model, X_tr, device)
    h_ho = embed_array(model, X_ho, device)
    if was_training:
        model.train()

    knn = KNeighborsClassifier(n_neighbors=min(10, len(h_tr)), metric="cosine", weights="distance")
    p_knn = knn.fit(h_tr, y_tr).predict(h_ho)
    rf = RandomForestClassifier(n_estimators=250, class_weight="balanced", n_jobs=-1, random_state=42)
    p_rf = rf.fit(h_tr, y_tr).predict(h_ho)
    return {
        "holdout_knn_acc": accuracy_score(y_ho, p_knn),
        "holdout_knn_bacc": balanced_accuracy_score(y_ho, p_knn),
        "holdout_rf_acc": accuracy_score(y_ho, p_rf),
        "holdout_rf_bacc": balanced_accuracy_score(y_ho, p_rf),
    }

def save_training_plots(history_df: pd.DataFrame, out_png: str, step_name: str):
    """損失曲線と学習率スケジュール (+ holdout 精度) のプロット"""
    has_holdout = 'holdout_rf_bacc' in history_df.columns and history_df['holdout_rf_bacc'].notna().any()
    fig, axes = plt.subplots(1, 3 if has_holdout else 2, figsize=(19 if has_holdout else 13, 4.5))
    
    # 1. Loss curve
    ax1 = axes[0]
    ax1.plot(history_df['epoch'], history_df['loss'], marker='o', color='#1f77b4', linewidth=2, label='Total Loss')
    if 'simclr_loss' in history_df.columns and 'supcon_loss' in history_df.columns:
        ax1.plot(history_df['epoch'], history_df['simclr_loss'], linestyle='--', color='#2ca02c', label='SimCLR Loss (Unlabeled)')
        ax1.plot(history_df['epoch'], history_df['supcon_loss'], linestyle=':', color='#d62728', linewidth=2, label='SupCon Loss (Labeled)')

    min_loss_row = history_df.loc[history_df['loss'].idxmin()]
    ax1.scatter(min_loss_row['epoch'], min_loss_row['loss'], color='red', s=100, zorder=5, 
                label=f"Best: {min_loss_row['loss']:.4f} (Ep {int(min_loss_row['epoch'])})")
    ax1.set_title(f'Contrastive Loss Curve ({step_name})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 2. Learning rate & Time per epoch
    ax2 = axes[1]
    ax2.plot(history_df['epoch'], history_df['lr'], marker='s', color='#2ca02c', linewidth=2, label='Learning Rate')
    ax2.set_title('Learning Rate & Epoch Duration', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate', color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 副軸: Epoch Duration (Seconds)
    ax2_right = ax2.twinx()
    ax2_right.bar(history_df['epoch'], history_df['epoch_sec'], alpha=0.25, color='gray', width=0.4, label='Duration (s)')
    ax2_right.set_ylabel('Epoch Time (s)', color='gray')
    ax2_right.tick_params(axis='y', labelcolor='gray')

    # 3. Holdout ラベルでの汎化精度
    if has_holdout:
        ax3 = axes[2]
        for col, style, label in [('holdout_rf_bacc', '-o', 'RF balanced acc'), ('holdout_rf_acc', '--', 'RF acc'),
                                  ('holdout_knn_bacc', '-s', 'kNN balanced acc'), ('holdout_knn_acc', ':', 'kNN acc')]:
            if col in history_df.columns:
                ax3.plot(history_df['epoch'], history_df[col], style, label=label)
        ax3.set_title('Holdout Label Accuracy (encoder h)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[+] Saved training curves plot to: {out_png}")

def save_latent_space_visualization(
    model,
    dataset,
    device,
    out_png: str,
    step_name: str,
    df_labels: Optional[pd.DataFrame] = None,
    labeled_dataset: Optional[PostageStampDataset] = None,
    n_samples: int = 2000
):
    """
    潜在表現空間の 2D プロジェクション (PCA) 可視化
    左図: クラスラベル（手動ラベル天体の3軸整列状況）
    右図: 観測パスバンド (u, g, r, i, z, y)
    """
    try:
        model.eval()
        feats = []
        bands = []
        labels_list = []

        label_dict = {}
        if df_labels is not None and not df_labels.empty:
            label_dict = dict(zip(df_labels["diaSourceId"], df_labels["class_name"]))

        with torch.no_grad():
            # 1. まず手動ラベル付きサンプルを全件確実に抽出して可視化に含める
            if labeled_dataset is not None:
                for idx in range(len(labeled_dataset)):
                    rec = labeled_dataset.records[idx]
                    img = labeled_dataset._read_image(rec["paths"])
                    t = torch.from_numpy(img).float().unsqueeze(0).to(device)
                    feat = model.get_latent(t).squeeze(0).cpu().numpy()
                    feats.append(feat)
                    bands.append(rec["band"])
                    labels_list.append(label_dict.get(rec["diaSourceId"], "Unlabeled"))

            # 2. 未ラベル天体をサンプリングして背景に配置
            n = min(n_samples, len(dataset))
            for i in range(n):
                rec = dataset.records[i]
                if rec["diaSourceId"] in label_dict and labeled_dataset is not None:
                    continue  # すでに登録済み
                if dataset.cached_images is not None:
                    img = dataset.cached_images[i]
                else:
                    img = dataset._read_image(rec["paths"])
                t = torch.from_numpy(img).float().unsqueeze(0).to(device)
                feat = model.get_latent(t).squeeze(0).cpu().numpy()
                feats.append(feat)
                bands.append(rec["band"])
                labels_list.append(label_dict.get(rec["diaSourceId"], "Unlabeled"))

        feats = np.array(feats)
        # 対照学習の超球面表現に基づき、極端なノルムを持つ外れ値によるスケーリング歪みを防ぐため L2 正規化を実施
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats_norm = feats / np.maximum(norms, 1e-9)

        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(feats_norm)

        has_classes = any(l != "Unlabeled" for l in labels_list)
        if has_classes:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            ax_cls = axes[0]
            ax_band = axes[1]
        else:
            fig, ax_band = plt.subplots(figsize=(8, 6.5))
            ax_cls = None

        # 1. クラスラベル別のプロット (3軸の幾何学的整列を可視化)
        if ax_cls is not None:
            # 未ラベルを背景に薄く描画
            unlab_mask = [l == "Unlabeled" for l in labels_list]
            ax_cls.scatter(proj[unlab_mask, 0], proj[unlab_mask, 1], c="#d3d3d3", s=10, alpha=0.35, label="Unlabeled")

            class_palette = {
                "positive_real": "#1f77b4",  # Blue
                "negative_real": "#d62728",  # Red
                "bogus": "#ff7f0e",          # Orange
                "star": "#2ca02c",
                "galaxy": "#9467bd",
                "no_host": "#7f7f7f"
            }
            # 各クラスのラベル付きサンプルを描画
            unique_classes = [c for c in sorted(list(set(labels_list))) if c != "Unlabeled"]
            for c in unique_classes:
                c_mask = [l == c for l in labels_list]
                c_color = class_palette.get(c, "#333333")
                ax_cls.scatter(proj[c_mask, 0], proj[c_mask, 1], c=c_color, s=40, edgecolors="black", linewidths=0.5, alpha=0.9, label=f"{c} (N={sum(c_mask)})")

            ax_cls.set_title(f'Latent Space PCA: Class Alignment ({step_name})\nExplained Var: {pca.explained_variance_ratio_.sum()*100:.1f}%', fontsize=12, fontweight='bold')
            ax_cls.set_xlabel('Principal Component 1')
            ax_cls.set_ylabel('Principal Component 2')
            ax_cls.grid(True, linestyle='--', alpha=0.5)
            ax_cls.legend(title='Class Label', loc='upper right')

        # 2. パスバンド別のプロット
        unique_bands = sorted(list(set(bands)))
        band_colors = {'u': '#9467bd', 'g': '#2ca02c', 'r': '#d62728', 'i': '#ff7f0e', 'z': '#8c564b', 'y': '#e377c2'}
        for b in unique_bands:
            mask = [curr_b == b for curr_b in bands]
            color = band_colors.get(b, None)
            ax_band.scatter(proj[mask, 0], proj[mask, 1], label=f"Band {b}", alpha=0.5, s=18, c=color)

        ax_band.set_title(f'Latent Space PCA: Passband Distribution ({step_name})', fontsize=12, fontweight='bold')
        ax_band.set_xlabel('Principal Component 1')
        ax_band.set_ylabel('Principal Component 2')
        ax_band.grid(True, linestyle='--', alpha=0.5)
        ax_band.legend(title='Passband', loc='upper right')

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)

        # プレゼンテーション・アニメーション用履歴ディレクトリにも初期フレームを保存
        history_dir = os.path.join(os.path.dirname(out_png), "pca_history")
        os.makedirs(history_dir, exist_ok=True)
        init_frame = os.path.join(history_dir, "cycle_000_pretrain.png")
        plt.savefig(init_frame, dpi=150)
        plt.close(fig)
        print(f"[+] Saved latent space PCA visualization to: {out_png}")
    except Exception as e:
        print(f"Notice: Latent space visualization skipped: {e}")

def train_simclr(
    step: int = 1,
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    limit: int = None,
    use_labels: bool = True,
    sup_weight: float = None,
    supcon_mode: str = None,
    holdout_frac: float = None
) -> str:
    """
    SimCLR 自己教師あり学習および半教師あり同時対照学習を実行。

    Args:
        step: 1 (difference_real_bogus) または 2 (host_classification)
        epochs: 学習エポック数
        batch_size: 未ラベルデータのバッチサイズ
        lr: 学習率 (AdamW)
        limit: デバッグ用レコード制限数
        use_labels: 手動ラベルが存在する場合、各ステップで教師あり対照損失 (SupCon) を同時に学習するか
        sup_weight: 教師あり対照損失の重み係数 lambda (デフォルト: 1.0)
        supcon_mode: 'mixed' (ラベル付きを未ラベルバッチに混ぜる) または 'separate' (従来方式)
        holdout_frac: 評価用に取り置くラベルの割合 (diaObjectId 単位)。学習には使わず、各エポックの汎化精度とベスト選択に使う
    """
    step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
    sim_cfg = SimCLRConfig()
    if epochs: sim_cfg.epochs = epochs
    if batch_size: sim_cfg.batch_size = batch_size
    if lr: sim_cfg.lr = lr
    if sup_weight is not None: sim_cfg.sup_weight = sup_weight
    if use_labels is not None: sim_cfg.use_labels = use_labels
    if supcon_mode is not None: sim_cfg.supcon_mode = supcon_mode
    if holdout_frac is not None: sim_cfg.holdout_frac = holdout_frac

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  Contrastive Learning: Step {step} ({step_config.name})")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Channels: {step_config.in_channels} ({', '.join(step_config.image_types)})")
    print(f"  Epochs: {sim_cfg.epochs} | Batch Size: {sim_cfg.batch_size}")
    print(f"  Semi-Supervised Mode: {'ACTIVE (' + sim_cfg.supcon_mode + ')' if sim_cfg.use_labels else 'OFF (Pure SimCLR)'}")
    print(f"  Holdout Fraction: {sim_cfg.holdout_frac}")
    print(f"=======================================================\n")

    os.makedirs(step_config.checkpoint_dir, exist_ok=True)

    # 1. DBからカタログをロード
    print("Loading catalog records from cosmos_alert...")
    db_cfg = DBConfig()
    df_cat = load_step_catalog(db_cfg, step_config, limit=limit)
    print(f"Loaded {len(df_cat)} records from database.")

    # 2. 未ラベル Dataset と DataLoader の構築
    dataset = PostageStampDataset(
        df_records=df_cat,
        step_config=step_config,
        fits_root=FITS_ROOT,
        mode="pair",
        same_object_prob=sim_cfg.pair_same_object_prob
    )
    print(f"Unlabeled dataset prepared with {len(dataset)} valid postage stamp cutouts.")
    if len(dataset) == 0:
        raise RuntimeError("No valid postage stamp images found. Please verify cutouts directory path.")

    dataloader = DataLoader(
        dataset,
        batch_size=sim_cfg.batch_size,
        shuffle=True,
        num_workers=sim_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(sim_cfg.num_workers > 0),
        prefetch_factor=2 if sim_cfg.num_workers > 0 else None
    )

    # 3. 手動ラベルの読み込みと 学習用 / holdout 評価用 への分割
    #    holdout は use_labels に関わらず (純 SimCLR でも) 評価に使う
    labeled_loader = None
    labeled_dataset = None
    df_labels = None
    eval_data = None
    # ランダム抽出の固定評価用セットがあれば holdout として優先使用 (学習用ラベルは全件を学習に使う)
    df_eval = load_human_labels(step_config.eval_labels_file) if step_config.eval_labels_file else pd.DataFrame()
    use_eval_set = (not df_eval.empty and len(df_eval) >= 30 and df_eval["class_name"].nunique() >= 2)
    if use_eval_set:
        print(f"[*] Using random evaluation set as holdout: {len(df_eval)} labels from {step_config.eval_labels_file}")

    if os.path.exists(step_config.labels_file):
        df_labels = load_human_labels(step_config.labels_file)
        if use_eval_set:
            df_labels = pd.concat([df_labels[~df_labels["diaSourceId"].isin(df_eval["diaSourceId"])], df_eval], ignore_index=True)
        valid_labels = df_labels[df_labels["class_name"].isin(step_config.classes)].copy()
        if len(valid_labels) >= 10 and len(valid_labels["class_name"].unique()) >= 2:
            class_to_idx = {c: i for i, c in enumerate(step_config.classes)}
            label_map = dict(zip(valid_labels["diaSourceId"], valid_labels["class_name"].map(class_to_idx)))

            labeled_cat = df_cat[df_cat["diaSourceId"].isin(valid_labels["diaSourceId"])].copy()
            if len(labeled_cat) < len(valid_labels):
                # デバッグ用 limit 設定時や同日重複で除外された分も補完取得
                df_labeled_full = load_catalog_from_db(db_cfg, dedup_daily_band=False)
                labeled_cat = df_labeled_full[df_labeled_full["diaSourceId"].isin(valid_labels["diaSourceId"])].copy()

            if use_eval_set:
                ho_mask = labeled_cat["diaSourceId"].isin(df_eval["diaSourceId"])
                # 評価用セットと同一天体の学習用ラベルは除外 (同一天体の別観測によるリークを防ぐ)
                leak = labeled_cat["diaObjectId"].isin(labeled_cat.loc[ho_mask, "diaObjectId"]) & ~ho_mask
                if leak.any():
                    print(f"[*] Excluding {int(leak.sum())} training labels sharing a diaObjectId with the evaluation set.")
                    labeled_cat = labeled_cat[~leak]
                    ho_mask = ho_mask[~leak]
            else:
                ho_mask = labeled_cat["diaObjectId"].apply(lambda o: is_holdout_object(o, sim_cfg.holdout_frac))
            train_cat = labeled_cat[~ho_mask]
            ho_cat = labeled_cat[ho_mask]

            split_df = labeled_cat[["diaSourceId", "diaObjectId"]].copy()
            split_df["class_name"] = split_df["diaSourceId"].map(dict(zip(valid_labels["diaSourceId"], valid_labels["class_name"])))
            split_df["split"] = np.where(ho_mask, "holdout", "train")
            split_csv = os.path.join(step_config.checkpoint_dir, "label_split.csv")
            split_df.to_csv(split_csv, index=False)
            print(f"[*] Label split: train={len(train_cat)}, holdout={len(ho_cat)} (by diaObjectId) -> {split_csv}")
            print("    Train classes:  ", split_df.loc[~ho_mask.values, "class_name"].value_counts().to_dict())
            print("    Holdout classes:", split_df.loc[ho_mask.values, "class_name"].value_counts().to_dict())

            if len(train_cat) >= 10:
                labeled_dataset = PostageStampDataset(
                    df_records=train_cat,
                    step_config=step_config,
                    fits_root=FITS_ROOT,
                    mode="labeled_joint",
                    labeled_data=label_map,
                    cache_in_ram=False,
                    augment_labeled=True
                )
                # ラベル付き少数は一度だけ読み込んで保持 (毎バッチの NAS 読み込みを回避)
                print(f"Loading {len(labeled_dataset)} labeled training stamps into RAM...")
                labeled_dataset.cached_images = load_stamp_array(labeled_dataset)
                y_tr = np.array([label_map[r["diaSourceId"]] for r in labeled_dataset.records])

                if len(ho_cat) > 0 and len(np.unique(y_tr)) >= 2:
                    ho_dataset = PostageStampDataset(
                        df_records=ho_cat, step_config=step_config, fits_root=FITS_ROOT,
                        mode="single", cache_in_ram=False
                    )
                    print(f"Loading {len(ho_dataset)} holdout stamps into RAM...")
                    X_ho = load_stamp_array(ho_dataset)
                    y_ho = np.array([label_map[r["diaSourceId"]] for r in ho_dataset.records])
                    eval_data = (labeled_dataset.cached_images, y_tr, X_ho, y_ho)

                if sim_cfg.use_labels:
                    print(f"[+] Semi-Supervised Active: {len(labeled_dataset)} human-labeled training stamps.")
                    sampler = None
                    if sim_cfg.balanced_labeled:
                        counts = np.bincount(y_tr, minlength=len(step_config.classes)).astype(np.float64)
                        weights = 1.0 / counts[y_tr]
                        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(y_tr), replacement=True)
                    labeled_loader = DataLoader(
                        labeled_dataset,
                        batch_size=min(sim_cfg.labeled_batch_size, len(labeled_dataset)),
                        shuffle=(sampler is None),
                        sampler=sampler,
                        drop_last=True,
                        pin_memory=(device.type == "cuda")
                    )
            else:
                print(f"[*] Notice: Only {len(train_cat)} labeled training stamps found in catalog. Running pure SimCLR.")
        else:
            print("[*] Notice: Insufficient labels for SupCon (< 10 or < 2 classes). Running pure SimCLR.")

    use_holdout = eval_data is not None
    select_metric = sim_cfg.select_metric if use_holdout else "loss"
    print(f"[*] Best checkpoint selection metric: {select_metric}")

    # 4. モデルと最適化手法の準備
    model = SimCLRModel(
        in_channels=step_config.in_channels,
        latent_dim=sim_cfg.latent_dim,
        proj_dim=sim_cfg.proj_dim
    ).to(device)

    mixed = labeled_loader is not None and sim_cfg.supcon_mode == "mixed"
    criterion_ntxent = NTXentLoss(temperature=sim_cfg.temperature)
    criterion_semi = SemiSupConLoss(temperature=sim_cfg.temperature) if mixed else None
    criterion_supcon = SupConLoss(temperature=sim_cfg.supcon_temperature) if (labeled_loader is not None and not mixed) else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=sim_cfg.lr, weight_decay=sim_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sim_cfg.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # 5. 学習ループ
    best_score = -float("inf")
    ckpt_path = os.path.join(step_config.checkpoint_dir, "best_simclr_encoder.pt")
    last_ckpt_path = os.path.join(step_config.checkpoint_dir, "last_simclr_encoder.pt")
    history = []

    labeled_iter = itertools.cycle(labeled_loader) if labeled_loader is not None else None

    if use_holdout:
        m0 = evaluate_holdout(model, *eval_data, device)
        print("Epoch 00 (random init) holdout: " + ", ".join(f"{k}={v:.4f}" for k, v in m0.items()))

    for epoch in range(1, sim_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_simclr_loss = 0.0
        total_supcon_loss = 0.0
        num_batches = 0
        t0 = time.time()

        mode_str = f"Semi-Supervised[{sim_cfg.supcon_mode}]" if labeled_loader is not None else "SimCLR"
        pbar = tqdm(dataloader, desc=f"{mode_str} Ep {epoch:02d}/{sim_cfg.epochs}")
        for view1, view2 in pbar:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                if mixed:
                    # ラベル付きを未ラベルバッチに連結し、単一の半教師あり対照損失で学習
                    xl1, xl2, yl, _ = next(labeled_iter)
                    xl1 = xl1.to(device, non_blocking=True)
                    xl2 = xl2.to(device, non_blocking=True)
                    yl = yl.to(device, non_blocking=True).long()
                    labels = torch.cat([torch.full((view1.shape[0],), -1, dtype=torch.long, device=device), yl])

                    z1 = model(torch.cat([view1, xl1], dim=0))
                    z2 = model(torch.cat([view2, xl2], dim=0))
                    per_anchor, lab_mask = criterion_semi(z1, z2, labels)
                    loss_simclr = per_anchor[~lab_mask].mean()
                    loss_supcon = per_anchor[lab_mask].mean()
                    loss = loss_simclr + sim_cfg.sup_weight * loss_supcon
                else:
                    # 1. 未ラベルデータに対する自己教師あり対照学習 (NT-Xent)
                    z1 = model(view1)
                    z2 = model(view2)
                    loss_simclr = criterion_ntxent(z1, z2)

                    # 2. 手動ラベルデータに対する教師あり対照学習 (SupCon, 従来方式)
                    if labeled_iter is not None:
                        xl1, xl2, yl, _ = next(labeled_iter)
                        xl1 = xl1.to(device, non_blocking=True)
                        xl2 = xl2.to(device, non_blocking=True)
                        yl = yl.to(device, non_blocking=True)

                        zl1 = model(xl1)
                        zl2 = model(xl2)
                        zl_all = torch.cat([zl1, zl2], dim=0)
                        yl_all = torch.cat([yl, yl], dim=0)
                        loss_supcon = criterion_supcon(zl_all, yl_all)

                        loss = loss_simclr + sim_cfg.sup_weight * loss_supcon
                    else:
                        loss = loss_simclr
                        loss_supcon = torch.tensor(0.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_simclr_loss += loss_simclr.item()
            if labeled_loader is not None:
                total_supcon_loss += loss_supcon.item()
            num_batches += 1

            postfix = {"loss": f"{loss.item():.4f}", "simclr": f"{loss_simclr.item():.4f}"}
            if labeled_loader is not None:
                postfix["supcon"] = f"{loss_supcon.item():.4f}"
            pbar.set_postfix(postfix)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_simclr = total_simclr_loss / max(num_batches, 1)
        avg_supcon = (total_supcon_loss / max(num_batches, 1)) if labeled_loader is not None else 0.0
        elapsed = time.time() - t0

        log_str = f"Epoch {epoch:02d} Finished: Total Loss = {avg_loss:.4f} (SimCLR: {avg_simclr:.4f}"
        if labeled_loader is not None:
            log_str += f", SupCon: {avg_supcon:.4f}"
        log_str += f") [{elapsed:.1f}s, LR = {current_lr:.6f}]"
        print(log_str)

        entry = {
            "epoch": epoch,
            "loss": avg_loss,
            "simclr_loss": avg_simclr,
            "supcon_loss": avg_supcon,
            "lr": current_lr,
            "epoch_sec": elapsed
        }
        if use_holdout:
            metrics = evaluate_holdout(model, *eval_data, device)
            entry.update(metrics)
            print("    Holdout: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        history.append(entry)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "semi_supervised": (labeled_loader is not None),
            "supcon_mode": sim_cfg.supcon_mode if labeled_loader is not None else None,
            "holdout_metrics": {k: v for k, v in entry.items() if k.startswith("holdout_")},
            "select_metric": select_metric,
            "config": sim_cfg
        }
        torch.save(ckpt, last_ckpt_path)

        # チェックポイント保存 (holdout があれば汎化精度、無ければ loss で選択)
        score = entry[select_metric] if select_metric != "loss" else -avg_loss
        if score > best_score:
            best_score = score
            torch.save(ckpt, ckpt_path)
            print(f"[*] New best model ({select_metric} = {entry[select_metric]:.4f}) saved to {ckpt_path}")

    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"\n[+] Contrastive Training Complete! Best epoch: {best_ckpt['epoch']} ({select_metric} = {best_ckpt['loss'] if select_metric == 'loss' else best_ckpt['holdout_metrics'][select_metric]:.4f})")

    # 6. 学習推移プロットおよび潜在空間可視化の出力 (ベストモデルで描画)
    if history:
        df_hist = pd.DataFrame(history)
        hist_csv = os.path.join(step_config.checkpoint_dir, "training_history.csv")
        df_hist.to_csv(hist_csv, index=False)
        print(f"[*] Saved training history to {hist_csv}")

        loss_png = os.path.join(step_config.checkpoint_dir, "loss_curve.png")
        save_training_plots(df_hist, loss_png, step_config.name)

        model.load_state_dict(best_ckpt["model_state_dict"])
        pca_png = os.path.join(step_config.checkpoint_dir, "latent_space_pca.png")
        save_latent_space_visualization(model, dataset, device, pca_png, step_config.name, df_labels=df_labels, labeled_dataset=labeled_dataset)

    return ckpt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR & Semi-Supervised Contrastive Training for LSST Postage Stamps")
    parser.add_argument("--step", type=int, default=1, choices=[1, 2], help="Step 1 (Difference) or Step 2 (Science/Template)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of catalog records for quick test")
    parser.add_argument("--use_labels", action="store_true", default=True, help="Enable semi-supervised simultaneous training if labels exist")
    parser.add_argument("--no_labels", action="store_false", dest="use_labels", help="Disable semi-supervised training (pure SimCLR)")
    parser.add_argument("--sup_weight", type=float, default=1.0, help="Weight for SupCon loss")
    parser.add_argument("--supcon_mode", type=str, default=None, choices=["mixed", "separate"], help="'mixed' (labeled mixed into unlabeled batch) or 'separate' (legacy)")
    parser.add_argument("--holdout_frac", type=float, default=None, help="Fraction of labels held out for evaluation (by diaObjectId)")
    args = parser.parse_args()

    train_simclr(
        step=args.step,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit=args.limit,
        use_labels=args.use_labels,
        sup_weight=args.sup_weight,
        supcon_mode=args.supcon_mode,
        holdout_frac=args.holdout_frac
    )
