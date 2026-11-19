"""
Feature Extractor
Extracts 128-dimensional latent representations for all postage stamps using the trained SimCLR encoder.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DBConfig, STEP1_CONFIG, STEP2_CONFIG, SimCLRConfig, FITS_ROOT, DATA_DIR
from .dataset import load_catalog_from_db, load_step_catalog, stamp_cache_path, PostageStampDataset
from .models import StampEncoder

def extract_embeddings(step: int = 1, batch_size: int = 512, limit: int = None, ckpt_path: str = None) -> str:
    step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DATA_DIR, exist_ok=True)

    if not ckpt_path:
        ckpt_path = os.path.join(step_config.checkpoint_dir, "best_simclr_encoder.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run train_simclr first.")

    print(f"=== Extracting Embeddings for Step {step} ===")
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # 1. エンコーダの復元
    encoder = StampEncoder(in_channels=step_config.in_channels, latent_dim=128)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder_state = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", {}))
    # 'encoder.' プレフィックスの調整
    clean_state = {}
    for k, v in encoder_state.items():
        clean_k = k.replace("encoder.", "")
        if clean_k in encoder.state_dict():
            clean_state[clean_k] = v
    encoder.load_state_dict(clean_state, strict=False)
    encoder.to(device)
    encoder.eval()

    # 2. カタログの取得とキャッシュ読み込み
    print("Loading catalog records from database...")
    db_cfg = DBConfig()
    df_cat = load_step_catalog(db_cfg, step_config, limit=limit)
    print(f"Loaded {len(df_cat)} records from database.")

    print("Initializing dataset and loading RAM cache...")
    dataset = PostageStampDataset(
        df_records=df_cat,
        step_config=step_config,
        fits_root=FITS_ROOT,
        mode="single"
    )
    print(f"Dataset ready. Starting GPU feature extraction for {len(dataset)} stamps...")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )

    all_embeddings = []
    all_source_ids = []
    all_object_ids = []
    all_mjds = []
    all_bands = []

    with torch.no_grad():
        for batch_x, rec_batch in tqdm(dataloader, desc="Extracting"):
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                feats = encoder(batch_x)
            
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))
            all_source_ids.extend(rec_batch["diaSourceId"].numpy() if isinstance(rec_batch["diaSourceId"], torch.Tensor) else rec_batch["diaSourceId"])
            all_object_ids.extend(rec_batch["diaObjectId"].numpy() if isinstance(rec_batch["diaObjectId"], torch.Tensor) else rec_batch["diaObjectId"])
            all_mjds.extend(rec_batch["midpointMjdTai"].numpy() if isinstance(rec_batch["midpointMjdTai"], torch.Tensor) else rec_batch["midpointMjdTai"])
            all_bands.extend(rec_batch["band"])

    embeddings = np.concatenate(all_embeddings, axis=0)

    # 3. 物理的形態・構造パラメータの抽出
    from .morphology import extract_morphological_features_batch, MORPH_FEATURE_NAMES
    n_morph = len(MORPH_FEATURE_NAMES)
    print(f"Extracting physical morphological parameters ({n_morph} dims) from cutouts...")
    if dataset.cached_images is not None:
        morph_features = extract_morphological_features_batch(dataset.cached_images)
    else:
        morph_features = np.zeros((len(embeddings), n_morph), dtype=np.float32)

    # 4. ハイブリッド特徴量の結合 (128-d SimCLR + Morph)
    combined_features = np.concatenate([embeddings, morph_features], axis=1)

    out_file = step_config.embeddings_file
    np.savez_compressed(
        out_file,
        embeddings=embeddings,
        morph_features=morph_features,
        combined_features=combined_features,
        diaSourceIds=np.array(all_source_ids, dtype=np.int64),
        diaObjectIds=np.array(all_object_ids, dtype=np.int64),
        midpointMjdTai=np.array(all_mjds, dtype=np.float64),
        bands=np.array(all_bands)
    )

    print(f"[+] Successfully extracted {len(embeddings)} hybrid features ({combined_features.shape[1]} dims) -> Saved to {out_file}")
    return out_file

def load_cached_embeddings(step: int = 1, use_hybrid: bool = True):
    step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
    if not os.path.exists(step_config.embeddings_file):
        raise FileNotFoundError(f"Embeddings cache not found at {step_config.embeddings_file}. Run extract_embeddings first.")
    data = np.load(step_config.embeddings_file, allow_pickle=True)
    
    # ハイブリッド特徴量 (136次元) を優先
    if use_hybrid and "combined_features" in data:
        embs = data["combined_features"]
    elif use_hybrid and "morph_features" in data:
        embs = np.concatenate([data["embeddings"], data["morph_features"]], axis=1)
    else:
        embs = data["embeddings"]

    return {
        "embeddings": embs,
        "simclr_embeddings": data["embeddings"],
        "morph_features": data["morph_features"] if "morph_features" in data else None,
        "diaSourceIds": data["diaSourceIds"],
        "diaObjectIds": data["diaObjectIds"],
        "midpointMjdTai": data["midpointMjdTai"],
        "bands": data["bands"]
    }

def append_morphology_to_cache(step: int = 1) -> str:
    """
    既存の embeddings.npz に、RAM キャッシュされたカットアウトから計算した
    8 次元の形態パラメータを瞬時に結合して保存（GPU再推論不要・約5秒で完了）。
    """
    step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
    if not os.path.exists(step_config.embeddings_file):
        raise FileNotFoundError(f"Embeddings cache not found at {step_config.embeddings_file}. Run extract_embeddings first.")

    print(f"Loading existing embeddings from {step_config.embeddings_file}...")
    data = np.load(step_config.embeddings_file, allow_pickle=True)
    embeddings = data["embeddings"]
    sids = data["diaSourceIds"]
    oids = data["diaObjectIds"]
    mjds = data["midpointMjdTai"]
    bands = data["bands"]

    cache_dir = os.path.join(DATA_DIR, "cache")
    stamps_cache = stamp_cache_path(step_config, sids)
    if not os.path.exists(stamps_cache):
        print(f"RAM stamps cache not found at {stamps_cache}. Running full extraction...")
        return extract_embeddings(step=step)

    print(f"Loading cutouts from RAM cache: {stamps_cache}...")
    stamps = np.load(stamps_cache, mmap_mode="r")

    from .morphology import extract_morphological_features_batch, MORPH_FEATURE_NAMES
    n_morph = len(MORPH_FEATURE_NAMES)
    print(f"Computing {n_morph}-dimensional morphological features for {len(stamps)} stamps...")
    morph_features = extract_morphological_features_batch(stamps)

    combined_features = np.concatenate([embeddings, morph_features], axis=1)

    np.savez_compressed(
        step_config.embeddings_file,
        embeddings=embeddings,
        morph_features=morph_features,
        combined_features=combined_features,
        diaSourceIds=sids,
        diaObjectIds=oids,
        midpointMjdTai=mjds,
        bands=bands
    )
    print(f"[+] Successfully merged morphology! Total features: {combined_features.shape[1]} dims (128-d SimCLR + {n_morph}-d Morphology).")
    return step_config.embeddings_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Latent Embeddings using SimCLR Encoder")
    parser.add_argument("--step", type=int, default=1, choices=[1, 2], help="Step 1 or Step 2")
    parser.add_argument("--batch_size", type=int, default=512, help="Inference batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records")
    parser.add_argument("--morph_only", action="store_true", help="Append morphology without GPU inference")
    args = parser.parse_args()

    if args.morph_only:
        append_morphology_to_cache(step=args.step)
    else:
        extract_embeddings(step=args.step, batch_size=args.batch_size, limit=args.limit)
