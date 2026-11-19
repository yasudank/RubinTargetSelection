"""
Postage Stamp Dataset & Multi-view Contrastive Pair Mining
Handles FITS loading, caching, and pairing records sharing the same diaObjectId across bands and epochs.
"""

import os
import hashlib
import time
from math import floor
import random
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from astropy.io import fits
import psycopg2
from sqlalchemy import create_engine, text
from tqdm import tqdm

from .config import DBConfig, StepConfig, FITS_ROOT, DATA_DIR
from .augmentations import normalize_stamp, AstronomicalAugmentation, crop_or_pad_stamp

def get_cutout_path(fits_root: str, dia_object_id: int, mjd: float, band: str, dia_source_id: int, image_type: str) -> str:
    """FITSカットアウトのファイルパスを生成"""
    day_obs = int(floor(mjd - 0.5))
    try:
        sub_dir = f"{int(dia_object_id) % 1000:03d}"
    except (ValueError, TypeError):
        sub_dir = str(dia_object_id)[-3:].zfill(3)
    return os.path.join(fits_root, sub_dir, str(dia_object_id), str(day_obs), str(band), str(dia_source_id), f"{image_type}.fits")

def load_catalog_from_db(
    db_config: DBConfig, 
    limit: Optional[int] = None, 
    dedup_daily_band: bool = True
) -> pd.DataFrame:
    """
    cosmos_alert DB の diaSource テーブルから必要なメタデータを高速取得
    dedup_daily_band=True (デフォルト):
        同一天体 (diaObjectId)・同日 (day_obs)・同バンド (band) の複数アラートがある場合、
        その中で一番新しいもの (midpointMjdTai 最大) のみを採用する。
    """
    engine = create_engine(db_config.url)
    with engine.connect() as conn:
        conn.execute(text("SET work_mem = '2GB';"))
        if dedup_daily_band:
            query = """
                SELECT DISTINCT ON ("diaObjectId", floor("midpointMjdTai" - 0.5)::int, band)
                    "diaSourceId", "diaObjectId", "midpointMjdTai", "band", "ra", "dec"
                FROM "diaSource"
                WHERE "diaObjectId" IS NOT NULL
                ORDER BY "diaObjectId", floor("midpointMjdTai" - 0.5)::int, band, "midpointMjdTai" DESC
            """
        else:
            query = """
                SELECT "diaSourceId", "diaObjectId", "midpointMjdTai", "band", "ra", "dec"
                FROM "diaSource"
                WHERE "diaObjectId" IS NOT NULL
            """
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql_query(text(query), conn)
    return df

def load_source_list(step_config: StepConfig) -> Optional[np.ndarray]:
    """step_config.source_list_file があれば対象 diaSourceId の配列を返す (無指定なら None)"""
    path = getattr(step_config, "source_list_file", "")
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Source list {path} for step {step_config.step} not found. "
            f"Run --mode aggregate_objects (on step 1 predictions) first to select the input sources.")
    return pd.read_csv(path, usecols=["diaSourceId"])["diaSourceId"].astype(np.int64).values

def load_step_catalog(db_config: DBConfig, step_config: StepConfig, limit: Optional[int] = None) -> pd.DataFrame:
    """DB のカタログを読み込み、step_config.source_list_file があればそのソースだけに絞り込む"""
    sids = load_source_list(step_config)
    if sids is None:
        return load_catalog_from_db(db_config, limit=limit)
    df = load_catalog_from_db(db_config)
    df = df[df["diaSourceId"].astype(np.int64).isin(sids)].reset_index(drop=True)
    print(f"[*] Restricted catalog to {len(df)} sources listed in {step_config.source_list_file} "
          f"({len(sids)} listed)")
    if limit:
        df = df.iloc[:limit]
    return df

def stamp_cache_path(step_config: StepConfig, source_ids) -> str:
    """
    RAM キャッシュのファイル名。ソース一覧で絞り込む場合は、件数だけでなく diaSourceId 集合のハッシュも含める
    (step1 の既存キャッシュとの互換のため、絞り込み無しのときは従来の名前のまま)。
    """
    N = len(source_ids)
    name = f"stamps_step{step_config.step}_N{N}"
    if getattr(step_config, "source_list_file", ""):
        h = hashlib.md5(np.asarray(source_ids, dtype=np.int64).tobytes()).hexdigest()[:10]
        name += f"_{h}"
    return os.path.join(DATA_DIR, "cache", name + ".npy")

class PostageStampDataset(Dataset):
    """
    LSST Postage Stamp 画像用 PyTorch Dataset
    - mode='pair': SimCLR 用の正例ペア (x1, x2) を生成
      同一 diaObjectId の別観測 (別バンド・別日) を正例としてマイニング
    - mode='single': 推論・特徴抽出用 (x, metadata)
    """
    def __init__(
        self,
        df_records: pd.DataFrame,
        step_config: StepConfig,
        fits_root: str = FITS_ROOT,
        mode: str = "pair",
        same_object_prob: float = 0.6,
        labeled_data: Optional[Dict[int, int]] = None,  # diaSourceId -> class_idx
        cache_in_ram: bool = True,
        augment_labeled: bool = False
    ):
        super().__init__()
        self.step_config = step_config
        self.fits_root = fits_root
        self.mode = mode
        self.same_object_prob = same_object_prob
        self.aug = AstronomicalAugmentation()
        self.labeled_data = labeled_data or {}
        self.augment_labeled = augment_labeled
        
        # パスとメタデータの構築 (メモリ上で即座に構築: 0.1秒)
        records = []
        for row in df_records.itertuples():
            paths = [
                get_cutout_path(fits_root, row.diaObjectId, row.midpointMjdTai, row.band, row.diaSourceId, itype)
                for itype in step_config.image_types
            ]
            records.append({
                "diaSourceId": int(row.diaSourceId),
                "diaObjectId": int(row.diaObjectId),
                "midpointMjdTai": float(row.midpointMjdTai),
                "day_obs": int(floor(float(row.midpointMjdTai) - 0.5)),
                "band": str(row.band),
                "ra": float(row.ra) if hasattr(row, "ra") else 0.0,
                "dec": float(row.dec) if hasattr(row, "dec") else 0.0,
                "paths": paths
            })
        self.records = records
        
        # 同一 diaObjectId のインデックス一覧を作成 (天文学的マルチビュー正例マイニング用)
        self.object_to_indices: Dict[int, List[int]] = {}
        for idx, rec in enumerate(self.records):
            obj_id = rec["diaObjectId"]
            if obj_id not in self.object_to_indices:
                self.object_to_indices[obj_id] = []
            self.object_to_indices[obj_id].append(idx)

        # メインメモリ (RAM) キャッシュの初期化
        self.cache_in_ram = cache_in_ram
        self.cached_images = None
        if self.cache_in_ram and len(self.records) > 0:
            self._init_ram_cache()

    def _init_ram_cache(self):
        """
        全カットアウト画像をメインメモリ (RAM) に事前ロード。
        NFS の個別小ファイル読み込みボトルネックを完全に解消し、
        GPU 使用率を連続 90〜100% で維持可能にする。
        """
        N = len(self.records)
        C = self.step_config.in_channels
        cache_dir = os.path.join(DATA_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = stamp_cache_path(self.step_config, [r["diaSourceId"] for r in self.records])

        # 既存の事前作成キャッシュファイルが存在する場合は 0.5 秒で高速ロード
        if os.path.exists(cache_path):
            print(f"Loading {N} stamps directly from local RAM cache file: {cache_path}...")
            try:
                self.cached_images = np.load(cache_path)
                if self.cached_images.shape == (N, C, 30, 30):
                    mem_mb = self.cached_images.nbytes / (1024**2)
                    print(f"[+] Instantly loaded {N} stamps into RAM ({mem_mb:.1f} MB).")
                    return
            except Exception as e:
                print(f"Cache load error ({e}), re-caching from FITS files...")

        print(f"Pre-loading {N} stamps into RAM using multi-threaded I/O...")
        t0 = time.time()
        self.cached_images = np.zeros((N, C, 30, 30), dtype=np.float32)

        def load_one(i):
            self.cached_images[i] = self._read_image(self.records[i]["paths"])

        # 16並列スレッドでマルチスレッド高速ロード (20コアCPU向けにヘッドルームを確保)
        with ThreadPoolExecutor(max_workers=16) as executor:
            list(tqdm(executor.map(load_one, range(N)), total=N, desc="RAM Caching"))

        elapsed = time.time() - t0
        mem_mb = self.cached_images.nbytes / (1024**2)
        print(f"[+] Cached {N} stamps in RAM ({mem_mb:.1f} MB) in {elapsed:.1f}s.")

        # 次回実行時のゼロ待機化のために保存
        try:
            np.save(cache_path, self.cached_images)
            print(f"[*] Saved RAM cache file to {cache_path} for instant loading next time.")
        except Exception as e:
            print(f"Notice: Failed to save cache file: {e}")

    def __len__(self) -> int:
        return len(self.records)

    def _read_image(self, paths: List[str]) -> np.ndarray:
        """FITS画像群を読み込み、(C, 30, 30) の正規化済み配列を生成"""
        channels = []
        for p in paths:
            try:
                with fits.open(p) as hdul:
                    data = hdul[0].data if hdul[0].data is not None else hdul[1].data
                    if data is None or data.ndim != 2:
                        data = np.zeros((30, 30), dtype=np.float32)
                    else:
                        data = crop_or_pad_stamp(data, 30)
                        data = normalize_stamp(data)
            except Exception:
                data = np.zeros((30, 30), dtype=np.float32)
            channels.append(data)
        
        # (C, 30, 30)
        img = np.stack(channels, axis=0)
        return img

    def __getitem__(self, idx: int):
        rec1 = self.records[idx]
        if self.cached_images is not None:
            img1 = self.cached_images[idx]
        else:
            img1 = self._read_image(rec1["paths"])
        tensor1 = torch.from_numpy(img1).float()

        if self.mode == "pair":
            obj_id = rec1["diaObjectId"]
            candidates = self.object_to_indices.get(obj_id, [idx])
            other_candidates = [c for c in candidates if c != idx]
            
            # 同一 diaObjectId の別エポック (別観測日) / 別バンドを正例として選択
            # (※同一天体・同日・同バンドの重複はカタログ読込時に除外済みのため、
            #  候補は必ず別バンドまたは別日となります。従来の優先度3分は同一アラートサンプリングへ移行)
            if other_candidates and random.random() < self.same_object_prob:
                band1 = rec1["band"]
                day1 = rec1.get("day_obs", 0)
                
                # 優先度1: 異なるバンドの観測 (カラー不変性)
                diff_band = [c for c in other_candidates if self.records[c]["band"] != band1]
                # 優先度2: 異なる観測日の観測 (長期時間変化・大気シーイング不変性を学習)
                diff_day = [c for c in other_candidates if self.records[c].get("day_obs", 0) != day1]

                # 優先度1 (別バンド) と 優先度2 (別観測日) からサンプリング
                r = random.random()
                if diff_band and (not diff_day or r < 0.60):
                    other_idx = random.choice(diff_band)
                elif diff_day:
                    other_idx = random.choice(diff_day)
                elif diff_band:
                    other_idx = random.choice(diff_band)
                else:
                    other_idx = None

                if other_idx is not None:
                    if self.cached_images is not None:
                        img2 = self.cached_images[other_idx]
                    else:
                        rec2 = self.records[other_idx]
                        img2 = self._read_image(rec2["paths"])
                    tensor2 = torch.from_numpy(img2).float()
                else:
                    tensor2 = tensor1.clone()
            else:
                tensor2 = tensor1.clone()

            # データ拡張の適用
            view1 = self.aug(tensor1)
            view2 = self.aug(tensor2)
            return view1, view2

        elif self.mode == "single":
            # 推論・潜在特徴抽出用
            return tensor1, rec1

        elif self.mode == "labeled":
            # 教師あり / 能動学習微調整用
            label = self.labeled_data.get(rec1["diaSourceId"], -1)
            if self.augment_labeled:
                view1 = self.aug(tensor1)
                return view1, label, rec1
            return tensor1, label, rec1

        elif self.mode == "labeled_joint":
            # SupCon + CrossEntropy 同時学習用: 2つの独立データ拡張ビューとラベルを返却
            label = self.labeled_data.get(rec1["diaSourceId"], -1)
            view1 = self.aug(tensor1)
            view2 = self.aug(tensor1.clone())
            return view1, view2, label, rec1

        raise ValueError(f"Unknown dataset mode: {self.mode}")
