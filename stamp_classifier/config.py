"""
Postage Stamp Classifier Configuration
Defines hyperparameters, database parameters, class labels, and directory paths for Step 1 and Step 2.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class DBConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "cosmos_alert"
    user: str = "lsst"
    password: str = "rubin"

    @property
    def url(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class StepConfig:
    step: int                           # 1 or 2
    name: str                           # Description
    classes: List[str]                  # Target classification classes
    image_types: List[str]              # ['difference'] or ['science', 'template']
    in_channels: int                    # 1 for difference, 2 for science+template
    checkpoint_dir: str
    embeddings_file: str
    labels_file: str
    eval_labels_file: str = ""          # ランダム抽出の固定評価用ラベル (学習には一切使わない)
    source_list_file: str = ""          # 空でなければ、この CSV の diaSourceId に含まれるソースだけを対象にする (step2 の入力の絞り込み)

# ステップ 1: Difference 画像を用いた突発天体リアル・ボガス判定
STEP1_CONFIG = StepConfig(
    step=1,
    name="difference_real_bogus",
    classes=["positive_real", "negative_real", "bogus"],
    image_types=["difference"],
    in_channels=1,
    checkpoint_dir="models/step1",
    embeddings_file="data/embeddings_step1.npz",
    labels_file="data/human_labels_step1.csv",
    eval_labels_file="data/eval_labels_step1.csv"
)

# ステップ 2: Science & Template 画像を用いた母天体分類
STEP2_CONFIG = StepConfig(
    step=2,
    name="host_classification",
    classes=["star", "galaxy", "no_host", "bogus"],
    image_types=["science", "template"],
    in_channels=2,
    checkpoint_dir="models/step2",
    embeddings_file="data/embeddings_step2.npz",
    labels_file="data/human_labels_step2.csv",
    eval_labels_file="data/eval_labels_step2.csv",
    # step1 で real / uncertain と判定された天体のソース (aggregate_objects で作成)
    source_list_file="data/step2_sources.csv"
)

@dataclass
class SimCLRConfig:
    latent_dim: int = 128               # エンコーダの潜在特徴量次元数
    proj_dim: int = 64                  # 射影ヘッドの出力次元数
    temperature: float = 0.1            # NT-Xent 損失の温度パラメータ
    batch_size: int = 256               # バッチサイズ
    lr: float = 1e-3                    # 学習率 (AdamW)
    weight_decay: float = 1e-4          # Weight decay
    epochs: int = 25                    # 事前学習エポック数
    num_workers: int = 4                # DataLoader ワーカー数
    pair_same_object_prob: float = 0.6  # 同一 diaObjectId の別観測 (別バンド/別日) をサンプリングする確率 (同一アラートは 40%)
    use_labels: bool = True             # 手動ラベルが存在する場合、半教師あり対照学習 (SupCon) を同時に行うか
    sup_weight: float = 1.0             # 教師あり対照損失 (SupCon) の重み係数 lambda
    labeled_batch_size: int = 64        # ラベル付きバッチサイズ
    supcon_mode: str = "mixed"          # 'mixed' (ラベル付きを未ラベルバッチに混ぜて単一の半教師あり対照損失) または 'separate' (従来: NT-Xent + ラベル付きのみの SupCon)
    supcon_temperature: float = 0.07    # 'separate' モードの SupCon 温度 ('mixed' は temperature を共用)
    balanced_labeled: bool = True       # ラベル付きバッチをクラス均衡サンプリングするか
    holdout_frac: float = 0.2           # 評価用に取り置くラベルの割合 (diaObjectId 単位で分割, 0 で無効)
    select_metric: str = "holdout_rf_bacc"  # ベストモデル選択指標 (holdout が無い場合は loss)

@dataclass
class RFConfig:
    n_estimators: int = 250             # ランダムフォレストの決定木数
    max_depth: int = None               # 最大深度
    min_samples_split: int = 2          # 分割に必要な最小サンプル数
    class_weight: str = "balanced"      # 不均衡データ対策
    n_jobs: int = -1                    # 並列スレッド数
    random_state: int = 42

@dataclass
class ActiveLearningConfig:
    batch_size_per_iter: int = 20       # 1 サイクルあたりに目視判定するサンプル数
    uncertainty_method: str = "margin"  # 'margin' または 'entropy'
    # 1 サイクルの選択枠の配分 (fink 方式に近づけ、境界事例だけに偏らないようにする)
    uncertainty_ratio: float = 0.5      # 不確実性 (境界付近)
    exploit_ratio: float = 0.15         # real (非 bogus) 確率が高いもの
    anomaly_ratio: float = 0.15         # IsolationForest の異常度が高いもの
    random_ratio: float = 0.1           # 一様ランダム (ラベル分布の偏りを抑える)
    diversity_ratio: float = 0.1        # クラスタリングによる多様性サンプリングの割合 (残り枠)
    eval_set_size: int = 150            # ランダム抽出の固定評価用セットの目標件数
    focus_classes: Optional[List[str]] = None  # 2 クラスを指定すると、その境界に集中して候補を選ぶ (例 ['galaxy', 'star'])
    exclude_objects_file: str = ""      # この CSV に載っている天体 (diaObjectId / objId / obj_id 列) を候補から除く
    supcon_epochs: int = 5              # 目視フィードバック後のエンコーダ微調整エポック数
    supcon_lr: float = 1e-4             # 微調整時の学習率
    feedback_mode: str = "fine_tune"    # 'fine_tune' (高速5エポック微調整), 'full_train' (フル学習), 'rf_only' (エンコーダ固定・RF 再学習のみ)
    full_train_epochs: int = 30         # フル学習時のエポック数

@dataclass
class SupervisedTrainConfig:
    epochs: int = 50                    # フル学習エポック数
    batch_size: int = 64                # バッチサイズ
    lr: float = 1e-3                    # 初期学習率 (AdamW)
    weight_decay: float = 1e-4          # Weight decay
    loss_type: str = "joint"            # 'joint' (SupCon + CE), 'ce' (CrossEntropyのみ), 'supcon' (SupConのみ)
    from_scratch: bool = False          # True: 重みをランダム初期化してスクラッチ学習, False: SimCLR事前学習重みから全層学習
    val_split: float = 0.2              # 検証データの割合
    augment: bool = True                # 天体データ拡張 (D4回転/反転/微小シフト) の適用

FITS_ROOT = "cutouts"
DATA_DIR = "data"
MODELS_DIR = "models"

