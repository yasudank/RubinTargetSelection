"""
Interactive Visual Inspection and Labeling Tool
Renders postage stamps with astronomical asinh stretching and records human judgments.
Works in both GUI (interactive matplotlib window) and terminal/headless (preview PNG + CLI input) modes.
"""

import os
import time
from math import floor
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits

from .config import StepConfig, STEP1_CONFIG, STEP2_CONFIG, FITS_ROOT, DATA_DIR
from .dataset import get_cutout_path
from .augmentations import normalize_stamp, crop_or_pad_stamp

# Science / Template の表示。両者で共通の表示範囲 [ST_VMIN_PERCENTILE, ST_VMAX_PERCENTILE] を使う。
#   asinh  (既定): 範囲を [0, 1] に規格化して arcsinh(ST_ASINH_SOFTENING * z) をかける。中心を飽和させずに淡い裾野も見える
#   linear       : 線形 (従来の表示)
# 環境変数で変更できる: STAMP_ST_STRETCH=linear|asinh, STAMP_ST_VMAX_PCT=99.5, STAMP_ST_ASINH_SOFTENING=10
ST_STRETCH = os.environ.get("STAMP_ST_STRETCH", "asinh").lower()
ST_VMIN_PERCENTILE = 1.0
ST_VMAX_PERCENTILE = float(os.environ.get("STAMP_ST_VMAX_PCT", "99.5" if ST_STRETCH == "asinh" else "97.0"))
ST_ASINH_SOFTENING = float(os.environ.get("STAMP_ST_ASINH_SOFTENING", "10"))

def science_template_norm(vmin: float, vmax: float):
    """Science / Template 用のカラー正規化 (カラーバーは元のピクセル値で表示される)"""
    from matplotlib.colors import FuncNorm, Normalize
    if ST_STRETCH != "asinh":
        return Normalize(vmin=vmin, vmax=vmax, clip=True)
    a, span = ST_ASINH_SOFTENING, vmax - vmin
    fwd = lambda x: np.arcsinh(a * (np.asarray(x, dtype=float) - vmin) / span)
    inv = lambda y: vmin + span * np.sinh(np.asarray(y, dtype=float)) / a
    return FuncNorm((fwd, inv), vmin=vmin, vmax=vmax, clip=True)

def cleanup_gui() -> None:
    """
    目視判定のあとで、GUI (TkAgg) の後始末をメインスレッドで行う。
    これをしないと、残った Tk のオブジェクトが RF の学習 (並列スレッド) 中にガベージコレクションされ、
    "RuntimeError: main thread is not in main loop" の警告が大量に出る (学習結果には影響しない)。
    """
    import gc
    plt.close("all")
    gc.collect()

def load_human_labels(labels_file: str) -> pd.DataFrame:
    """既存の目視ラベルCSVを読み込む"""
    if os.path.exists(labels_file):
        try:
            return pd.read_csv(labels_file)
        except Exception:
            pass
    return pd.DataFrame(columns=["diaSourceId", "diaObjectId", "label", "class_name", "timestamp"])

def save_human_label(labels_file: str, dia_source_id: int, dia_object_id: int, label: int, class_name: str,
                     selection: Optional[str] = None):
    """目視ラベルを追記保存 (selection: 能動サンプリングでの選択理由)"""
    df = load_human_labels(labels_file)
    # 既存の同一 diaSourceId があれば上書き
    df = df[df["diaSourceId"] != dia_source_id]
    row = {
        "diaSourceId": int(dia_source_id),
        "diaObjectId": int(dia_object_id),
        "label": int(label),
        "class_name": class_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    if selection is not None:
        row["selection"] = selection
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)
    df.to_csv(labels_file, index=False)

def read_stamp_fits(path: str) -> np.ndarray:
    """FITSファイルからNaN/Inf補正と30x30中心クロップ済みの2D画像配列を読み込む"""
    if os.path.exists(path):
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data if hdul[0].data is not None else hdul[1].data
                if data is not None and data.ndim == 2:
                    clean = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    return crop_or_pad_stamp(clean, 30)
        except Exception:
            pass
    return np.zeros((30, 30), dtype=np.float32)

class LabelingViewer:
    """
    目視判定用ビューア
    """
    def __init__(self, step: int = 1, fits_root: str = FITS_ROOT):
        self.step = step
        self.step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
        self.fits_root = fits_root
        self.classes = self.step_config.classes
        self.labels_file = self.step_config.labels_file

    def review_sample(self, rec: Dict, pred_class: Optional[str], confidence: Optional[float], margin: Optional[float],
                      interactive: bool = True, selection: Optional[str] = None,
                      allow_unsure: bool = False, title_note: Optional[str] = None) -> Optional[int]:
        """
        1 件のカットアウト画像を可視化し、ユーザーからの目視ラベル判定を受け付ける。
        pred_class=None の場合はモデル予測を表示しない (評価用セットのブラインド判定)。
        allow_unsure=True の場合は [u] Unsure (判定不能) を選択可能にし、label=-1, class_name="unsure" として保存する。
        """
        source_id = rec["diaSourceId"]
        obj_id = rec["diaObjectId"]
        mjd = rec["midpointMjdTai"]
        band = rec["band"]

        # 画像パスの取得
        diff_path = get_cutout_path(self.fits_root, obj_id, mjd, band, source_id, "difference")
        sci_path = get_cutout_path(self.fits_root, obj_id, mjd, band, source_id, "science")
        tpl_path = get_cutout_path(self.fits_root, obj_id, mjd, band, source_id, "template")

        diff_raw = read_stamp_fits(diff_path)
        sci_raw = read_stamp_fits(sci_path)
        tpl_raw = read_stamp_fits(tpl_path)

        # 真に空（欠損または全ピクセル0）のFITSファイルは自動スキップ
        if (np.all(diff_raw == 0) and np.all(sci_raw == 0)) or (diff_raw.std() < 1e-4 and sci_raw.std() < 1e-4):
            print(f"[!] diaSourceId {source_id}: FITS file is genuinely empty/missing. Skipping automatically.")
            return "SKIP_EMPTY"

        # 1. Science と Template の共通ピクセルストレッチ (同一フラックス範囲)
        comb_st = np.concatenate([sci_raw.ravel(), tpl_raw.ravel()])
        vmin_st = float(np.percentile(comb_st, ST_VMIN_PERCENTILE))
        vmax_st = float(np.percentile(comb_st, ST_VMAX_PERCENTILE))
        if vmax_st <= vmin_st:
            vmax_st = vmin_st + 1.0
        st_norm = science_template_norm(vmin_st, vmax_st)

        # 2. Difference の Linear (線形) 対称ストレッチ (ゼロ中心)
        diff_med = float(np.median(diff_raw))
        diff_mad = float(np.median(np.abs(diff_raw - diff_med)) * 1.4826)
        if diff_mad < 1e-6:
            diff_mad = float(np.std(diff_raw) + 1e-6)
        
        diff_centered = diff_raw - diff_med
        # 3σ〜外れ値99.5%タイルに基づき、過大/過小にならずに信号が最も鮮明に見える対称範囲を設定
        vlim_diff = max(3.0 * diff_mad, float(np.percentile(np.abs(diff_centered), 99.5)))
        vmin_diff = -vlim_diff
        vmax_diff = vlim_diff

        # プレビュー画像の生成 (3パネル: Science, Template, Difference)
        title_text = (
            f"Step {self.step}: diaSourceId={source_id} (obj={obj_id}, band={band}, MJD={mjd:.2f})\n"
            + (f"Model Prediction: [{pred_class}] (Conf: {confidence:.2f}, Margin: {margin:.2f})"
               if pred_class is not None else (title_note or "Blind review (evaluation set)"))
            + (f" | Selected by: {selection}" if selection and pred_class is not None else "")
            + (f"\n{title_note}" if title_note and pred_class is not None else "")
        )
        # タイトルの行数に応じて上余白を確保し、各パネルのタイトルと重ならないようにする
        n_title_lines = title_text.count("\n") + 1
        fig_h = 4.0 + 0.25 * n_title_lines
        fig, axes = plt.subplots(1, 3, figsize=(10, fig_h))
        fig.suptitle(title_text, fontsize=11, y=0.985, va="top")
        top_frac = 1.0 - (0.30 + 0.24 * n_title_lines) / fig_h

        im0 = axes[0].imshow(sci_raw, cmap="viridis", origin="lower", norm=st_norm)
        axes[0].set_title(f"Science ({ST_STRETCH})")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(tpl_raw, cmap="viridis", origin="lower", norm=st_norm)
        axes[1].set_title(f"Template ({ST_STRETCH})")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(diff_centered, cmap="coolwarm", origin="lower", vmin=vmin_diff, vmax=vmax_diff)
        axes[2].set_title("Difference (Linear)")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.subplots_adjust(bottom=0.22, top=top_frac, left=0.05, right=0.95)

        # 画像をプレビュー用キャッシュに保存
        preview_dir = os.path.join(DATA_DIR, f"previews_step{self.step}")
        os.makedirs(preview_dir, exist_ok=True)
        preview_path = os.path.join(preview_dir, f"stamp_{source_id}.png")
        current_preview_path = os.path.join(preview_dir, "current_stamp.png")
        plt.savefig(preview_path, dpi=120)
        plt.savefig(current_preview_path, dpi=120)

        # GUI / 端末プロンプトの切り替え (TkAgg等の対話型バックエンドかつDISPLAY設定時にGUI有効化)
        backend = matplotlib.get_backend().lower()
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        use_gui = interactive and has_display and (backend != "agg")

        user_choice = [None]

        def make_choice(val):
            user_choice[0] = str(val).lower()
            plt.close(fig)

        # GUIボタンの配置
        from matplotlib.widgets import Button
        btn_objects = []
        btn_items = [(f"[{i+1}] {c}", str(i+1)) for i, c in enumerate(self.classes)]
        if allow_unsure:
            btn_items.append(("[u] Unsure", "u"))
        btn_items.append(("[s] Skip", "s"))
        btn_items.append(("[q] Quit", "q"))

        n_btns = len(btn_items)
        spacing = 0.015
        total_width = 0.90
        btn_width = (total_width - (n_btns - 1) * spacing) / n_btns
        btn_y = 0.05
        btn_h = 0.09
        colors = ["#d0e1fd", "#d4edda", "#f8d7da", "#fff3cd", "#e2e3e5", "#f0f0f0"]

        for i, (blabel, bval) in enumerate(btn_items):
            bx = 0.05 + i * (btn_width + spacing)
            ax_b = fig.add_axes([bx, btn_y, btn_width, btn_h])
            c = colors[i % len(colors)]
            btn = Button(ax_b, blabel, color=c, hovercolor="#b0d0ff")
            btn.on_clicked(lambda event, v=bval: make_choice(v))
            btn_objects.append(btn)

        # キーボードショートカット (ウィンドウにフォーカスがある状態で 1,2,3,s,q を押すだけで判定可能)
        def on_key(event):
            if event.key:
                k = event.key.lower()
                valid_keys = [str(i+1) for i in range(len(self.classes))] + ["s", "q"] + (["u"] if allow_unsure else [])
                if k in valid_keys:
                    make_choice(k)
                elif k == "escape":
                    make_choice("q")

        fig.canvas.mpl_connect("key_press_event", on_key)

        def on_close(event):
            if user_choice[0] is None:
                user_choice[0] = "s"

        fig.canvas.mpl_connect("close_event", on_close)

        prompt_text = "Select class: " + ", ".join([f"[{i+1}] {c}" for i, c in enumerate(self.classes)]) + (" | [u] Unsure" if allow_unsure else "") + " | [s] Skip, [q] Quit: "

        if use_gui:
            print(f"\n[GUI Active] Reviewing diaSourceId {source_id} in desktop window.")
            print(" -> Click a button or press [1, 2, 3, s, q] on your keyboard to proceed.")
            plt.show()  # ブロッキングでウィンドウを確実に表示・描画イベントループを維持
            user_input = user_choice[0]
        else:
            print(f"\n[Preview Saved] {preview_path}")
            user_input = input(prompt_text).strip().lower()
            plt.close(fig)

        if user_input == "q":
            return "QUIT"
        elif user_input == "s" or not user_input:
            return None
        elif user_input == "u" and allow_unsure:
            save_human_label(self.labels_file, source_id, obj_id, -1, "unsure", selection=selection)
            print(f"[+] Recorded: diaSourceId {source_id} -> 'unsure'")
            return -1
        
        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(self.classes):
                cname = self.classes[idx]
                save_human_label(self.labels_file, source_id, obj_id, idx, cname, selection=selection)
                print(f"[+] Recorded: diaSourceId {source_id} -> '{cname}' (Class {idx})")
                return idx
        except ValueError:
            pass

        print("[-] Invalid input, skipped.")
        return None
