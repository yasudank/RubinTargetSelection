"""
Astronomical Morphological Feature Extractor
Extracts physical and structural shape parameters from 30x30 postage stamps
to robustly distinguish real astrophysical transients from subtraction artifacts (bogus).
"""

import numpy as np

# 定義済み定数（高速化のためモジュールレベルで事前生成）
GRID_Y, GRID_X = np.mgrid[0:30, 0:30].astype(np.float32)
CENTER_COORD = 14.5
GRID_DX = GRID_X - CENTER_COORD
GRID_DY = GRID_Y - CENTER_COORD
GRID_R2 = GRID_DX**2 + GRID_DY**2
CORE_MASK = GRID_R2 <= (2.5**2)
ANN_MASK = GRID_R2 <= (6.0**2)
BORDER_MASK = GRID_R2 > (10.0**2)

MORPH_FEATURE_NAMES = [
    # 既存の物理・構造パラメータ
    "centroid_dist",            # ピーク/重心の中心 (14.5, 14.5) からの距離
    "peak_snr",                 # ピーク S/N 比 (Max / MAD_noise)
    "dipole_ratio",             # ダイポール対称度: min(F+, F-) / (F+ + F-) (双極アーチファクトで約0.5)
    "dipole_separation",        # 正ピークと負ピークの空間距離 (画素単位)
    "concentration_ratio",      # 集中度: r < 2.5px フラックス / r < 6.0px フラックス
    "radial_dispersion",        # 全画素の2次中心モーメント半径 (FWHM/有効サイズ指標)
    "ellipticity",              # 全画素の楕円率・扁平度 (点源は ~0、回折スパイク/飛跡は >0.6)
    "border_flux_ratio",        # 境界部 (r > 10px) フラックス比率 (明るい星のリング残差検知)
    
    # 追加: 双極アーティファクトの幾何学的測定
    "sign_dominance_center",    # 中心部 (r <= 6px) 正負フラックス優勢度: (F+ - F-) / (F+ + F-) (+1:正, -1:負, ~0:双極)
    "dipole_antisymmetry",      # 180度回転反対称度: I(dx,dy) ≈ -I(-dx,-dy) の一致度 (双極で極大、点源で極小)
    
    # 追加: 低S/Nピクセル除外 (SNR > 3σ) のサイズ・広がり
    "radial_dispersion_masked", # 低S/Nピクセル除外後の有効広がり半径
    "a_masked",                 # 低S/N除外後の長軸半径 (semi-major axis)
    "b_masked",                 # 低S/N除外後の短軸半径 (semi-minor axis)
    "roundness_masked",         # 低S/N除外後の真円度 (b / a)
    "n_pix_masked"              # SNR >= 3σ のコア有意ピクセル数
]

def extract_morphological_features_single(stamp: np.ndarray) -> np.ndarray:
    """
    1 枚の 30x30 差分画像から 15 次元の形態・幾何構造パラメータベクトルを抽出。
    """
    if stamp.ndim != 2 or stamp.shape != (30, 30):
        return np.zeros(len(MORPH_FEATURE_NAMES), dtype=np.float32)

    x = GRID_X
    y = GRID_Y
    r2 = GRID_R2

    # 1. 背景ノイズと中央値
    med = np.median(stamp)
    mad = np.median(np.abs(stamp - med)) * 1.4826
    noise = mad if mad > 1e-6 else (np.std(stamp) + 1e-6)
    diff = stamp - med

    # 2. 正負フラックス
    pos_mask = diff > 0
    neg_mask = diff < 0
    pos_flux = float(np.sum(diff[pos_mask]))
    neg_flux = float(np.sum(np.abs(diff[neg_mask])))
    tot_abs_flux = pos_flux + neg_flux + 1e-6

    # 3. ピーク値と全画素重み付き重心
    abs_diff = np.abs(diff)
    sum_abs = float(np.sum(abs_diff)) + 1e-6
    xc = float(np.sum(x * abs_diff)) / sum_abs
    yc = float(np.sum(y * abs_diff)) / sum_abs
    centroid_dist = np.sqrt((xc - CENTER_COORD)**2 + (yc - CENTER_COORD)**2)

    # 4. ピーク S/N 比
    peak_val = float(np.max(abs_diff))
    peak_snr = peak_val / noise

    # 5. ダイポール対称比: min(F+, F-) / (F+ + F-)
    dipole_ratio = min(pos_flux, neg_flux) / tot_abs_flux

    # 6. 正ピークと負ピークの重心間距離
    if pos_flux > 0 and neg_flux > 0:
        pos_yc = np.sum(y[pos_mask] * diff[pos_mask]) / (pos_flux + 1e-6)
        pos_xc = np.sum(x[pos_mask] * diff[pos_mask]) / (pos_flux + 1e-6)
        neg_yc = np.sum(y[neg_mask] * np.abs(diff[neg_mask])) / (neg_flux + 1e-6)
        neg_xc = np.sum(x[neg_mask] * np.abs(diff[neg_mask])) / (neg_flux + 1e-6)
        dipole_separation = np.sqrt((pos_xc - neg_xc)**2 + (pos_yc - neg_yc)**2)
    else:
        dipole_separation = 0.0

    # 7. 集中度 (Concentration Ratio)
    core_flux = float(np.sum(abs_diff[CORE_MASK]))
    ann_flux = float(np.sum(abs_diff[ANN_MASK])) + 1e-6
    concentration = core_flux / ann_flux

    # 8. 全画素の 2次中心モーメントと楕円率
    dx_c = x - xc
    dy_c = y - yc
    mu_xx = float(np.sum(dx_c**2 * abs_diff)) / sum_abs
    mu_yy = float(np.sum(dy_c**2 * abs_diff)) / sum_abs
    mu_xy = float(np.sum(dx_c * dy_c * abs_diff)) / sum_abs

    radial_dispersion = np.sqrt(max(0.0, mu_xx + mu_yy))
    denom = mu_xx + mu_yy + 1e-6
    ellipticity = np.sqrt(max(0.0, (mu_xx - mu_yy)**2 + 4.0 * (mu_xy**2))) / denom
    ellipticity = min(1.0, float(ellipticity))

    # 9. 境界部フラックス比率 (r > 10px)
    border_flux = float(np.sum(abs_diff[BORDER_MASK]))
    border_flux_ratio = border_flux / sum_abs

    # --- 双極アーティファクトの幾何学的測定 ---
    # 10. 中心部 (r <= 6px) 正負フラックス優勢度: (F+ - F-) / (F+ + F-)
    c_pos_flux = float(np.sum(diff[ANN_MASK & pos_mask]))
    c_neg_flux = float(np.sum(np.abs(diff[ANN_MASK & neg_mask])))
    c_tot_flux = c_pos_flux + c_neg_flux + 1e-6
    sign_dominance_center = (c_pos_flux - c_neg_flux) / c_tot_flux

    # 11. 180度回転反対称度: I(dx, dy) ≈ -I(-dx, -dy) の幾何的一致度
    # 中心 15x15 パッチにおける点対称反対称性をスコア化 (双極で 1.0、単独点源で ~0.0)
    patch = diff[7:22, 7:22]
    patch_rot180 = np.rot90(patch, 2)
    diff_sum = np.sum(np.abs(patch + patch_rot180))
    norm_sum = np.sum(np.abs(patch) + np.abs(patch_rot180)) + 1e-6
    dipole_antisymmetry = max(0.0, float(1.0 - (diff_sum / norm_sum)))

    # --- 低 S/N ピクセル除外 (> 3σ) のサイズ・広がり測定 ---
    snr_mask = abs_diff > (3.0 * noise)
    core_snr_mask = snr_mask & (r2 <= (10.0**2))
    n_pix_masked = float(np.sum(core_snr_mask))

    if n_pix_masked >= 3:
        m_flux = abs_diff[core_snr_mask]
        m_sum = float(np.sum(m_flux)) + 1e-6
        m_xc = float(np.sum(x[core_snr_mask] * m_flux)) / m_sum
        m_yc = float(np.sum(y[core_snr_mask] * m_flux)) / m_sum

        dx_m = x[core_snr_mask] - m_xc
        dy_m = y[core_snr_mask] - m_yc
        mu_xx_m = float(np.sum(dx_m**2 * m_flux)) / m_sum
        mu_yy_m = float(np.sum(dy_m**2 * m_flux)) / m_sum
        mu_xy_m = float(np.sum(dx_m * dy_m * m_flux)) / m_sum

        radial_dispersion_masked = np.sqrt(max(0.0, mu_xx_m + mu_yy_m))

        tr = mu_xx_m + mu_yy_m
        det = mu_xx_m * mu_yy_m - mu_xy_m**2
        discriminant = max(0.0, (tr / 2.0)**2 - det)
        a_sq = tr / 2.0 + np.sqrt(discriminant)
        b_sq = tr / 2.0 - np.sqrt(discriminant)
        a_masked = np.sqrt(max(0.0, a_sq))
        b_masked = np.sqrt(max(0.0, b_sq))
        roundness_masked = b_masked / a_masked if a_masked > 1e-6 else 0.0
    else:
        radial_dispersion_masked = 0.0
        a_masked = 0.0
        b_masked = 0.0
        roundness_masked = 0.0

    features = np.array([
        centroid_dist,
        peak_snr,
        dipole_ratio,
        dipole_separation,
        concentration,
        radial_dispersion,
        ellipticity,
        border_flux_ratio,
        sign_dominance_center,
        dipole_antisymmetry,
        radial_dispersion_masked,
        a_masked,
        b_masked,
        roundness_masked,
        n_pix_masked
    ], dtype=np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

def extract_morphological_features_batch(stamps: np.ndarray) -> np.ndarray:
    """
    (N, 30, 30) または (N, C, 30, 30) の配列から高速に形態パラメータ行列 (N, 15) を算出。
    """
    if stamps.ndim == 4:
        # Difference チャンネル (チャンネル 0) を使用
        stamps = stamps[:, 0]
    
    N = len(stamps)
    out = np.zeros((N, len(MORPH_FEATURE_NAMES)), dtype=np.float32)
    for i in range(N):
        out[i] = extract_morphological_features_single(stamps[i])
    return out
