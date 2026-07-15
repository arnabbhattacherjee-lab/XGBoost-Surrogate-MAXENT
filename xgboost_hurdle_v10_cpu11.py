"""
XGBoost Hurdle V10-CPU — Rank Features + Squared-Error Regressor
=================================================================

Feature set  : 235-D
  [0   :225]  log1p(contact) patch — 15×15 window
  [225]       normalised distance  — abs(r-c) / (SIZE-1)  ∈ [0,1]
  [226]       row_sum[r]           — raw row marginal
  [227]       col_sum[c]           — raw col marginal
  [228]       row_std[r]           — std of row r
  [229]       col_std[c]           — std of col c
  [230]       O/E ratio            — contact[r,c] / diag_mean[|r-c|]
  [231]       decay_alpha (α)      — slope of log P(s) vs log s per locus
  [232]       contact_sparsity     — fraction of upper-tri contacts that are 0
  [233]       center_rank_in_patch — rank of centre log1p in its 15×15 nbhd [0,1]
  [234]       oe_rank_in_diag      — rank of O/E within diagonal band [0,1]

Lesson from V8 → V9 → V10
--------------------------
  V8 introduced contact_p95 as a raw-space H1/K562 scale separator, but the
  contact file is row_col_norm_map (ICE/KR-normalised), so all values are
  tiny fractions and the 95th percentile of the full upper triangle is always
  zero.  V9 attempted to fix this with nz_mean + nz_p95 (statistics over
  non-zero entries only), but ICE normalisation inverts the expected direction:
  because K562 loci are sparser, the normalisation budget is spread over fewer
  non-zero contacts, so K562 non-zero values are numerically LARGER than H1 —
  the opposite of what we assumed.  Both nz_mean and nz_p95 showed ranges of
  0.0000–0.0006 and 0.0000–0.0026 respectively: negligible signal with the
  wrong polarity.  They are dropped in V10.

  V9 also switched the regressor to reg:pseudohubererror with default
  huber_slope=1.  Because lambda values span hundreds to thousands, essentially
  every residual lands in the linear (MAE) regime.  MAE optimises the
  conditional median rather than the conditional mean.  The median-based
  solution correctly calibrated the overall scale (nanog_K562 R² jumped from
  0.563 to 0.916, MAE from 21.1 to 11.3) but scrambled fine-grained rank
  ordering within each locus (nanog_K562 Spearman crashed from 0.804 to 0.684;
  overall mean Spearman fell from 0.895 to 0.876).  V10 reverts to
  reg:squarederror which preserves Spearman.

  What V9 got RIGHT — and V10 keeps:
    center_rank_in_patch: rank of the centre log1p within its 15×15 nbhd.
      This is what actually rescued nanog_K562; it provided scale-independent
      structural context that allowed the model to calibrate K562 loci correctly
      without needing any cell-type label or raw-space intensity statistic.
    oe_rank_in_diag: rank-normalised O/E signal within each diagonal band.
      Removes absolute-scale dependence from the O/E feature, benefiting
      generalisation to held-out loci.
    contact_sparsity: clean H1/K562 separator on normalised maps
      (K562: 0.95–0.97; H1: 0.80–0.90).

  V10 hypothesis: rank features (centre_rank, oe_diag_rank) supply the
  calibration signal that rescues nanog_K562 R²; squared-error loss then
  correctly ranks within-locus predictions to maintain Spearman.

Lambda files : <locus_folder>/lamda_0100.txt  (3-col sparse format)
Loci         : 12  (all loci)
Marginals    : raw, un-normalised

Three operating modes (set via env vars)
-----------------------------------------
  Mode A  no env vars       : all 12 LOOCV folds + full training
  Mode B  FOLD_ID=0..11     : one fold only, saves JSON + NPY, then exits
  Mode C  FULL_TRAIN_ONLY=1 : skip LOOCV, read fold JSONs, train & save model

Saved outputs
-------------
  xgb_v10_loocv_results.csv
  xgb_v10_fold{i:02d}_{name}.png  /  _pred.npy  /  _metrics.json
  xgb_v10_clf.json
  xgb_v10_reg.json
  xgb_v10_meta.json
"""

import os
import sys
import gc
import json
import time
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from matplotlib.colors import SymLogNorm
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SIZE    = 1001
WINDOW  = 11
HALF    = WINDOW // 2   # 7
N_PATCH = WINDOW ** 2   # 225
CENTER  = HALF * WINDOW + HALF  # flat index of centre in 11×11 patch = 121

# Feature layout (235-D):
#  [0   :225]  log1p 15×15 patch
#  [225]       normalised distance
#  [226]       row_sum
#  [227]       col_sum
#  [228]       row_std
#  [229]       col_std
#  [230]       O/E ratio
#  [231]       decay_alpha
#  [232]       contact_sparsity
#  [233]       center_rank_in_patch
#  [234]       oe_rank_in_diag
N_FEATURES = N_PATCH + 10   # 235

N_CORES    = os.cpu_count() or 8
BATCH_ROWS = max(32, SIZE // max(1, N_CORES // 4))

ROOT = ""   # SLURM script must cd to scratch dir before running

print(f"[INFO] CWD          : {os.getcwd()}")
print(f"[INFO] CPU cores    : {N_CORES}")
print(f"[INFO] Batch rows   : {BATCH_ROWS}")
print(f"[INFO] Feature dim  : {N_FEATURES}")

# ── LOOCV params (no early stopping)
CLF_PARAMS_LOOCV = dict(
    tree_method="hist", device="cpu",
    n_estimators=350, max_depth=6,
    learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=10,
    n_jobs=N_CORES,
)
REG_PARAMS_LOOCV = dict(
    # reg:squarederror — preserves rank ordering (Spearman) by penalising all
    # residuals quadratically; pseudohubererror with default slope=1 was
    # effectively MAE for our lambda scale (residuals >> 1) and hurt Spearman.
    tree_method="hist", device="cpu",
    n_estimators=600, max_depth=8,
    learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5,
    n_jobs=N_CORES,
)

# ── Full-training params (5 % holdout + early stopping)
CLF_PARAMS_FULL = dict(
    tree_method="hist", device="cpu",
    n_estimators=1000, max_depth=6,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=10,
    n_jobs=N_CORES, eval_metric="logloss", early_stopping_rounds=30,
)
REG_PARAMS_FULL = dict(
    tree_method="hist", device="cpu",
    n_estimators=1500, max_depth=8,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5,
    n_jobs=N_CORES, eval_metric="mae", early_stopping_rounds=30,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOCUS REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

TARGET_CONFIGS = [
    # (display_name,    contact_subpath,                  lambda_subpath)
    ("alphaglobin_H1", "alphaglobin_H1/row_col_norm_map", "alphaglobin_H1/lamda_0100.txt"),
    ("cbx8_H1",        "cbx8_H1/row_col_norm_map",        "cbx8_H1/lamda_0100.txt"),
    ("hoxa_H1",        "hoxa_H1/row_col_norm_map",        "hoxa_H1/lamda_0100.txt"),
    ("hoxb_H1",        "hoxb_H1/row_col_norm_map",        "hoxb_H1/lamda_0100.txt"),
    ("hoxc11_H1",      "hoxc11_H1/row_col_norm_map",      "hoxc11_H1/lamda_0100.txt"),
    ("nanog_H1",       "nanog_H1/row_col_norm_map",        "nanog_H1/lamda_0100.txt"),
    ("ppm1g_H1",       "ppm1g_H1/row_col_norm_map",        "ppm1g_H1/lamda_0100.txt"),
    ("lmo2_K562",      "lmo2_K562/row_col_norm_map",       "lmo2_K562/lamda_0100.txt"),
    ("myc_K562",       "myc_K562/row_col_norm_map",        "myc_K562/lamda_0100.txt"),
    ("nanog_K562",     "nanog_K562/row_col_norm_map",      "nanog_K562/lamda_0100.txt"),
    ("sox2_K562",      "sox2_K562/row_col_norm_map",       "sox2_K562/lamda_0100.txt"),
    ("tal1_K562",      "tal1_K562/row_col_norm_map",       "tal1_K562/lamda_0100.txt"),
]
N_FOLDS = len(TARGET_CONFIGS)   # 12

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def read_contact_matrix(filename, size=SIZE):
    """3-column (row, col, value) sparse text → symmetric float32 matrix."""
    matrix = np.zeros((size, size), dtype=np.float32)
    if not os.path.exists(filename):
        print(f"  [WARNING] Not found: {filename}")
        return matrix
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                r, c, val = int(parts[0]), int(parts[1]), float(parts[2])
                if r < size and c < size:
                    matrix[r, c] = val
                    matrix[c, r] = val
            except ValueError:
                continue
    np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(matrix, 0.0, None, out=matrix)
    return matrix


def read_lambda_matrix(filename, size=SIZE):
    """3-column (row, col, value) sparse text → symmetric float32 lambda matrix."""
    matrix = np.zeros((size, size), dtype=np.float32)
    if not os.path.exists(filename):
        print(f"  [WARNING] Not found: {filename}")
        return matrix
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                r, c, val = int(parts[0]), int(parts[1]), float(parts[2])
                if r < size and c < size:
                    matrix[r, c] = val
                    matrix[c, r] = val
            except ValueError:
                continue
    np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix

# ─────────────────────────────────────────────────────────────────────────────
# PER-LOCUS STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_diagonal_means(m1, size=SIZE):
    """Mean contact at each genomic distance d = 0 … size-1 (for O/E and α)."""
    diag_means = np.zeros(size, dtype=np.float32)
    for d in range(size):
        length = size - d
        if length > 0:
            diag_means[d] = np.trace(m1, offset=d) / length
    return diag_means


def compute_decay_exponent(diag_means, size=SIZE, s_min=5, s_max_frac=0.5):
    """
    Fit log P(s) = α·log(s) + β; return only α (decay slope).

    Typical range: −1.0 … −2.0.  Weighted by sqrt(size − s) so long,
    noisy diagonals contribute less.
    """
    s_max = max(s_min + 1, int(size * s_max_frac))
    s     = np.arange(s_min, s_max, dtype=np.float64)
    ps    = diag_means[s_min:s_max].astype(np.float64)
    valid = ps > 1e-12
    if valid.sum() < 5:
        return -1.5
    log_s   = np.log(s[valid])
    log_ps  = np.log(ps[valid])
    weights = np.sqrt((size - s[valid]).astype(np.float64))
    alpha, _ = np.polyfit(log_s, log_ps, 1, w=weights)
    return float(alpha)


def compute_contact_sparsity(m1, size=SIZE):
    """
    Fraction of upper-triangle contacts that are exactly zero.

    Works correctly on ICE/KR-normalised (row_col_norm_map) contact matrices:
    K562 loci: 0.95–0.97;  H1 loci: 0.80–0.90.
    Sparsity mirrors lambda matrix sparsity and separates cell lines without
    any external annotation.

    Note: nz_mean and nz_p95 (tried in V9) are inappropriate for normalised
    maps.  ICE normalisation spreads equal budget over fewer contacts in K562,
    making K562 non-zero values numerically larger than H1 — the opposite
    direction from what a raw-contact magnitude feature should capture.
    Sparsity avoids this confound by operating on the zero/nonzero distinction
    rather than the absolute contact magnitude.
    """
    r_idx, c_idx = np.triu_indices(size, k=1)
    ut = m1[r_idx, c_idx]
    return float((ut == 0.0).mean())


def compute_oe_diag_ranks(m1, diag_means, size=SIZE):
    """
    For every off-diagonal entry (r, c) compute the fractional rank of its
    O/E value within all entries on the same diagonal band (same |r-c|).

    Result: (size, size) float32 array, values ∈ [0, 1].
      0 → lowest O/E on its diagonal (below-expected contact)
      1 → highest O/E (strongly above-expected, e.g. loop anchor)

    Rank-normalising the O/E signal removes absolute-scale dependence so the
    feature generalises equally well to dense H1 and sparse K562 loci.
    Precomputed once per locus — O(n²) total.
    """
    oe_ranks = np.zeros((size, size), dtype=np.float32)
    for d in range(1, size):
        r_idx = np.arange(0, size - d)
        c_idx = r_idx + d
        raw   = m1[r_idx, c_idx].astype(np.float64)
        exp   = float(diag_means[d]) if diag_means[d] > 1e-9 else 1e-9
        oe_d  = (raw / exp).astype(np.float32)
        n     = len(oe_d)
        if n > 1:
            order        = np.argsort(oe_d)
            ranks        = np.empty(n, dtype=np.float32)
            ranks[order] = np.arange(n, dtype=np.float32) / (n - 1)
        else:
            ranks = np.full(n, 0.5, dtype=np.float32)
        oe_ranks[r_idx, c_idx] = ranks
        oe_ranks[c_idx, r_idx] = ranks
    return oe_ranks

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (235-D, vectorised + thread-parallel)
# ─────────────────────────────────────────────────────────────────────────────

def _build_view_log(m1):
    """log1p-transform m1, reflect-pad, return sliding-window view."""
    log_m1 = np.log1p(m1.astype(np.float32))
    m1_pad = np.pad(log_m1, HALF, mode="reflect").astype(np.float32)
    return sliding_window_view(m1_pad, (WINDOW, WINDOW))


def _extract_row(r, view_log, raw_contact_r,
                 row_sums, col_sums, row_stds, col_stds,
                 diag_means, decay_alpha, contact_sparsity,
                 oe_diag_ranks,
                 m2, size=SIZE):
    """
    Build 235-D feature matrix and target vector for upper-triangle row r.

    [0   :225]  log1p 15×15 patch centred at (r, c)
    [225]       abs(r-c)/(size-1)  normalised distance  ∈ [0,1]
    [226]       row_sum[r]
    [227]       col_sum[c]
    [228]       row_std[r]
    [229]       col_std[c]
    [230]       contact[r,c] / diag_mean[|r-c|]  (O/E ratio)
    [231]       decay_alpha         (locus-level P(s) slope)
    [232]       contact_sparsity    (locus-level fraction-zero contact)
    [233]       center_rank_in_patch  (rank of centre log1p in 15×15 nbhd [0,1])
    [234]       oe_rank_in_diag       (rank of O/E within diagonal band [0,1])
    """
    c_range = np.arange(r, size, dtype=np.int64)
    nc      = len(c_range)

    # ── log1p patch features  (nc, 225)
    patches_log = view_log[r, c_range].reshape(nc, -1).astype(np.float32)

    # ── Normalised distance
    dist_raw  = (c_range - r).astype(np.float32)
    dist_norm = dist_raw / (size - 1)

    # ── Marginals
    rs   = np.full(nc, row_sums[r],  dtype=np.float32)
    cs   = col_sums[c_range].astype(np.float32)
    rstd = np.full(nc, row_stds[r],  dtype=np.float32)
    cstd = col_stds[c_range].astype(np.float32)

    # ── O/E ratio (raw contact / expected-at-distance)
    raw_vals = raw_contact_r[c_range].astype(np.float32)
    dist_int = dist_raw.astype(np.int32)
    exp_vals = diag_means[dist_int]
    oe       = np.where(exp_vals > 1e-9,
                        raw_vals / exp_vals,
                        0.0).astype(np.float32)

    # ── Locus-level scalars (same for every pair in this locus)
    alpha_col = np.full(nc, decay_alpha,       dtype=np.float32)
    spar_col  = np.full(nc, contact_sparsity,  dtype=np.float32)

    # ── center_rank_in_patch: rank of centre log1p within its 15×15 nbhd [0,1]
    #    CENTER = 121 (flat index of pixel [HALF, HALF] in a 11×11 patch)
    centre_vals  = patches_log[:, CENTER]          # (nc,)
    centre_rank  = ((patches_log < centre_vals[:, None])
                    .sum(axis=1)
                    .astype(np.float32)) / (N_PATCH - 1)

    # ── oe_rank_in_diag: precomputed fractional rank of O/E per diagonal
    oe_diag_r = oe_diag_ranks[r, c_range].astype(np.float32)   # (nc,)

    X_row = np.concatenate([
        patches_log,          # (nc, 225)
        dist_norm [:, None],  # (nc, 1)
        rs        [:, None],  # (nc, 1)
        cs        [:, None],  # (nc, 1)
        rstd      [:, None],  # (nc, 1)
        cstd      [:, None],  # (nc, 1)
        oe        [:, None],  # (nc, 1)
        alpha_col [:, None],  # (nc, 1)
        spar_col  [:, None],  # (nc, 1)
        centre_rank[:, None], # (nc, 1)
        oe_diag_r [:, None],  # (nc, 1)
    ], axis=1)                # (nc, 235)

    y_row = m2[r, c_range].astype(np.float32)
    return X_row, y_row


def _extract_locus(m1, m2, size=SIZE):
    """
    Extract all upper-triangle 235-D features for one locus.
    All statistics are derived from m1 — no external labels required.
    Returns (X, y, decay_alpha, contact_sparsity).
    """
    view_log         = _build_view_log(m1)
    row_sums         = m1.sum(axis=1).astype(np.float32)
    col_sums         = m1.sum(axis=0).astype(np.float32)
    row_stds         = m1.std(axis=1).astype(np.float32)
    col_stds         = m1.std(axis=0).astype(np.float32)
    diag_means       = compute_diagonal_means(m1, size)
    decay_alpha      = compute_decay_exponent(diag_means, size)
    contact_sp       = compute_contact_sparsity(m1, size)
    oe_diag_ranks    = compute_oe_diag_ranks(m1, diag_means, size)

    results = Parallel(n_jobs=N_CORES, prefer="threads")(
        delayed(_extract_row)(
            r, view_log, m1[r],
            row_sums, col_sums, row_stds, col_stds,
            diag_means, decay_alpha, contact_sp,
            oe_diag_ranks,
            m2, size,
        )
        for r in range(size)
    )

    X = np.concatenate([rx[0] for rx in results], axis=0)
    y = np.concatenate([rx[1] for rx in results], axis=0)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, decay_alpha, contact_sp


def build_dataset(m1_list, m2_list):
    """
    Build full feature matrix for a list of loci.
    Returns (X, y, locus_stats) where locus_stats is a list of
    (decay_alpha, contact_sparsity) per locus.
    """
    n_upper = (SIZE * (SIZE + 1)) // 2
    total   = n_upper * len(m1_list)
    X = np.empty((total, N_FEATURES), dtype=np.float32)
    y = np.empty(total,               dtype=np.float32)
    stats = []

    start = 0
    for k, (m1, m2) in enumerate(zip(m1_list, m2_list)):
        t0 = time.time()
        Xl, yl, alpha, spar = _extract_locus(m1, m2)
        X[start : start + n_upper] = Xl
        y[start : start + n_upper] = yl
        start += n_upper
        stats.append((alpha, spar))
        del Xl, yl; gc.collect()
        print(f"    locus {k+1}/{len(m1_list)} done in {time.time()-t0:.1f}s"
              f"  α={alpha:.3f}  sparsity={spar:.3f}")

    return X, y, stats

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE  (vectorised + batch)
# ─────────────────────────────────────────────────────────────────────────────

def predict_hurdle(m1, clf, reg, size=SIZE):
    """
    Predict the full (size × size) λ matrix.
    All statistics derived from m1 — no external annotation required.
    """
    view_log      = _build_view_log(m1)
    row_sums      = m1.sum(axis=1).astype(np.float32)
    col_sums      = m1.sum(axis=0).astype(np.float32)
    row_stds      = m1.std(axis=1).astype(np.float32)
    col_stds      = m1.std(axis=0).astype(np.float32)
    diag_means    = compute_diagonal_means(m1, size)
    decay_alpha   = compute_decay_exponent(diag_means, size)
    contact_sp    = compute_contact_sparsity(m1, size)
    oe_diag_ranks = compute_oe_diag_ranks(m1, diag_means, size)
    dummy_m2      = np.zeros((size, size), dtype=np.float32)
    full_pred     = np.zeros((size, size), dtype=np.float32)

    print(f"    [predict] α={decay_alpha:.3f}  sparsity={contact_sp:.3f}")

    for batch_start in range(0, size, BATCH_ROWS):
        batch_rows = list(range(batch_start,
                                min(batch_start + BATCH_ROWS, size)))

        rows = Parallel(n_jobs=N_CORES, prefer="threads")(
            delayed(_extract_row)(
                r, view_log, m1[r],
                row_sums, col_sums, row_stds, col_stds,
                diag_means, decay_alpha, contact_sp,
                oe_diag_ranks,
                dummy_m2, size,
            )
            for r in batch_rows
        )

        X_batch     = np.concatenate([rx[0] for rx in rows], axis=0)
        row_lengths = [len(rx[1]) for rx in rows]
        del rows
        np.nan_to_num(X_batch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        is_nz  = clf.predict(X_batch)
        preds  = np.zeros(len(X_batch), dtype=np.float32)
        nz_idx = np.where(is_nz == 1)[0]
        if len(nz_idx) > 0:
            raw = reg.predict(X_batch[nz_idx])
            preds[nz_idx] = np.nan_to_num(
                raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        del X_batch

        offset = 0
        for r, nc in zip(batch_rows, row_lengths):
            c_range = np.arange(r, size)
            full_pred[r,       c_range] = preds[offset : offset + nc]
            full_pred[c_range, r      ] = preds[offset : offset + nc]
            offset += nc

    np.nan_to_num(full_pred, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return full_pred

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL MATRICES
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading all matrices …")
all_mat1, all_mat2, names = [], [], []
for display_name, m1_sub, m2_sub in TARGET_CONFIGS:
    m1_path = os.path.join(ROOT, m1_sub) if ROOT else m1_sub
    m2_path = os.path.join(ROOT, m2_sub) if ROOT else m2_sub
    m1 = read_contact_matrix(m1_path)
    m2 = read_lambda_matrix(m2_path)
    np.fill_diagonal(m1, 0)
    np.fill_diagonal(m2, 0)
    all_mat1.append(m1)
    all_mat2.append(m2)
    names.append(display_name)
    print(f"  Loaded: {display_name}")

# Print per-locus contact statistics for diagnostics
print("\nPer-locus contact distribution stats (V10 features):")
print(f"  {'Locus':<22}  {'α':>7}  {'sparsity':>9}")
for name, m1 in zip(names, all_mat1):
    dm   = compute_diagonal_means(m1)
    alp  = compute_decay_exponent(dm)
    spar = compute_contact_sparsity(m1)
    print(f"  {name:<22}  {alp:>7.4f}  {spar:>9.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# OPERATING MODE
# ─────────────────────────────────────────────────────────────────────────────

_fold_env       = os.environ.get("FOLD_ID",         "").strip()
_fulltrain_only = os.environ.get("FULL_TRAIN_ONLY", "").strip() == "1"

if _fulltrain_only:
    FOLDS_TO_RUN      = []
    RUN_FULL_TRAINING = True
    print("\n[MODE C] Full-train-only: reading fold JSONs → train production model.")
elif _fold_env != "":
    FOLDS_TO_RUN      = [int(_fold_env)]
    RUN_FULL_TRAINING = False
    print(f"\n[MODE B] Single fold {FOLDS_TO_RUN[0]}: {names[FOLDS_TO_RUN[0]]}")
else:
    FOLDS_TO_RUN      = list(range(N_FOLDS))
    RUN_FULL_TRAINING = True
    print(f"\n[MODE A] All {N_FOLDS} LOOCV folds + full training.")

# ─────────────────────────────────────────────────────────────────────────────
# LOOCV
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
results = []

for i in FOLDS_TO_RUN:
    test_name = names[i]
    print(f"\n{'='*65}")
    print(f"  FOLD {i+1}/{N_FOLDS}: {test_name}  ({N_CORES} CPU cores)")
    print("="*65)

    train_m1 = [all_mat1[j] for j in range(N_FOLDS) if j != i]
    train_m2 = [all_mat2[j] for j in range(N_FOLDS) if j != i]

    print(f"  Building {N_FEATURES}-D features from {len(train_m1)} loci …")
    t0 = time.time()
    X_train, y_train, train_stats = build_dataset(train_m1, train_m2)
    print(f"  Feature extraction: {(time.time()-t0)/60:.1f} min  X={X_train.shape}")

    # Hurdle training
    y_binary = (np.abs(y_train) > 1e-7).astype(int)

    print(f"  Training classifier  "
          f"(n_est={CLF_PARAMS_LOOCV['n_estimators']}, "
          f"lr={CLF_PARAMS_LOOCV['learning_rate']}) …")
    clf = xgb.XGBClassifier(**CLF_PARAMS_LOOCV)
    clf.fit(X_train, y_binary)

    X_reg = X_train[y_binary == 1]
    y_reg = y_train[y_binary == 1]
    print(f"  Training regressor on {len(X_reg):,} non-zero samples  "
          f"(n_est={REG_PARAMS_LOOCV['n_estimators']}, "
          f"lr={REG_PARAMS_LOOCV['learning_rate']}, "
          f"obj=squarederror) …")
    reg = xgb.XGBRegressor(**REG_PARAMS_LOOCV)
    reg.fit(X_reg, y_reg)

    del X_train, y_train, X_reg, y_reg; gc.collect()

    # Prediction
    print("  Predicting …")
    t0         = time.time()
    prediction = predict_hurdle(all_mat1[i], clf, reg)
    print(f"  Inference: {(time.time()-t0)/60:.1f} min")

    # Metrics (upper triangle)
    mask   = np.triu(np.ones((SIZE, SIZE), dtype=bool), k=1)
    y_true = all_mat2[i][mask]
    y_pred = prediction[mask]

    r2         = float(r2_score(y_true, y_pred))
    p_corr     = float(pearsonr(y_true, y_pred)[0])
    s_corr     = float(spearmanr(y_true, y_pred)[0])
    mae        = float(mean_absolute_error(y_true, y_pred))
    tot_zeros  = int(np.sum(np.abs(y_true) <= 1e-7))
    match_zero = int(np.sum((np.abs(y_true) <= 1e-7) & (np.abs(y_pred) <= 1e-7)))

    # Val-locus contact stats
    val_dm   = compute_diagonal_means(all_mat1[i])
    val_alpha = compute_decay_exponent(val_dm)
    val_spar  = compute_contact_sparsity(all_mat1[i])

    print(f"  R²={r2:.4f}  Pearson={p_corr:.4f}  Spearman={s_corr:.4f}  "
          f"MAE={mae:.4f}")
    print(f"  α={val_alpha:.4f}  sparsity={val_spar:.4f}")

    fold_result = {
        "Folder"          : test_name,
        "R2"              : r2,
        "Pearson"         : p_corr,
        "Spearman"        : s_corr,
        "MAE"             : mae,
        "Y_Zeros"         : tot_zeros,
        "Y_Pred_Overlap"  : match_zero,
        "decay_alpha"     : val_alpha,
        "contact_sparsity": val_spar,
    }
    results.append(fold_result)

    with open(f"xgb_v10_fold{i:02d}_{test_name}_metrics.json", "w") as f:
        json.dump(fold_result, f, indent=2)
    np.save(f"xgb_v10_fold{i:02d}_{test_name}_pred.npy",
            prediction.astype(np.float32))

    norm = SymLogNorm(linthresh=1e-5,
                      vmin=all_mat2[i].min(), vmax=all_mat2[i].max(), base=10)
    plt.figure(figsize=(20, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(all_mat2[i] * mask, cmap="RdBu_r", norm=norm)
    plt.title(f"Actual: {test_name}\n(Zeros: {tot_zeros})", fontsize=14)
    plt.colorbar(label=r"$\lambda$ (SymLog)")
    plt.subplot(1, 2, 2)
    plt.imshow(prediction * mask, cmap="RdBu_r", norm=norm)
    plt.title(f"V10 Prediction  R²={r2:.4f}  Pearson={p_corr:.4f}  "
              f"Spearman={s_corr:.4f}\n"
              f"α={val_alpha:.3f}  sparsity={val_spar:.3f}", fontsize=13)
    plt.colorbar(label=r"$\lambda$ (SymLog)")
    plt.tight_layout()
    plt.savefig(f"xgb_v10_fold{i:02d}_{test_name}.png", dpi=100,
                bbox_inches="tight")
    plt.close()

    print(f"  Saved: xgb_v10_fold{i:02d}_{test_name}_metrics.json  _pred.npy  .png")
    del clf, reg, prediction; gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# MODE B EXIT
# ─────────────────────────────────────────────────────────────────────────────

if not RUN_FULL_TRAINING:
    print(f"\n[MODE B] Fold {FOLDS_TO_RUN[0]} complete. "
          "Submit submit_xgb_v10_fulltrain.slurm after all folds finish.")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# LOOCV SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

if len(results) == 0:
    print("\n  Reading per-fold metrics from JSON files …")
    for i, (name, _, _) in enumerate(TARGET_CONFIGS):
        p = f"xgb_v10_fold{i:02d}_{name}_metrics.json"
        if os.path.exists(p):
            with open(p) as f:
                results.append(json.load(f))
            print(f"  Loaded: {p}")
        else:
            print(f"  [WARNING] Missing: {p}")

df  = pd.DataFrame(results)
hdr = "=" * 80

print(f"\n{hdr}")
print("  FINAL SUMMARY — XGBoost Hurdle V10-CPU  (LOOCV  |  12 loci)")
print(f"  {N_FEATURES}-D: 11×11 log1p patch + norm_dist + O/E + "
      "row/col marginals + α + sparsity + centre_rank + oe_diag_rank")
print(hdr)

show_cols = ["Folder", "R2", "Pearson", "Spearman", "MAE",
             "decay_alpha", "contact_sparsity"]
show_cols = [c for c in show_cols if c in df.columns]
print(df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\n  Mean R²       = {df['R2'].mean():.4f}")
print(f"  Mean Pearson  = {df['Pearson'].mean():.4f}")
print(f"  Mean Spearman = {df['Spearman'].mean():.4f}")
print(f"  Mean MAE      = {df['MAE'].mean():.4f}")

k562 = df["Folder"].str.contains("K562")
h1   = df["Folder"].str.contains("H1")
print(f"\n  K562  R²={df.loc[k562,'R2'].mean():.4f}"
      f"  Pearson={df.loc[k562,'Pearson'].mean():.4f}"
      f"  Spearman={df.loc[k562,'Spearman'].mean():.4f}")
print(f"  H1    R²={df.loc[h1,'R2'].mean():.4f}"
      f"  Pearson={df.loc[h1,'Pearson'].mean():.4f}"
      f"  Spearman={df.loc[h1,'Spearman'].mean():.4f}")

print(f"\n{hdr}")
print("  SPARSITY (lambda zeros)")
print(hdr)
print(df[["Folder", "Y_Zeros", "Y_Pred_Overlap"]].to_string(index=False))

df.to_csv("xgb_v10_loocv_results.csv", index=False, float_format="%.6f")
print("\n[Saved] xgb_v10_loocv_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FULL TRAINING ON ALL 12 LOCI
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{hdr}")
print("  FULL TRAINING — all 12 loci  (production model)")
print(hdr)

t_full = time.time()
print(f"  Building {N_FEATURES}-D features from all 12 loci …")
X_full, y_full, all_stats = build_dataset(all_mat1, all_mat2)
print(f"  Full dataset: X={X_full.shape}  y={y_full.shape}")

# 5 % holdout for early stopping
np.random.seed(42)
n_total = len(X_full)
n_val   = max(1, int(0.05 * n_total))
val_idx = np.random.choice(n_total, size=n_val, replace=False)
tr_mask = np.ones(n_total, dtype=bool)
tr_mask[val_idx] = False

X_ft, y_ft = X_full[tr_mask],  y_full[tr_mask]
X_fv, y_fv = X_full[~tr_mask], y_full[~tr_mask]
del X_full, y_full; gc.collect()

# Classifier
y_bin_ft = (np.abs(y_ft) > 1e-7).astype(int)
y_bin_fv = (np.abs(y_fv) > 1e-7).astype(int)
print(f"  Training classifier  "
      f"(n_est={CLF_PARAMS_FULL['n_estimators']}, "
      f"early_stop={CLF_PARAMS_FULL['early_stopping_rounds']}) …")
clf_full = xgb.XGBClassifier(**CLF_PARAMS_FULL)
clf_full.fit(X_ft, y_bin_ft, eval_set=[(X_fv, y_bin_fv)], verbose=False)
print(f"  Classifier best_iteration = {clf_full.best_iteration}")

# Regressor
nz_ft = y_bin_ft == 1
nz_fv = (np.abs(y_fv) > 1e-7).astype(int) == 1
X_rft, y_rft = X_ft[nz_ft], y_ft[nz_ft]
X_rfv, y_rfv = X_fv[nz_fv], y_fv[nz_fv]
fin_ft = np.isfinite(X_rft).all(axis=1) & np.isfinite(y_rft)
fin_fv = np.isfinite(X_rfv).all(axis=1) & np.isfinite(y_rfv)
X_rft, y_rft = X_rft[fin_ft], y_rft[fin_ft]
X_rfv, y_rfv = X_rfv[fin_fv], y_rfv[fin_fv]

print(f"  Training regressor on {len(X_rft):,} non-zero samples  "
      f"(n_est={REG_PARAMS_FULL['n_estimators']}, "
      f"obj=squarederror, "
      f"early_stop={REG_PARAMS_FULL['early_stopping_rounds']}) …")
reg_full = xgb.XGBRegressor(**REG_PARAMS_FULL)
reg_full.fit(X_rft, y_rft, eval_set=[(X_rfv, y_rfv)], verbose=False)
print(f"  Regressor best_iteration = {reg_full.best_iteration}")

del X_ft, y_ft, X_fv, y_fv, X_rft, y_rft, X_rfv, y_rfv
gc.collect()

clf_full.save_model("xgb_v10_clf.json")
reg_full.save_model("xgb_v10_reg.json")

meta = {
    "version"               : "v10",
    "n_features"            : N_FEATURES,
    "feature_description"   : (
        f"log1p_{WINDOW}x{WINDOW}_patch({N_PATCH}) | "
        "norm_dist(1) | row_sum(1) | col_sum(1) | "
        "row_std(1) | col_std(1) | oe_ratio(1) | "
        "decay_alpha(1) | contact_sparsity(1) | "
        "center_rank_in_patch(1) | oe_rank_in_diag(1)"
    ),
    "window"                : WINDOW,
    "size"                  : SIZE,
    "distance_norm"         : "abs(r-c) / (SIZE-1) in [0,1]",
    "decay_alpha_fit"       : (
        "log P(s) = alpha*log(s)+beta, s in [5,SIZE/2], "
        "weighted sqrt(SIZE-s); only alpha retained"
    ),
    "contact_sparsity_desc" : (
        "fraction of upper-triangle contacts == 0 per locus; "
        "valid for ICE/KR-normalised maps (K562: 0.95-0.97, H1: 0.80-0.90)"
    ),
    "center_rank_desc"      : (
        "fractional rank of centre log1p value within its 11x11 log1p "
        "neighbourhood [0,1]; 0=local minimum, 1=local maximum; "
        "scale-independent — rescued nanog_K562 calibration in V9"
    ),
    "oe_rank_in_diag_desc"  : (
        "fractional rank of O/E value within all entries on same diagonal band "
        "[0,1]; rank-normalised so H1/K562 absolute scale differences cancel"
    ),
    "regressor_objective"   : (
        "reg:squarederror (default); preserves rank ordering (Spearman); "
        "pseudohubererror with slope=1 was avoided because it collapses to "
        "MAE for lambda residuals >> 1, hurting Spearman"
    ),
    "lambda_file"           : "<locus_folder>/lamda_0100.txt",
    "training_loci"         : names,
    "n_loci"                : N_FOLDS,
    "clf_best_iteration"    : int(clf_full.best_iteration),
    "reg_best_iteration"    : int(reg_full.best_iteration),
    "loocv_mean_R2"         : float(df["R2"].mean()),
    "loocv_mean_Pearson"    : float(df["Pearson"].mean()),
    "loocv_mean_Spearman"   : float(df["Spearman"].mean()),
    "n_cores_used"          : N_CORES,
    "load_instructions"     : (
        "clf=xgb.XGBClassifier(); clf.load_model('xgb_v10_clf.json'); "
        "reg=xgb.XGBRegressor();  reg.load_model('xgb_v10_reg.json')"
    ),
    "predict_instructions"  : (
        "pred = predict_hurdle(contact_matrix, clf, reg)  "
        "# no cell-type label needed; all stats derived from contact_matrix"
    ),
}
with open("xgb_v10_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n  Full training done in {(time.time()-t_full)/60:.1f} minutes.")
print("\n  Saved:")
for fname in ["xgb_v10_clf.json", "xgb_v10_reg.json", "xgb_v10_meta.json"]:
    sz = os.path.getsize(fname) / 1e6 if os.path.exists(fname) else 0
    print(f"    {fname:<35}  {sz:.1f} MB")

print(f"""
{hdr}
  HOW TO PREDICT A NEW LOCUS  (any cell type; ~2 kb resolution)
{hdr}
  import xgboost as xgb
  clf = xgb.XGBClassifier(); clf.load_model("xgb_v10_clf.json")
  reg = xgb.XGBRegressor();  reg.load_model("xgb_v10_reg.json")

  # contact_matrix : (N × N) float32, ICE/KR-normalised, same resolution
  # All features derived from the contact matrix — no labels needed
  pred = predict_hurdle(contact_matrix, clf, reg, size=contact_matrix.shape[0])
  # pred → (N × N) float32 lambda matrix
{hdr}
""")
