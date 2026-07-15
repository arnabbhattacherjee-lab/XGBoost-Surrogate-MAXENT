"""
linear_baseline_v10.py
=======================
Purpose: JCTC reviewers reading the feature-ablation study will want direct
evidence that gradient boosting is doing something a plain linear model
CANNOT do on the same features, not just that boosting-on-patch < boosting-
on-scalars < boosting-on-full. This script answers that by holding the
FEATURE SET fixed at the 10-column "scalars" subset (the same columns used
in xgboost_hurdle_v10_feature_ablation.py with FEATURE_SET=scalars) and
swapping only the MODEL CLASS: logistic + ridge/linear regression instead
of the XGBoost hurdle model.

This is a drop-in companion to xgboost_hurdle_v10_feature_ablation.py. It
reuses the identical data loading, feature extraction (scalars-only slice),
LOOCV loop, masking, and metric computation, so the numbers it produces are
directly comparable, row for row, against xgb_v10_scalars_loocv_results.csv.

WHY THIS MATTERS FOR THE PAPER
-------------------------------
If linear-on-scalars performs close to XGBoost-on-scalars, that would say
the 10 scalar descriptors are so strongly (near-)linearly related to lambda
that boosting isn't buying much beyond the raw features, i.e. the interesting
claim would shift from "gradient boosting is essential" to "the scalar
descriptors are essential, any reasonable model on top of them works." If
linear-on-scalars performs markedly worse, that is direct evidence that the
mapping from these ten descriptors to lambda is genuinely nonlinear
(thresholds, interactions between distance and O/E, etc.), which is exactly
the claim the manuscript currently only ASSERTS in the "why gradient-boosted
trees outperform CNNs" discussion paragraph.

Either outcome is publishable; you just need to know which one you have
before the paper goes out, because a reviewer will ask.

MODEL DETAILS
-------------
  Stage 1 (classification): sklearn LogisticRegression, L2-penalised,
    class_weight="balanced" (the 89/11 class split is moderate but a linear
    boundary may still need the reweighting to avoid a majority-class
    collapse), solver="lbfgs", max_iter=1000.
  Stage 2 (regression): sklearn Ridge (alpha selected by a small internal
    grid via RidgeCV on the training folds only, never touching the held-out
    locus), trained on the raw lambda target -- identical target definition
    to the V10 XGBoost regressor (Eq. y^reg_ij = lambda_ij, squared-error
    fit), so the comparison isolates model class, not target transform.

Usage (identical calling convention to the ablation script):
  FOLD_ID=0            python linear_baseline_v10.py   # one fold, for testing/SLURM arrays
  (no env vars)         python linear_baseline_v10.py   # all 12 LOOCV folds

Output: xgb_v10_linear_scalars_loocv_results.csv, with the same columns as
the other *_loocv_results.csv files, so it drops straight into
ablation_compare.py / your own comparison notebook alongside the XGBoost
results.
"""

import os
import sys
import gc
import json
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION -- identical to xgboost_hurdle_v10_feature_ablation.py
# ─────────────────────────────────────────────────────────────────────────────

SIZE    = 1001
WINDOW  = 11                     # matches the production XGBoost patch size
HALF    = WINDOW // 2
N_PATCH = WINDOW ** 2
CENTER  = HALF * WINDOW + HALF
N_FEATURES = N_PATCH + 10        # 131, only used to size the intermediate array

# We only ever train on the 10 scalar columns here -- see SCALAR_COLS below.
SCALAR_COLS = slice(N_PATCH, N_FEATURES)

N_CORES = os.cpu_count() or 8
ROOT    = ""   # SLURM script must cd to scratch dir before running

TARGET_CONFIGS = [
    ("alphaglobin_H1", "alphaglobin_H1/row_col_norm_map", "alphaglobin_H1/lamda_0100.txt"),
    ("cbx8_H1",        "cbx8_H1/row_col_norm_map",        "cbx8_H1/lamda_0100.txt"),
    ("hoxa_H1",        "hoxa_H1/row_col_norm_map",        "hoxa_H1/lamda_0100.txt"),
    ("hoxb_H1",        "hoxb_H1/row_col_norm_map",        "hoxb_H1/lamda_0100.txt"),
    ("hoxc11_H1",      "hoxc11_H1/row_col_norm_map",      "hoxc11_H1/lamda_0100.txt"),
    ("nanog_H1",       "nanog_H1/row_col_norm_map",       "nanog_H1/lamda_0100.txt"),
    ("ppm1g_H1",       "ppm1g_H1/row_col_norm_map",       "ppm1g_H1/lamda_0100.txt"),
    ("lmo2_K562",      "lmo2_K562/row_col_norm_map",      "lmo2_K562/lamda_0100.txt"),
    ("myc_K562",       "myc_K562/row_col_norm_map",       "myc_K562/lamda_0100.txt"),
    ("nanog_K562",     "nanog_K562/row_col_norm_map",     "nanog_K562/lamda_0100.txt"),
    ("sox2_K562",      "sox2_K562/row_col_norm_map",      "sox2_K562/lamda_0100.txt"),
    ("tal1_K562",      "tal1_K562/row_col_norm_map",      "tal1_K562/lamda_0100.txt"),
]
N_FOLDS = len(TARGET_CONFIGS)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (identical to the ablation script)
# ─────────────────────────────────────────────────────────────────────────────

def read_matrix(filename, size=SIZE, clip_nonneg=False):
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
    if clip_nonneg:
        np.clip(matrix, 0.0, None, out=matrix)
    return matrix


def compute_diagonal_means(m1, size=SIZE):
    diag_means = np.zeros(size, dtype=np.float32)
    for d in range(size):
        length = size - d
        if length > 0:
            diag_means[d] = np.trace(m1, offset=d) / length
    return diag_means


def compute_decay_exponent(diag_means, size=SIZE, s_min=5, s_max_frac=0.5):
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
    r_idx, c_idx = np.triu_indices(size, k=1)
    ut = m1[r_idx, c_idx]
    return float((ut == 0.0).mean())


def compute_oe_diag_ranks(m1, diag_means, size=SIZE):
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


def _build_view_log(m1):
    log_m1 = np.log1p(m1.astype(np.float32))
    m1_pad = np.pad(log_m1, HALF, mode="reflect").astype(np.float32)
    return sliding_window_view(m1_pad, (WINDOW, WINDOW))


def _extract_row_scalars_only(r, view_log, raw_contact_r,
                               row_sums, col_sums, row_stds, col_stds,
                               diag_means, decay_alpha, contact_sparsity,
                               oe_diag_ranks, m2, size=SIZE):
    """Same 10 scalar columns as the ablation script's SCALAR_COLS slice,
    computed WITHOUT ever materialising the 121-pixel patch array for every
    row (we still need the patch to get center_rank_in_patch, but we discard
    it immediately after)."""
    c_range = np.arange(r, size, dtype=np.int64)
    nc      = len(c_range)

    patches_log = view_log[r, c_range].reshape(nc, -1).astype(np.float32)

    dist_raw  = (c_range - r).astype(np.float32)
    dist_norm = dist_raw / (size - 1)

    rs   = np.full(nc, row_sums[r],  dtype=np.float32)
    cs   = col_sums[c_range].astype(np.float32)
    rstd = np.full(nc, row_stds[r],  dtype=np.float32)
    cstd = col_stds[c_range].astype(np.float32)

    raw_vals = raw_contact_r[c_range].astype(np.float32)
    dist_int = dist_raw.astype(np.int32)
    exp_vals = diag_means[dist_int]
    oe       = np.where(exp_vals > 1e-9, raw_vals / exp_vals, 0.0).astype(np.float32)

    alpha_col = np.full(nc, decay_alpha,      dtype=np.float32)
    spar_col  = np.full(nc, contact_sparsity, dtype=np.float32)

    centre_vals = patches_log[:, CENTER]
    centre_rank = ((patches_log < centre_vals[:, None]).sum(axis=1)
                   .astype(np.float32)) / (N_PATCH - 1)
    oe_diag_r   = oe_diag_ranks[r, c_range].astype(np.float32)

    X_row = np.concatenate([
        dist_norm[:, None], rs[:, None], cs[:, None], rstd[:, None],
        cstd[:, None], oe[:, None], alpha_col[:, None], spar_col[:, None],
        centre_rank[:, None], oe_diag_r[:, None],
    ], axis=1)   # (nc, 10)

    y_row = m2[r, c_range].astype(np.float32)
    return X_row, y_row


def _extract_locus_scalars(m1, m2, size=SIZE):
    view_log      = _build_view_log(m1)
    row_sums      = m1.sum(axis=1).astype(np.float32)
    col_sums      = m1.sum(axis=0).astype(np.float32)
    row_stds      = m1.std(axis=1).astype(np.float32)
    col_stds      = m1.std(axis=0).astype(np.float32)
    diag_means    = compute_diagonal_means(m1, size)
    decay_alpha   = compute_decay_exponent(diag_means, size)
    contact_sp    = compute_contact_sparsity(m1, size)
    oe_diag_ranks = compute_oe_diag_ranks(m1, diag_means, size)

    results = Parallel(n_jobs=N_CORES, prefer="threads")(
        delayed(_extract_row_scalars_only)(
            r, view_log, m1[r], row_sums, col_sums, row_stds, col_stds,
            diag_means, decay_alpha, contact_sp, oe_diag_ranks, m2, size,
        )
        for r in range(size)
    )
    X = np.concatenate([rx[0] for rx in results], axis=0)
    y = np.concatenate([rx[1] for rx in results], axis=0)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, decay_alpha, contact_sp


def build_dataset_scalars(m1_list, m2_list):
    n_upper = (SIZE * (SIZE + 1)) // 2
    total   = n_upper * len(m1_list)
    X = np.empty((total, 10), dtype=np.float32)
    y = np.empty(total,       dtype=np.float32)
    start = 0
    for k, (m1, m2) in enumerate(zip(m1_list, m2_list)):
        t0 = time.time()
        Xl, yl, alpha, spar = _extract_locus_scalars(m1, m2)
        X[start:start + n_upper] = Xl
        y[start:start + n_upper] = yl
        start += n_upper
        del Xl, yl; gc.collect()
        print(f"    locus {k+1}/{len(m1_list)} done in {time.time()-t0:.1f}s"
              f"  alpha={alpha:.3f}  sparsity={spar:.3f}")
    return X, y


def predict_linear(m1, scaler, clf, reg, size=SIZE):
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

    BATCH_ROWS = max(32, size // max(1, (os.cpu_count() or 8) // 4))
    for batch_start in range(0, size, BATCH_ROWS):
        batch_rows = list(range(batch_start, min(batch_start + BATCH_ROWS, size)))
        rows = Parallel(n_jobs=N_CORES, prefer="threads")(
            delayed(_extract_row_scalars_only)(
                r, view_log, m1[r], row_sums, col_sums, row_stds, col_stds,
                diag_means, decay_alpha, contact_sp, oe_diag_ranks, dummy_m2, size,
            )
            for r in batch_rows
        )
        X_batch     = np.concatenate([rx[0] for rx in rows], axis=0)
        row_lengths = [len(rx[1]) for rx in rows]
        del rows
        np.nan_to_num(X_batch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = scaler.transform(X_batch)
        is_nz    = clf.predict(X_scaled)
        preds    = np.zeros(len(X_batch), dtype=np.float32)
        nz_idx   = np.where(is_nz == 1)[0]
        if len(nz_idx) > 0:
            raw = reg.predict(X_scaled[nz_idx])
            preds[nz_idx] = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        del X_batch, X_scaled

        offset = 0
        for r, nc in zip(batch_rows, row_lengths):
            c_range = np.arange(r, size)
            full_pred[r, c_range] = preds[offset:offset + nc]
            full_pred[c_range, r] = preds[offset:offset + nc]
            offset += nc

    np.nan_to_num(full_pred, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return full_pred


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL MATRICES
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading all matrices ...")
all_mat1, all_mat2, names = [], [], []
for display_name, m1_sub, m2_sub in TARGET_CONFIGS:
    m1_path = os.path.join(ROOT, m1_sub) if ROOT else m1_sub
    m2_path = os.path.join(ROOT, m2_sub) if ROOT else m2_sub
    m1 = read_matrix(m1_path, clip_nonneg=True)
    m2 = read_matrix(m2_path, clip_nonneg=False)
    np.fill_diagonal(m1, 0)
    np.fill_diagonal(m2, 0)
    all_mat1.append(m1)
    all_mat2.append(m2)
    names.append(display_name)
    print(f"  Loaded: {display_name}")

_fold_env = os.environ.get("FOLD_ID", "").strip()
FOLDS_TO_RUN = [int(_fold_env)] if _fold_env != "" else list(range(N_FOLDS))
print(f"\nFolds to run: {FOLDS_TO_RUN}")

# ─────────────────────────────────────────────────────────────────────────────
# LOOCV
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
results = []

for i in FOLDS_TO_RUN:
    test_name = names[i]
    print(f"\n{'='*65}\n  FOLD {i+1}/{N_FOLDS}: {test_name}\n{'='*65}")

    train_m1 = [all_mat1[j] for j in range(N_FOLDS) if j != i]
    train_m2 = [all_mat2[j] for j in range(N_FOLDS) if j != i]

    print("  Building 10-D scalar-only features ...")
    t0 = time.time()
    X_train, y_train = build_dataset_scalars(train_m1, train_m2)
    print(f"  Feature extraction: {(time.time()-t0)/60:.1f} min  X={X_train.shape}")

    y_binary = (np.abs(y_train) > 1e-7).astype(int)

    # Standardise -- essential for both LogisticRegression convergence and
    # for RidgeCV's alpha grid to be meaningful across features of very
    # different scales (row_sum is raw and unnormalised; ranks are in [0,1]).
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print("  Training logistic classifier (class_weight=balanced) ...")
    clf = LogisticRegression(penalty="l2", class_weight="balanced",
                              solver="lbfgs", max_iter=1000, n_jobs=N_CORES)
    clf.fit(X_scaled, y_binary)

    X_reg = X_scaled[y_binary == 1]
    y_reg = y_train[y_binary == 1]
    print(f"  Training RidgeCV regressor on {len(X_reg):,} non-zero samples ...")
    reg = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
    reg.fit(X_reg, y_reg)
    print(f"  RidgeCV selected alpha = {reg.alpha_:.4g}")

    del X_train, X_scaled, X_reg, y_reg; gc.collect()

    print("  Predicting ...")
    t0 = time.time()
    prediction = predict_linear(all_mat1[i], scaler, clf, reg)
    print(f"  Inference: {(time.time()-t0)/60:.1f} min")

    mask   = np.triu(np.ones((SIZE, SIZE), dtype=bool), k=1)
    y_true = all_mat2[i][mask]
    y_pred = prediction[mask]

    r2     = float(r2_score(y_true, y_pred))
    p_corr = float(pearsonr(y_true, y_pred)[0])
    s_corr = float(spearmanr(y_true, y_pred)[0])
    mae    = float(mean_absolute_error(y_true, y_pred))

    print(f"  R2={r2:.4f}  Pearson={p_corr:.4f}  Spearman={s_corr:.4f}  MAE={mae:.4f}")

    results.append({
        "Folder": test_name, "FeatureSet": "scalars_linear",
        "R2": r2, "Pearson": p_corr, "Spearman": s_corr, "MAE": mae,
        "ridge_alpha": float(reg.alpha_),
    })
    del clf, reg, prediction; gc.collect()

df = pd.DataFrame(results)
print("\n" + "=" * 80)
print("LINEAR BASELINE (logistic + ridge, 10 scalar features) -- LOOCV SUMMARY")
print("=" * 80)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
if len(df) == N_FOLDS:
    print(f"\n  Mean R2       = {df['R2'].mean():.4f}")
    print(f"  Mean Pearson  = {df['Pearson'].mean():.4f}")
    print(f"  Mean Spearman = {df['Spearman'].mean():.4f}")

df.to_csv("xgb_v10_linear_scalars_loocv_results.csv", index=False, float_format="%.6f")
print("\n[Saved] xgb_v10_linear_scalars_loocv_results.csv")
print("""
Compare this file directly against xgb_v10_scalars_loocv_results.csv
(XGBoost on the identical 10-D feature set). The per-locus and mean gap
between the two is the evidence for or against "gradient boosting is doing
real nonlinear work on these features," independent of whether raw patch
pixels help.
""")
