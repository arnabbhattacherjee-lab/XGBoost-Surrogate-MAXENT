"""
conformal_calibration_v10.py
==============================
Purpose: uncertainty_quantile_v10.py's raw pinball-loss intervals came back
badly miscalibrated on the real data (empirical coverage 1.5-6.4% against an
80% nominal target, across all 12 LOOCV loci -- see
xgb_v10_uq_loocv_results.csv). This script fixes that with split conformal
calibration, specifically Conformalized Quantile Regression (CQR: Romano,
Patterson & Candes, "Conformalized Quantile Regression", NeurIPS 2019),
which is the standard, distribution-free way to take an uncalibrated
quantile-regression interval and rescale it to hit a target coverage level,
using only a held-out calibration set and no assumptions about *why* the raw
interval was miscalibrated.

WHY A LOCUS-LEVEL CALIBRATION SPLIT, NOT A PAIR-LEVEL ONE
------------------------------------------------------------
Every other train/test split in this project (LOOCV itself, hyperparameter
selection) is done at the LOCUS level, never by randomly splitting pairs,
because pairs within the same locus are highly non-independent (neighbouring
patches overlap almost entirely -- see the manuscript's discussion of local
continuity in the classifier section) and randomly mixing pairs from the same
locus into both "train" and "calibration" would leak locus-specific structure
into the calibration set, giving an optimistic and misleading coverage
estimate. This script therefore reserves a subset of the 11 TRAINING loci
per LOOCV fold purely for calibration, fits the classifier and quantile
regressors on the remaining loci only, calibrates on the reserved loci, and
then evaluates on the true held-out test locus -- exactly the same
locus-disjoint structure as the rest of the paper.

CAVEAT THIS SCRIPT DOES NOT HIDE
---------------------------------
Conformal prediction's coverage guarantee formally requires the calibration
data and the test point to be exchangeable (drawn from the same
distribution). Here, calibration loci and the held-out test locus are
different genomic loci, possibly in different cell types, so exchangeability
is an approximation, not a proof. Report the empirical coverage actually
observed on the held-out locus (this script computes it directly), not just
the nominal target, and say so explicitly in the paper -- conformal
calibration will get you much closer to nominal coverage than the raw
pinball intervals, but it is not a formal guarantee under locus-level
distribution shift.

METHOD (CQR)
------------
For each LOOCV fold with held-out test locus L:
  1. Split the 11 training loci into a FIT set (used to train the
     classifier, point regressor, and the tau_lo/tau_hi quantile
     regressors -- identical models to uncertainty_quantile_v10.py) and a
     CALIBRATION set (N_CAL_LOCI loci, default 2, reserved and NEVER used
     for model fitting).
  2. On the calibration loci, restrict to pairs the FIT classifier predicts
     as non-zero (matching the population the interval will actually be
     applied to at deployment), and compute the CQR nonconformity score for
     each:
         score_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))
     A positive score means the true value fell outside the raw interval;
     a negative score means it was inside, by that margin.
  3. Compute Q = the finite-sample-corrected (1-alpha) quantile of the
     calibration scores, using the ceil((n_cal+1)(1-alpha))/n_cal level
     (Romano et al.'s correction for exact finite-sample marginal coverage
     in the exchangeable case).
  4. On the held-out TEST locus, widen (or narrow, if Q<0) every raw
     interval by Q on each side:
         [q_lo(x) - Q,  q_hi(x) + Q]
     and report empirical coverage and interval width using these adjusted
     bounds, exactly as uncertainty_quantile_v10.py did for the raw ones.

Usage: identical convention to the other V10 scripts.
  FOLD_ID=0  N_CAL_LOCI=2  ALPHA=0.2   python conformal_calibration_v10.py
  (no env vars, all defaults, all 12 folds)   python conformal_calibration_v10.py

Output: xgb_v10_conformal_loocv_results.csv, with both the RAW (pre-conformal)
and CALIBRATED coverage/width side by side per locus, so you can quote the
before/after improvement directly in the paper.
"""

import os
import sys
import gc
import time
import warnings
import math
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION -- matches uncertainty_quantile_v10.py (11x11 patch, 131-D)
# ─────────────────────────────────────────────────────────────────────────────

SIZE    = 1001
WINDOW  = 11
HALF    = WINDOW // 2
N_PATCH = WINDOW ** 2
CENTER  = HALF * WINDOW + HALF
N_FEATURES = N_PATCH + 10   # 131

TAU_LO = float(os.environ.get("TAU_LO", "0.1"))
TAU_HI = float(os.environ.get("TAU_HI", "0.9"))
ALPHA  = float(os.environ.get("ALPHA", str(1.0 - (TAU_HI - TAU_LO))))  # 0.2 by default
NOMINAL_COVERAGE = 1.0 - ALPHA
N_CAL_LOCI = int(os.environ.get("N_CAL_LOCI", "2"))
assert 0.0 < ALPHA < 1.0

N_CORES = os.cpu_count() or 8
ROOT    = ""

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

CLF_PARAMS = dict(tree_method="hist", device="cpu", n_estimators=350, max_depth=6,
                   learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                   min_child_weight=10, n_jobs=N_CORES)
REG_PARAMS_BASE = dict(tree_method="hist", device="cpu", n_estimators=600, max_depth=8,
                        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=5, n_jobs=N_CORES)


def pinball_objective(tau):
    """Same custom pinball objective as uncertainty_quantile_v10.py, using the
    sklearn-API (y_true, y_pred) calling convention -- verify this against
    obj(preds, dtrain) if you're on an older XGBoost; see that script's
    docstring for the one-line check."""
    def _obj(y_true, y_pred):
        residual = y_true - y_pred
        grad = np.where(residual >= 0, -tau, 1.0 - tau).astype(np.float32)
        hess = np.ones_like(y_pred, dtype=np.float32)
        return grad, hess
    return _obj


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING / FEATURE EXTRACTION -- identical to the other V10 scripts
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
    log_s, log_ps = np.log(s[valid]), np.log(ps[valid])
    weights = np.sqrt((size - s[valid]).astype(np.float64))
    alpha, _ = np.polyfit(log_s, log_ps, 1, w=weights)
    return float(alpha)


def compute_contact_sparsity(m1, size=SIZE):
    r_idx, c_idx = np.triu_indices(size, k=1)
    return float((m1[r_idx, c_idx] == 0.0).mean())


def compute_oe_diag_ranks(m1, diag_means, size=SIZE):
    oe_ranks = np.zeros((size, size), dtype=np.float32)
    for d in range(1, size):
        r_idx = np.arange(0, size - d); c_idx = r_idx + d
        raw = m1[r_idx, c_idx].astype(np.float64)
        exp = float(diag_means[d]) if diag_means[d] > 1e-9 else 1e-9
        oe_d = (raw / exp).astype(np.float32)
        n = len(oe_d)
        if n > 1:
            order = np.argsort(oe_d)
            ranks = np.empty(n, dtype=np.float32)
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


def _extract_row(r, view_log, raw_contact_r, row_sums, col_sums, row_stds, col_stds,
                  diag_means, decay_alpha, contact_sparsity, oe_diag_ranks, m2, size=SIZE):
    c_range = np.arange(r, size, dtype=np.int64)
    nc = len(c_range)
    patches_log = view_log[r, c_range].reshape(nc, -1).astype(np.float32)
    dist_raw  = (c_range - r).astype(np.float32)
    dist_norm = dist_raw / (size - 1)
    rs, cs   = np.full(nc, row_sums[r], dtype=np.float32), col_sums[c_range].astype(np.float32)
    rstd, cstd = np.full(nc, row_stds[r], dtype=np.float32), col_stds[c_range].astype(np.float32)
    raw_vals = raw_contact_r[c_range].astype(np.float32)
    dist_int = dist_raw.astype(np.int32)
    exp_vals = diag_means[dist_int]
    oe = np.where(exp_vals > 1e-9, raw_vals / exp_vals, 0.0).astype(np.float32)
    alpha_col = np.full(nc, decay_alpha, dtype=np.float32)
    spar_col  = np.full(nc, contact_sparsity, dtype=np.float32)
    centre_vals = patches_log[:, CENTER]
    centre_rank = ((patches_log < centre_vals[:, None]).sum(axis=1).astype(np.float32)) / (N_PATCH - 1)
    oe_diag_r = oe_diag_ranks[r, c_range].astype(np.float32)
    X_row = np.concatenate([patches_log, dist_norm[:, None], rs[:, None], cs[:, None],
                             rstd[:, None], cstd[:, None], oe[:, None], alpha_col[:, None],
                             spar_col[:, None], centre_rank[:, None], oe_diag_r[:, None]], axis=1)
    y_row = m2[r, c_range].astype(np.float32)
    return X_row, y_row


def _extract_locus(m1, m2, size=SIZE):
    view_log = _build_view_log(m1)
    row_sums, col_sums = m1.sum(axis=1).astype(np.float32), m1.sum(axis=0).astype(np.float32)
    row_stds, col_stds = m1.std(axis=1).astype(np.float32), m1.std(axis=0).astype(np.float32)
    diag_means  = compute_diagonal_means(m1, size)
    decay_alpha = compute_decay_exponent(diag_means, size)
    contact_sp  = compute_contact_sparsity(m1, size)
    oe_diag_ranks = compute_oe_diag_ranks(m1, diag_means, size)
    results = Parallel(n_jobs=N_CORES, prefer="threads")(
        delayed(_extract_row)(r, view_log, m1[r], row_sums, col_sums, row_stds, col_stds,
                               diag_means, decay_alpha, contact_sp, oe_diag_ranks, m2, size)
        for r in range(size))
    X = np.concatenate([rx[0] for rx in results], axis=0)
    y = np.concatenate([rx[1] for rx in results], axis=0)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def build_dataset(m1_list, m2_list):
    n_upper = (SIZE * (SIZE + 1)) // 2
    total = n_upper * len(m1_list)
    X = np.empty((total, N_FEATURES), dtype=np.float32)
    y = np.empty(total, dtype=np.float32)
    start = 0
    for k, (m1, m2) in enumerate(zip(m1_list, m2_list)):
        t0 = time.time()
        Xl, yl = _extract_locus(m1, m2)
        X[start:start + n_upper] = Xl
        y[start:start + n_upper] = yl
        start += n_upper
        del Xl, yl; gc.collect()
        print(f"    locus {k+1}/{len(m1_list)} done in {time.time()-t0:.1f}s")
    return X, y


def predict_with_intervals(m1, clf, reg_point, reg_lo, reg_hi, size=SIZE, Q=0.0):
    """Same as uncertainty_quantile_v10.py's predict_with_intervals, but with
    an extra conformal correction Q applied symmetrically: lo -= Q, hi += Q
    (Q may be negative, which narrows the interval)."""
    view_log = _build_view_log(m1)
    row_sums, col_sums = m1.sum(axis=1).astype(np.float32), m1.sum(axis=0).astype(np.float32)
    row_stds, col_stds = m1.std(axis=1).astype(np.float32), m1.std(axis=0).astype(np.float32)
    diag_means  = compute_diagonal_means(m1, size)
    decay_alpha = compute_decay_exponent(diag_means, size)
    contact_sp  = compute_contact_sparsity(m1, size)
    oe_diag_ranks = compute_oe_diag_ranks(m1, diag_means, size)
    dummy_m2 = np.zeros((size, size), dtype=np.float32)

    full_pred = np.zeros((size, size), dtype=np.float32)
    full_lo   = np.zeros((size, size), dtype=np.float32)
    full_hi   = np.zeros((size, size), dtype=np.float32)

    BATCH_ROWS = max(32, size // max(1, N_CORES // 4))
    for batch_start in range(0, size, BATCH_ROWS):
        batch_rows = list(range(batch_start, min(batch_start + BATCH_ROWS, size)))
        rows = Parallel(n_jobs=N_CORES, prefer="threads")(
            delayed(_extract_row)(r, view_log, m1[r], row_sums, col_sums, row_stds, col_stds,
                                   diag_means, decay_alpha, contact_sp, oe_diag_ranks, dummy_m2, size)
            for r in batch_rows)
        X_batch = np.concatenate([rx[0] for rx in rows], axis=0)
        row_lengths = [len(rx[1]) for rx in rows]
        del rows
        np.nan_to_num(X_batch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        is_nz = clf.predict(X_batch)
        preds = np.zeros(len(X_batch), dtype=np.float32)
        lo    = np.zeros(len(X_batch), dtype=np.float32)
        hi    = np.zeros(len(X_batch), dtype=np.float32)
        nz_idx = np.where(is_nz == 1)[0]
        if len(nz_idx) > 0:
            Xnz = X_batch[nz_idx]
            preds[nz_idx] = np.nan_to_num(reg_point.predict(Xnz).astype(np.float32), nan=0.0)
            lo_raw = reg_lo.predict(Xnz).astype(np.float32)
            hi_raw = reg_hi.predict(Xnz).astype(np.float32)
            lo_sorted = np.minimum(lo_raw, hi_raw)
            hi_sorted = np.maximum(lo_raw, hi_raw)
            # conformal widening/narrowing
            lo[nz_idx] = lo_sorted - Q
            hi[nz_idx] = hi_sorted + Q
        del X_batch

        offset = 0
        for r, nc in zip(batch_rows, row_lengths):
            c_range = np.arange(r, size)
            full_pred[r, c_range] = preds[offset:offset + nc]; full_pred[c_range, r] = preds[offset:offset + nc]
            full_lo[r, c_range]   = lo[offset:offset + nc];    full_lo[c_range, r]   = lo[offset:offset + nc]
            full_hi[r, c_range]   = hi[offset:offset + nc];    full_hi[c_range, r]   = hi[offset:offset + nc]
            offset += nc

    for arr in (full_pred, full_lo, full_hi):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return full_pred, full_lo, full_hi


def compute_cqr_scores_on_calibration_loci(cal_m1_list, cal_m2_list, clf, reg_lo, reg_hi):
    """For each calibration locus, restrict to pairs the FIT classifier calls
    non-zero, and compute the CQR nonconformity score
        score = max(q_lo(x) - y, y - q_hi(x))
    Returns a flat array of scores pooled across all calibration loci."""
    all_scores = []
    for m1, m2 in zip(cal_m1_list, cal_m2_list):
        view_log = _build_view_log(m1)
        row_sums, col_sums = m1.sum(axis=1).astype(np.float32), m1.sum(axis=0).astype(np.float32)
        row_stds, col_stds = m1.std(axis=1).astype(np.float32), m1.std(axis=0).astype(np.float32)
        diag_means  = compute_diagonal_means(m1, SIZE)
        decay_alpha = compute_decay_exponent(diag_means, SIZE)
        contact_sp  = compute_contact_sparsity(m1, SIZE)
        oe_diag_ranks = compute_oe_diag_ranks(m1, diag_means, SIZE)

        rows = Parallel(n_jobs=N_CORES, prefer="threads")(
            delayed(_extract_row)(r, view_log, m1[r], row_sums, col_sums, row_stds, col_stds,
                                   diag_means, decay_alpha, contact_sp, oe_diag_ranks, m2, SIZE)
            for r in range(SIZE))
        X = np.concatenate([rx[0] for rx in rows], axis=0)
        y = np.concatenate([rx[1] for rx in rows], axis=0)
        del rows
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        is_nz = clf.predict(X)
        nz_idx = np.where(is_nz == 1)[0]
        if len(nz_idx) == 0:
            continue
        Xnz, ynz = X[nz_idx], y[nz_idx]
        lo_raw = reg_lo.predict(Xnz).astype(np.float32)
        hi_raw = reg_hi.predict(Xnz).astype(np.float32)
        lo_sorted = np.minimum(lo_raw, hi_raw)
        hi_sorted = np.maximum(lo_raw, hi_raw)
        scores = np.maximum(lo_sorted - ynz, ynz - hi_sorted)
        all_scores.append(scores)
        del X, y, Xnz, ynz
        gc.collect()

    if len(all_scores) == 0:
        return np.array([0.0], dtype=np.float32)
    return np.concatenate(all_scores)


def conformal_quantile(scores, alpha):
    """Finite-sample-corrected (1-alpha) quantile of the calibration scores
    (Romano, Patterson & Candes 2019, Eq. for split-conformal correction)."""
    n = len(scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level))


# ─────────────────────────────────────────────────────────────────────────────
# LOAD, LOOCV
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading all matrices ...")
all_mat1, all_mat2, names = [], [], []
for display_name, m1_sub, m2_sub in TARGET_CONFIGS:
    m1 = read_matrix(os.path.join(ROOT, m1_sub) if ROOT else m1_sub, clip_nonneg=True)
    m2 = read_matrix(os.path.join(ROOT, m2_sub) if ROOT else m2_sub, clip_nonneg=False)
    np.fill_diagonal(m1, 0); np.fill_diagonal(m2, 0)
    all_mat1.append(m1); all_mat2.append(m2); names.append(display_name)
    print(f"  Loaded: {display_name}")

_fold_env = os.environ.get("FOLD_ID", "").strip()
FOLDS_TO_RUN = [int(_fold_env)] if _fold_env != "" else list(range(N_FOLDS))
print(f"\nFolds to run: {FOLDS_TO_RUN}   nominal = {NOMINAL_COVERAGE*100:.0f}%   "
      f"N_CAL_LOCI = {N_CAL_LOCI}")

np.random.seed(42)
results = []

for i in FOLDS_TO_RUN:
    test_name = names[i]
    print(f"\n{'='*65}\n  FOLD {i+1}/{N_FOLDS}: {test_name}\n{'='*65}")

    train_idx = [j for j in range(N_FOLDS) if j != i]
    if len(train_idx) <= N_CAL_LOCI:
        print(f"  [SKIP] Not enough training loci to reserve {N_CAL_LOCI} for calibration.")
        continue

    # Deterministic split: last N_CAL_LOCI training loci (in registry order,
    # excluding the held-out test locus) go to calibration, the rest to fit.
    cal_idx = train_idx[-N_CAL_LOCI:]
    fit_idx = train_idx[:-N_CAL_LOCI]
    print(f"  Fit loci ({len(fit_idx)}): {[names[j] for j in fit_idx]}")
    print(f"  Calibration loci ({len(cal_idx)}): {[names[j] for j in cal_idx]}")

    fit_m1 = [all_mat1[j] for j in fit_idx]
    fit_m2 = [all_mat2[j] for j in fit_idx]

    print(f"  Building {N_FEATURES}-D features on fit set ...")
    t0 = time.time()
    X_train, y_train = build_dataset(fit_m1, fit_m2)
    print(f"  Feature extraction: {(time.time()-t0)/60:.1f} min  X={X_train.shape}")

    y_binary = (np.abs(y_train) > 1e-7).astype(int)
    clf = xgb.XGBClassifier(**CLF_PARAMS)
    clf.fit(X_train, y_binary)

    X_reg = X_train[y_binary == 1]
    y_reg = y_train[y_binary == 1]

    reg_point = xgb.XGBRegressor(**REG_PARAMS_BASE)
    reg_point.fit(X_reg, y_reg)
    reg_lo = xgb.XGBRegressor(**{**REG_PARAMS_BASE, "objective": pinball_objective(TAU_LO)})
    reg_lo.fit(X_reg, y_reg)
    reg_hi = xgb.XGBRegressor(**{**REG_PARAMS_BASE, "objective": pinball_objective(TAU_HI)})
    reg_hi.fit(X_reg, y_reg)

    del X_train, y_train, X_reg, y_reg; gc.collect()

    print("  Computing CQR nonconformity scores on calibration loci ...")
    cal_m1 = [all_mat1[j] for j in cal_idx]
    cal_m2 = [all_mat2[j] for j in cal_idx]
    scores = compute_cqr_scores_on_calibration_loci(cal_m1, cal_m2, clf, reg_lo, reg_hi)
    Q = conformal_quantile(scores, ALPHA)
    print(f"  n_calibration_pairs={len(scores)}  Q={Q:.4f}")

    print("  Predicting on held-out test locus (raw and conformal) ...")
    pred, lo_raw, hi_raw = predict_with_intervals(all_mat1[i], clf, reg_point, reg_lo, reg_hi, Q=0.0)
    _, lo_cal, hi_cal   = predict_with_intervals(all_mat1[i], clf, reg_point, reg_lo, reg_hi, Q=Q)

    mask = np.triu(np.ones((SIZE, SIZE), dtype=bool), k=1)
    y_true = all_mat2[i][mask]
    y_pred = pred[mask]

    r2     = float(r2_score(y_true, y_pred))
    p_corr = float(pearsonr(y_true, y_pred)[0])
    s_corr = float(spearmanr(y_true, y_pred)[0])
    mae    = float(mean_absolute_error(y_true, y_pred))

    def coverage_and_width(lo_full, hi_full):
        y_lo, y_hi = lo_full[mask], hi_full[mask]
        nz_mask = (np.abs(y_lo) > 0) | (np.abs(y_hi) > 0)
        if nz_mask.sum() == 0:
            return float("nan"), float("nan"), float("nan")
        covered = (y_true[nz_mask] >= y_lo[nz_mask]) & (y_true[nz_mask] <= y_hi[nz_mask])
        widths  = y_hi[nz_mask] - y_lo[nz_mask]
        return float(covered.mean()), float(widths.mean()), float(np.median(widths))

    cov_raw, w_raw_mean, w_raw_med = coverage_and_width(lo_raw, hi_raw)
    cov_cal, w_cal_mean, w_cal_med = coverage_and_width(lo_cal, hi_cal)

    print(f"  R2={r2:.4f}  Pearson={p_corr:.4f}  Spearman={s_corr:.4f}  MAE={mae:.4f}")
    print(f"  RAW        coverage={cov_raw:.4f}  mean_width={w_raw_mean:.2f}")
    print(f"  CONFORMAL  coverage={cov_cal:.4f}  mean_width={w_cal_mean:.2f}  (Q={Q:.2f})")

    results.append({
        "Folder": test_name, "R2": r2, "Pearson": p_corr, "Spearman": s_corr, "MAE": mae,
        "Coverage_nominal": NOMINAL_COVERAGE,
        "Coverage_raw": cov_raw, "MeanWidth_raw": w_raw_mean, "MedianWidth_raw": w_raw_med,
        "Coverage_conformal": cov_cal, "MeanWidth_conformal": w_cal_mean, "MedianWidth_conformal": w_cal_med,
        "Q": Q, "N_calibration_pairs": len(scores),
    })
    del clf, reg_point, reg_lo, reg_hi, pred, lo_raw, hi_raw, lo_cal, hi_cal; gc.collect()

df = pd.DataFrame(results)
print("\n" + "=" * 100)
print(f"CONFORMAL-CALIBRATED UNCERTAINTY -- LOOCV SUMMARY  (nominal {NOMINAL_COVERAGE*100:.0f}%)")
print("=" * 100)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
if len(df) == N_FOLDS:
    print(f"\n  Mean coverage RAW       = {df['Coverage_raw'].mean():.4f}")
    print(f"  Mean coverage CONFORMAL = {df['Coverage_conformal'].mean():.4f}  (target {NOMINAL_COVERAGE:.2f})")
    print(f"  Mean width RAW          = {df['MeanWidth_raw'].mean():.2f}")
    print(f"  Mean width CONFORMAL    = {df['MeanWidth_conformal'].mean():.2f}")

df.to_csv("xgb_v10_conformal_loocv_results.csv", index=False, float_format="%.6f")
print("\n[Saved] xgb_v10_conformal_loocv_results.csv")
print("""
Report Coverage_conformal (not Coverage_raw) as your headline uncertainty
result, alongside MeanWidth_conformal so a reader can judge informativeness,
not just calibration. If Coverage_conformal is still noticeably off nominal
for specific loci (most likely the loci most different from the bulk of
training data -- check K562 nanog and K562 tal1 first, given their behaviour
elsewhere in this study), say so explicitly rather than only reporting the
mean across folds: that per-locus breakdown is exactly the kind of honesty
that heads off a reviewer's next question.
""")
