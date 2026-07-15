"""
uncertainty_quantile_v10.py
============================
Purpose: the manuscript currently lists "no uncertainty estimates" as a
limitation for future work. That is exactly the kind of gap a JCTC reviewer
turns into a major-revision request rather than letting it slide, especially
for a surrogate meant to replace an expensive physical calculation -- readers
need to know when to trust a given predicted lambda_ij and when not to. This
script adds calibrated prediction intervals to the existing two-stage hurdle
model with a modest, well-understood addition: two extra XGBoost regressors
trained with a pinball (quantile) loss instead of squared error, run on the
SAME 131-D feature set and the SAME nonzero-support training pairs as the
point-estimate regressor in xgboost_hurdle_v10_cpu.py.

WHY PINBALL LOSS VIA A CUSTOM OBJECTIVE, NOT XGBoost's NATIVE
"reg:quantileerror"
-------------------------------------------------------------
Native quantile regression (objective="reg:quantileerror") was only added in
XGBoost 2.0. The manuscript's Methods section states XGBoost 1.7 was used, so
this script implements the pinball loss directly as a custom objective
(gradient, pseudo-Hessian) passed via the sklearn API's `obj=` argument, which
XGBoost 1.7 fully supports. If you upgrade to XGBoost >= 2.0 you can instead
just pass objective="reg:quantileerror" and quantile_alpha=tau directly to
XGBRegressor and skip the custom objective below; the rest of the script
(data handling, coverage evaluation) is unchanged either way.

METHOD
------
For a target quantile tau in (0, 1), the pinball loss is
    L_tau(y, yhat) = tau * (y - yhat)       if y >= yhat
                   = (tau - 1) * (y - yhat) if y <  yhat
Its gradient w.r.t. yhat is -tau or (1 - tau) depending on the sign of the
residual. The true second derivative is zero almost everywhere, which breaks
XGBoost's Newton step, so we use the standard pseudo-Hessian trick of setting
hess = 1 everywhere (equivalent to a gradient-descent step within XGBoost's
boosting framework for this objective; this is the widely used approach for
quantile regression with boosted trees, see e.g. scikit-learn's
GradientBoostingRegressor(loss="quantile") for the analogous non-XGBoost
implementation).

We train two additional regressors, at tau_lo and tau_hi (default 0.1 and
0.9, giving a nominal 80% interval), using IDENTICAL hyperparameters,
training data, and nonzero-support masking as the point-estimate ("median-ish"
squared-error) regressor already in xgboost_hurdle_v10_cpu.py. The classifier
stage is unchanged; a pair predicted zero by the classifier is reported with
a degenerate interval [0, 0], since there is no continuous magnitude to
bound. This mirrors the existing hurdle architecture rather than replacing
it, so the added uncertainty machinery slots directly into the existing
predict_hurdle() call pattern.

EVALUATION: CALIBRATION, NOT JUST SHARPNESS
--------------------------------------------
Reporting an interval is meaningless without checking it is actually
calibrated. For each held-out locus we report:
  - empirical coverage: the fraction of true, nonzero lambda_ij values that
    fall inside [q_lo_hat, q_hi_hat], which should be close to the nominal
    80% if the model is well calibrated (not merely precise);
  - mean and median interval width, as the accompanying sharpness metric
    (a model that always predicts [-1e6, 1e6] would have perfect coverage
    and be useless; width tells you whether the intervals are actually
    informative).
Both numbers, reported per locus AND pooled, are what should go in the
paper's new uncertainty-quantification subsection, not just the point
estimate correlations already reported.

Usage (same convention as the other V10 scripts):
  FOLD_ID=0  TAU_LO=0.1  TAU_HI=0.9   python uncertainty_quantile_v10.py
  (no env vars, all defaults)          python uncertainty_quantile_v10.py

Output: xgb_v10_uq_loocv_results.csv with columns
  Folder, R2, Pearson, Spearman, MAE (point estimate, identical definition
  to the production model), Coverage_nominal, Coverage_empirical,
  MeanIntervalWidth, MedianIntervalWidth
"""

import os
import sys
import gc
import time
import warnings
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION -- matches the production model (11x11 patch, 131-D)
# ─────────────────────────────────────────────────────────────────────────────

SIZE    = 1001
WINDOW  = 11
HALF    = WINDOW // 2
N_PATCH = WINDOW ** 2
CENTER  = HALF * WINDOW + HALF
N_FEATURES = N_PATCH + 10   # 131

TAU_LO = float(os.environ.get("TAU_LO", "0.1"))
TAU_HI = float(os.environ.get("TAU_HI", "0.9"))
NOMINAL_COVERAGE = TAU_HI - TAU_LO
assert 0.0 < TAU_LO < TAU_HI < 1.0

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
    """Custom XGBoost objective: pinball (quantile) loss with pseudo-Hessian=1.
    Passed via the sklearn API's `objective=` argument, which on the XGBoost
    version this was tested against (3.1.2) invokes the callable as
    obj(y_true, y_pred) with plain numpy arrays (sklearn convention), NOT the
    low-level Booster API's obj(preds, dtrain). If you are on an older
    XGBoost (e.g. 1.7, as cited in the manuscript's Methods) and this
    signature does not match, swap to obj(preds, dtrain) with
    y = dtrain.get_label() instead -- check with a one-line print inside
    _obj before trusting either version silently."""
    def _obj(y_true, y_pred):
        residual = y_true - y_pred
        grad = np.where(residual >= 0, -tau, 1.0 - tau).astype(np.float32)
        hess = np.ones_like(y_pred, dtype=np.float32)
        return grad, hess
    return _obj


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING / FEATURE EXTRACTION -- identical to xgboost_hurdle_v10_cpu.py
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


def predict_with_intervals(m1, clf, reg_point, reg_lo, reg_hi, size=SIZE):
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
            # enforce lo <= hi (quantile crossing can happen with independently
            # trained quantile models; sorting per-pair is the standard fix)
            lo[nz_idx] = np.minimum(lo_raw, hi_raw)
            hi[nz_idx] = np.maximum(lo_raw, hi_raw)
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
print(f"\nFolds to run: {FOLDS_TO_RUN}   nominal interval = [{TAU_LO}, {TAU_HI}]"
      f" -> {NOMINAL_COVERAGE*100:.0f}% coverage target")

np.random.seed(42)
results = []

for i in FOLDS_TO_RUN:
    test_name = names[i]
    print(f"\n{'='*65}\n  FOLD {i+1}/{N_FOLDS}: {test_name}\n{'='*65}")

    train_m1 = [all_mat1[j] for j in range(N_FOLDS) if j != i]
    train_m2 = [all_mat2[j] for j in range(N_FOLDS) if j != i]

    print(f"  Building {N_FEATURES}-D features ...")
    t0 = time.time()
    X_train, y_train = build_dataset(train_m1, train_m2)
    print(f"  Feature extraction: {(time.time()-t0)/60:.1f} min  X={X_train.shape}")

    y_binary = (np.abs(y_train) > 1e-7).astype(int)
    clf = xgb.XGBClassifier(**CLF_PARAMS)
    clf.fit(X_train, y_binary)

    X_reg = X_train[y_binary == 1]
    y_reg = y_train[y_binary == 1]

    print(f"  Training point-estimate regressor on {len(X_reg):,} samples ...")
    reg_point = xgb.XGBRegressor(**REG_PARAMS_BASE)
    reg_point.fit(X_reg, y_reg)

    print(f"  Training tau={TAU_LO} quantile regressor ...")
    reg_lo = xgb.XGBRegressor(**{**REG_PARAMS_BASE, "objective": pinball_objective(TAU_LO)})
    reg_lo.fit(X_reg, y_reg)

    print(f"  Training tau={TAU_HI} quantile regressor ...")
    reg_hi = xgb.XGBRegressor(**{**REG_PARAMS_BASE, "objective": pinball_objective(TAU_HI)})
    reg_hi.fit(X_reg, y_reg)

    del X_train, y_train, X_reg, y_reg; gc.collect()

    print("  Predicting with intervals ...")
    t0 = time.time()
    pred, lo, hi = predict_with_intervals(all_mat1[i], clf, reg_point, reg_lo, reg_hi)
    print(f"  Inference: {(time.time()-t0)/60:.1f} min")

    mask = np.triu(np.ones((SIZE, SIZE), dtype=bool), k=1)
    y_true = all_mat2[i][mask]
    y_pred = pred[mask]
    y_lo, y_hi = lo[mask], hi[mask]

    r2     = float(r2_score(y_true, y_pred))
    p_corr = float(pearsonr(y_true, y_pred)[0])
    s_corr = float(spearmanr(y_true, y_pred)[0])
    mae    = float(mean_absolute_error(y_true, y_pred))

    # Calibration is only meaningful over pairs the model itself thinks are
    # nonzero (zero-classified pairs get a degenerate [0,0] interval and
    # would trivially inflate coverage if included with true-nonzero targets
    # incorrectly routed to a zero interval).
    nz_pred_mask = (np.abs(y_lo) > 0) | (np.abs(y_hi) > 0)
    if nz_pred_mask.sum() > 0:
        covered  = (y_true[nz_pred_mask] >= y_lo[nz_pred_mask]) & (y_true[nz_pred_mask] <= y_hi[nz_pred_mask])
        coverage = float(covered.mean())
        widths   = (y_hi[nz_pred_mask] - y_lo[nz_pred_mask])
        mean_w, med_w = float(widths.mean()), float(np.median(widths))
    else:
        coverage, mean_w, med_w = float("nan"), float("nan"), float("nan")

    print(f"  R2={r2:.4f}  Pearson={p_corr:.4f}  Spearman={s_corr:.4f}  MAE={mae:.4f}")
    print(f"  Empirical coverage={coverage:.4f}  (nominal={NOMINAL_COVERAGE:.2f})"
          f"  mean_width={mean_w:.2f}  median_width={med_w:.2f}")

    results.append({
        "Folder": test_name, "R2": r2, "Pearson": p_corr, "Spearman": s_corr, "MAE": mae,
        "Coverage_nominal": NOMINAL_COVERAGE, "Coverage_empirical": coverage,
        "MeanIntervalWidth": mean_w, "MedianIntervalWidth": med_w,
    })
    del clf, reg_point, reg_lo, reg_hi, pred, lo, hi; gc.collect()

df = pd.DataFrame(results)
print("\n" + "=" * 90)
print(f"UNCERTAINTY QUANTIFICATION -- LOOCV SUMMARY  (nominal {NOMINAL_COVERAGE*100:.0f}% interval)")
print("=" * 90)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
if len(df) == N_FOLDS:
    print(f"\n  Mean empirical coverage = {df['Coverage_empirical'].mean():.4f}"
          f"  (target {NOMINAL_COVERAGE:.2f})")
    print(f"  Mean interval width     = {df['MeanIntervalWidth'].mean():.2f}")

df.to_csv("xgb_v10_uq_loocv_results.csv", index=False, float_format="%.6f")
print("\n[Saved] xgb_v10_uq_loocv_results.csv")
print("""
If mean empirical coverage is well below the nominal target across most
loci, the intervals are overconfident (too narrow) and should be widened,
e.g. via conformal calibration on a held-out validation split before
reporting in the paper. If it is well above nominal, the intervals are
conservative (safe, but less informative) -- report both the coverage AND
the width, not coverage alone, so a reviewer can judge whether the interval
is actually useful rather than trivially wide.
""")
