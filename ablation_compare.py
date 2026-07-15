#!/usr/bin/env python3
"""
ablation_compare.py
====================
Merges the three LOOCV result CSVs produced by
xgboost_hurdle_v10_feature_ablation.py (run once per FEATURE_SET) and reports
per-locus and mean Delta-r / Delta-R^2 between the full 235-D model and each
reduced feature set.

Expects, in the current directory:
  xgb_v10_loocv_results.csv          (FEATURE_SET=full)
  xgb_v10_scalars_loocv_results.csv  (FEATURE_SET=scalars)
  xgb_v10_patch_loocv_results.csv    (FEATURE_SET=patch)

Usage:
  python ablation_compare.py
"""

import os
import sys
import pandas as pd

FULL_CSV    = "xgb_v10_loocv_results.csv"
SCALARS_CSV = "xgb_v10_scalars_loocv_results.csv"
PATCH_CSV   = "xgb_v10_patch_loocv_results.csv"


def _load(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[MISSING] {path} -- run FEATURE_SET=... "
              f"python xgboost_hurdle_v10_feature_ablation.py first.")
        return None
    df = pd.read_csv(path)
    keep = [c for c in ["Folder", "R2", "Pearson", "Spearman", "MAE"] if c in df.columns]
    return df[keep].rename(columns={c: f"{c}_{label}" for c in keep if c != "Folder"})


def main() -> int:
    full    = _load(FULL_CSV,    "full")
    scalars = _load(SCALARS_CSV, "scalars")
    patch   = _load(PATCH_CSV,   "patch")

    have = [d for d in (full, scalars, patch) if d is not None]
    if len(have) < 2:
        print("\nNeed at least two of the three result files to compare. Exiting.")
        return 1

    merged = have[0]
    for d in have[1:]:
        merged = merged.merge(d, on="Folder", how="inner")

    print("\n" + "=" * 100)
    print("PER-LOCUS COMPARISON")
    print("=" * 100)
    print(merged.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 100)
    print("MEAN METRICS")
    print("=" * 100)
    metric_cols = [c for c in merged.columns if c != "Folder"]
    means = merged[metric_cols].mean(numeric_only=True)
    print(means.to_string(float_format=lambda x: f"{x:.4f}"))

    if full is not None and scalars is not None:
        d_r  = means.get("Pearson_full", float("nan")) - means.get("Pearson_scalars", float("nan"))
        d_r2 = means.get("R2_full", float("nan"))      - means.get("R2_scalars", float("nan"))
        print(f"\nDelta-r  (full - scalars-only) = {d_r:.4f}")
        print(f"Delta-R2 (full - scalars-only) = {d_r2:.4f}")
        frac = means.get("Pearson_scalars", float("nan")) / means.get("Pearson_full", float("nan")) \
            if means.get("Pearson_full", 0) else float("nan")
        print(f"Scalars-only recovers {100*frac:.1f}% of the full model's mean Pearson r "
              f"(if this is high, e.g. >80-90%, the 'local geometry' claim needs softening)")

    if full is not None and patch is not None:
        d_r  = means.get("Pearson_full", float("nan")) - means.get("Pearson_patch", float("nan"))
        d_r2 = means.get("R2_full", float("nan"))      - means.get("R2_patch", float("nan"))
        print(f"\nDelta-r  (full - patch-only)   = {d_r:.4f}")
        print(f"Delta-R2 (full - patch-only)   = {d_r2:.4f}")

    out_path = "ablation_comparison_summary.csv"
    merged.to_csv(out_path, index=False)
    print(f"\n[Saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
