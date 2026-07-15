#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Input files
# =====================================================
FONT_SIZE = 25 
files = {
    "Full":     "xgb_v10_loocv_results.csv",
    "Patch":    "xgb_v10_patch_loocv_results.csv",
    "Scalars":  "xgb_v10_scalars_loocv_results.csv",
    "Global8":  "xgb_v10_global8_loocv_results.csv"
}

# =====================================================
# Read all files
# =====================================================

dfs = []

for feature, file in files.items():
    df = pd.read_csv(file)
    df["FeatureSet"] = feature
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# =====================================================
# Order
# =====================================================

order = ["Full", "Patch", "Scalars", "Global8"]

df["FeatureSet"] = pd.Categorical(
    df["FeatureSet"],
    categories=order,
    ordered=True
)
#print(df.head())
#print(df.tail())
# =====================================================
# Compute mean and std
# =====================================================

stats = (
    df.groupby("FeatureSet")[["Pearson", "Spearman", "R2"]]
      .agg(["mean", "std"])
)

pearson_mean  = stats["Pearson"]["mean"]
pearson_std   = stats["Pearson"]["std"]

spearman_mean = stats["Spearman"]["mean"]
spearman_std  = stats["Spearman"]["std"]

r2_mean       = stats["R2"]["mean"]
r2_std        = stats["R2"]["std"]
#print(stats)
# =====================================================
# Plot
# =====================================================

x = np.arange(len(order))
width = 0.25

plt.figure(figsize=(10,8))

colors = ["#4C72B0", "#55A868", "#C44E52"]

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE - 4,
})

plt.bar(
    x - width,
    pearson_mean,
    width,
    yerr=pearson_std,
    capsize=5,
    color=colors[0],
    label="Pearson"
)

plt.bar(
    x,
    spearman_mean,
    width,
    yerr=spearman_std,
    capsize=5,
    color=colors[1],
    label="Spearman"
)

plt.bar(
    x + width,
    r2_mean,
    width,
    yerr=r2_std,
    capsize=5,
    color=colors[2],
    label=r"$R^2$"
)

plt.xticks(x, order, fontsize=25)
plt.ylabel("Mean Score", fontsize=25)
plt.xlabel("Feature Set", fontsize=25)

plt.ylim(0, 1.15)

plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(frameon=True)

plt.tight_layout()

plt.savefig("feature_ablation_comparison.pdf")


plt.show()
