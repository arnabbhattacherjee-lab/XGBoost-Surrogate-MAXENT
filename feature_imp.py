import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "xgb_v10_reg.json"   # or "xgb_v10_clf.json"
IMPORTANCE_TYPE = "gain"          # "gain", "weight", "cover", "total_gain", "total_cover"
TOP_N = 25                        # show top N features only (set None for all 131)
FONT_SIZE = 25

# ==============================
# FEATURE NAMES
# ==============================
# Build readable names for your 131-D feature vector.
# Adjust this to match the EXACT order your feature-builder produces.
feature_names = []

# 11x11 reflect-padded contact patch (121 features)
for i in range(-5, 6):
    for j in range(-5, 6):
        feature_names.append(f"patch_r{i}_c{j}")

# remaining 10 engineered features, IN THIS EXACT ORDER (from xgboost_hurdle_v10_cpu11.py):
feature_names += [
    "norm_dist",            # [121] abs(r-c) / (SIZE-1)
    "row_sum",               # [122] raw row marginal
    "col_sum",               # [123] raw col marginal
    "row_std",                # [124] std of row r
    "col_std",                # [125] std of col c
    "oe_ratio",               # [126] contact[r,c] / diag_mean[|r-c|]
    "decay_alpha",            # [127] slope of log P(s) vs log s per locus
    "contact_sparsity",       # [128] fraction of upper-tri contacts that are 0
    "center_rank_in_patch",   # [129] rank of centre log1p in its 11x11 nbhd
    "oe_rank_in_diag",        # [130] rank of O/E within diagonal band
]

assert len(feature_names) == 131, f"Got {len(feature_names)} names, expected 131 — fix the list above"

# ==============================
# LOAD MODEL & GET IMPORTANCE
# ==============================
model = xgb.XGBRegressor()   # use xgb.XGBClassifier() if loading the classifier
model.load_model(MODEL_PATH)

booster = model.get_booster()
booster.feature_names = feature_names

score_dict = booster.get_score(importance_type=IMPORTANCE_TYPE)

# fill in zero for any feature never used in a split
importances = np.array([score_dict.get(f, 0.0) for f in feature_names])

# ==============================
# SORT & SELECT TOP N
# ==============================
order = np.argsort(importances)[::-1]
if TOP_N is not None:
    order = order[:TOP_N]

sorted_names = [feature_names[i] for i in order]
sorted_scores = importances[order]

# ==============================
# PLOT (horizontal bar, summary style)
# ==============================
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE - 4,
})

fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(sorted_names))))

y_pos = np.arange(len(sorted_names))
ax.barh(y_pos, sorted_scores, color="tab:blue")

ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names)
ax.invert_yaxis()  # most important feature at top

ax.set_xlabel(f"Importance ({IMPORTANCE_TYPE})")
ax.set_title("XGBoost Feature Importance")

plt.tight_layout()
plt.savefig("feature_importance.pdf", dpi=300)
plt.show()
