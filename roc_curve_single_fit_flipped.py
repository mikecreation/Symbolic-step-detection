# ===================================================================
#  roc_curve_residual_pysr.py
#
#  Monte-Carlo ROC for symbolic-regression residual test
#  n = 40   σ = 0.08   B = 1000
#  Author: you  |  July 2024
# ===================================================================

import os, tempfile
import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from scipy.stats import median_abs_deviation, ttest_ind
from sklearn.metrics import roc_curve, auc

# -------------------------------------------------------------------
#  SETTINGS
# -------------------------------------------------------------------
np.random.seed(2024)

n, sigma = 40, 0.08
t        = np.linspace(0, (n - 1) * 0.2, n)
deltas   = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01]   # plot largest→smallest
B        = 1000                                         # raise to 3000 for paper

# -------------------------------------------------------------------
#  SYNTHETIC GENERATOR
# -------------------------------------------------------------------
def simulate_series(delta: float) -> np.ndarray:
    y = np.where(t < 2.0, 1.0, 1.0 + delta)
    return y + np.random.normal(0, sigma, size=n)

# -------------------------------------------------------------------
#  FIT SYMBOLIC REGRESSION ON NULL ONLY
# -------------------------------------------------------------------
base_series = simulate_series(0.0)

model = PySRRegressor(
    niterations      = 2000,
    binary_operators = ["+", "-", "*", "/"],
    unary_operators  = [],                # keep the model smooth
    elementwise_loss = "L2DistLoss()",
    alpha            = 2e-3,              # complexity penalty
    deterministic    = True,
    parallelism      = "serial",
    random_state     = 2024,
    progress         = False,
    tempdir          = tempfile.mkdtemp(),
    verbosity        = 0,
)
model.fit(t.reshape(-1, 1), base_series)

# -------------------------------------------------------------------
#  DETECTION SCORE  (bigger ⇒ more step-like)
# -------------------------------------------------------------------
def detect_score(series: np.ndarray) -> float:
    yhat   = model.predict(t.reshape(-1, 1))
    resid  = series - yhat
    sigma_r = 1.4826 * median_abs_deviation(resid, scale="normal")
    return np.max(np.abs(resid)) / sigma_r

# orientation sanity
print("\nQuick check (should be null < alt):")
print("  mean score  Δ=0.00  ", np.mean([detect_score(simulate_series(0.00)) for _ in range(40)]))
print("  mean score  Δ=0.30  ", np.mean([detect_score(simulate_series(0.30)) for _ in range(40)]))

# -------------------------------------------------------------------
#  MONTE-CARLO SCORES
# -------------------------------------------------------------------
scores_null = np.array([detect_score(simulate_series(0.0)) for _ in range(B)])
scores_alt  = {
    d: np.array([detect_score(simulate_series(d)) for _ in range(B)])
    for d in deltas
}

# -------------------------------------------------------------------
#  PLOT ROC CURVES
# -------------------------------------------------------------------
plt.figure(figsize=(7.2, 5.2))
alpha_ref = 0.05
auc_table = []

for d in deltas:                      # already ordered high→low
    vals     = scores_alt[d]
    y_scores = np.concatenate([scores_null, vals])
    y_true   = np.concatenate([np.zeros_like(scores_null), np.ones_like(vals)])

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1, drop_intermediate=False)
    roc_auc     = auc(fpr, tpr)

    # store for table
    idx         = np.searchsorted(fpr, alpha_ref, side="right") - 1
    power       = tpr[idx] if idx >= 0 else np.nan
    auc_table.append((d, roc_auc, power))

    # plot
    line, = plt.plot(fpr, tpr, lw=1.6, label=rf"$\Delta={d:.2f}$  (AUC {roc_auc:.2f})")
    plt.scatter(fpr[idx], power, s=26, color=line.get_color())   # marker @ 5 % FPR

# t-test ceiling (Δ = 0.30)
tt_scores, tt_labels = [], []
for _ in range(B):
    y0 = simulate_series(0.0)
    y1 = simulate_series(0.30)
    tt_scores.extend([-ttest_ind(y0[:20], y0[20:], equal_var=False).pvalue,
                      -ttest_ind(y1[:20], y1[20:], equal_var=False).pvalue])
    tt_labels.extend([0, 1])
fpr_t, tpr_t, _ = roc_curve(tt_labels, tt_scores, pos_label=1)
plt.plot(fpr_t, tpr_t, "k--", lw=1.4, label="t-test  Δ=0.30")

# cosmetics
plt.axvline(alpha_ref, color="grey", ls=":", lw=1.1, alpha=0.8)
plt.xlabel("False-positive rate")
plt.ylabel("True-positive rate")
plt.title("ROC: PySR residual test (solid) vs t-test ceiling (dash)")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

# save
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/roc_curve_residual_pysr.png", dpi=300)
plt.savefig("outputs/roc_curve_residual_pysr.pdf")
plt.show()

# -------------------------------------------------------------------
#  PRINT TABLE
# -------------------------------------------------------------------
print("\nAUC & power @ 5 % FPR")
print("Δ     AUC    TPR(α=0.05)")
for d, a, p in auc_table:
    print(f"{d:4.2f}  {a:5.2f}     {p:5.2f}")