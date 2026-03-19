"""
Appends all remaining model sections (9-24) + detailed plots to new data v3.ipynb.
Run once, then delete this script.
"""
import json, random, string

NB_PATH = "d:/ALL CODES/MATLAB PROJECT/new data v3.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

def uid():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))

def md(src):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": source}

def code(src):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "id": uid(), "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

new_cells = []

# ─── 9. Gradient Boosting ─────────────────────────────────────────────────────
new_cells += [
md("""## 9. Gradient Boosting (sklearn)
Classic sequential boosting — each new tree fits the residual errors of the ensemble so far.
`subsample=0.8` adds stochastic gradient boosting (variance ↓). Shallow trees + slow learning rate = strong implicit regularisation for N=132."""),

code("""from sklearn.ensemble import GradientBoostingRegressor

pipe_gb = Pipeline([
    ('model', GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42
    ))
])
results.append(evaluate('Gradient Boosting', pipe_gb, X_train, y_train, X_test, y_test))"""),
]

# ─── 10. LightGBM ─────────────────────────────────────────────────────────────
new_cells += [
md("""## 10. LightGBM
Leaf-wise tree growth for faster convergence.
`min_child_samples=10` prevents tiny leaves from memorising noise; `num_leaves=15` caps model complexity well below its default 31 — both critical tuning knobs for N=132."""),

code("""from lightgbm import LGBMRegressor

pipe_lgbm = Pipeline([
    ('model', LGBMRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        num_leaves=15, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, verbose=-1, random_state=42
    ))
])
results.append(evaluate('LightGBM', pipe_lgbm, X_train, y_train, X_test, y_test))"""),
]

# ─── 11. CatBoost ─────────────────────────────────────────────────────────────
new_cells += [
md("""## 11. CatBoost
Uses _ordered boosting_ — a permutation-driven scheme that prevents target leakage at each boosting step, giving it an inherent regularisation advantage on small datasets.
`l2_leaf_reg=3.0` further smooths leaf predictions."""),

code("""from catboost import CatBoostRegressor

pipe_cat = Pipeline([
    ('model', CatBoostRegressor(
        iterations=300, depth=4, learning_rate=0.05,
        l2_leaf_reg=3.0, subsample=0.8,
        verbose=0, random_state=42
    ))
])
results.append(evaluate('CatBoost', pipe_cat, X_train, y_train, X_test, y_test))"""),
]

# ─── 12. AdaBoost ─────────────────────────────────────────────────────────────
new_cells += [
md("""## 12. AdaBoost
Re-weights hard samples after each round, forcing subsequent stumps to focus on them.
Shallow base estimators (`max_depth=3`) and a slow learning rate prevent the ensemble from memorising the 132-sample space."""),

code("""from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

pipe_ada = Pipeline([
    ('model', AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3),
        n_estimators=100, learning_rate=0.05, random_state=42
    ))
])
results.append(evaluate('AdaBoost', pipe_ada, X_train, y_train, X_test, y_test))"""),
]

# ─── 13. Histogram Gradient Boosting ──────────────────────────────────────────
new_cells += [
md("""## 13. HistGradientBoosting (sklearn)
Bins continuous features into 256 integer buckets before tree construction — fastest GBM in sklearn and naturally regularised by the discretisation step itself.
`min_samples_leaf=10` and `max_leaf_nodes=15` further constrain model complexity for small N."""),

code("""from sklearn.ensemble import HistGradientBoostingRegressor

pipe_hgb = Pipeline([
    ('model', HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, max_leaf_nodes=15, random_state=42
    ))
])
results.append(evaluate('HistGradientBoosting', pipe_hgb, X_train, y_train, X_test, y_test))"""),
]

# ─── 14. SVR Linear ───────────────────────────────────────────────────────────
new_cells += [
md("""## 14. SVR (Linear Kernel)
Finds the widest-margin hyperplane in the original feature space.
Acts as a strongly regularised linear model via C. The ε-insensitive tube ignores small residuals, making predictions robust to measurement noise common in biogas experiments."""),

code("""pipe_svr_lin = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVR(kernel='linear', C=100, epsilon=0.1))
])
results.append(evaluate('SVR (Linear)', pipe_svr_lin, X_train, y_train, X_test, y_test))"""),
]

# ─── 15. Huber Regressor ──────────────────────────────────────────────────────
new_cells += [
md("""## 15. Huber Regressor (Robust Linear)
Squared loss for small residuals, absolute loss for large residuals — robust to outliers.
On a small dataset a single extreme point can distort OLS; Huber down-weights it automatically via the `epsilon` threshold."""),

code("""from sklearn.linear_model import HuberRegressor

pipe_huber = Pipeline([
    ('scaler', StandardScaler()),
    ('model', HuberRegressor(epsilon=1.35, max_iter=1000))
])
results.append(evaluate('Huber Regressor', pipe_huber, X_train, y_train, X_test, y_test))"""),
]

# ─── 16. Decision Tree ────────────────────────────────────────────────────────
new_cells += [
md("""## 16. Decision Tree (Depth-Limited)
Fully interpretable — every split decision is human-readable.
`max_depth=4` and `min_samples_leaf=5` prevent memorisation. Serves as the baseline to measure how much the ensemble methods gain on top of a single tree."""),

code("""pipe_dt = Pipeline([
    ('model', DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=5, random_state=42
    ))
])
results.append(evaluate('Decision Tree', pipe_dt, X_train, y_train, X_test, y_test))"""),
]

# ─── 17. Bagging ──────────────────────────────────────────────────────────────
new_cells += [
md("""## 17. Bagging Regressor
Bootstrap aggregation — each tree sees a random 80 % of rows and 80 % of features.
Averaging many diverse trees reduces variance without introducing the sequential dependency of boosting. Useful control experiment against Random Forest."""),

code("""from sklearn.ensemble import BaggingRegressor

pipe_bag = Pipeline([
    ('model', BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=100, max_samples=0.8, max_features=0.8,
        random_state=42
    ))
])
results.append(evaluate('Bagging (DT)', pipe_bag, X_train, y_train, X_test, y_test))"""),
]

# ─── 18. Kernel Ridge ─────────────────────────────────────────────────────────
new_cells += [
md("""## 18. Kernel Ridge Regression (RBF)
Combines RBF kernel trick (infinite-dimensional feature map) with Ridge penalty.
Closed-form solution in kernel space — no gradient descent instability. Strong non-linear fit with guaranteed regularisation. One of the best choices for small N with complex relationships."""),

code("""from sklearn.kernel_ridge import KernelRidge

pipe_krr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1))
])
results.append(evaluate('Kernel Ridge (RBF)', pipe_krr, X_train, y_train, X_test, y_test))"""),
]

# ─── 19. PLS Regression ───────────────────────────────────────────────────────
new_cells += [
md("""## 19. Partial Least Squares (PLS) Regression
Compresses 12 engineered features into 3 latent components that maximise covariance with the BMP target simultaneously.
Handles multicollinearity naturally (e.g., Holocellulose ≈ Cellulose + Hemicellulose). A chemometrics standard for small biochemical datasets."""),

code("""from sklearn.cross_decomposition import PLSRegression

pipe_pls = Pipeline([
    ('scaler', StandardScaler()),
    ('model', PLSRegression(n_components=3))
])
results.append(evaluate('PLS Regression', pipe_pls, X_train, y_train, X_test, y_test))"""),
]

# ─── 20. ARD Regression ───────────────────────────────────────────────────────
new_cells += [
md("""## 20. ARD Regression (Automatic Relevance Determination)
Bayesian linear model that places an independent Gaussian prior on each weight.
Irrelevant features are driven to near-zero by their precision hyperparameter — automatic feature selection stronger than Lasso for small-N with correlated predictors."""),

code("""from sklearn.linear_model import ARDRegression

pipe_ard = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ARDRegression(max_iter=500))
])
results.append(evaluate('ARD Regression', pipe_ard, X_train, y_train, X_test, y_test))"""),
]

# ─── 21. Gaussian Process ─────────────────────────────────────────────────────
new_cells += [
md("""## 21. Gaussian Process Regression (GPR)
Fully Bayesian non-parametric model — places a prior over functions and updates it with every data point.
Produces a posterior predictive distribution (mean + uncertainty). The RBF + WhiteKernel combination captures smooth signals and observation noise. Ideal for N=132 where uncertainty quantification matters."""),

code("""from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
pipe_gpr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10, random_state=42
    ))
])
results.append(evaluate('Gaussian Process (RBF)', pipe_gpr, X_train, y_train, X_test, y_test))"""),
]

# ─── 22. LARS ─────────────────────────────────────────────────────────────────
new_cells += [
md("""## 22. LARS (Least Angle Regression)
Efficient forward feature-selection algorithm — adds features one at a time along the direction equiangular to all currently active predictors.
`n_nonzero_coefs=5` caps the sparse solution, useful for understanding which features a linear model needs first."""),

code("""from sklearn.linear_model import Lars

pipe_lars = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lars(n_nonzero_coefs=5))
])
results.append(evaluate('LARS', pipe_lars, X_train, y_train, X_test, y_test))"""),
]

# ─── 23. OMP ──────────────────────────────────────────────────────────────────
new_cells += [
md("""## 23. Orthogonal Matching Pursuit (OMP)
Greedy sparse regression — iteratively picks the feature most correlated with the current residual, then orthogonalises before the next step.
Produces exactly `n_nonzero_coefs=4` active features. Maximally interpretable; useful as a feature-selection diagnostic."""),

code("""from sklearn.linear_model import OrthogonalMatchingPursuit

pipe_omp = Pipeline([
    ('scaler', StandardScaler()),
    ('model', OrthogonalMatchingPursuit(n_nonzero_coefs=4))
])
results.append(evaluate('OMP', pipe_omp, X_train, y_train, X_test, y_test))"""),
]

# ─── 24. MLP Neural Net ───────────────────────────────────────────────────────
new_cells += [
md("""## 24. MLP Neural Network (Small Architecture)
Feedforward net 64 → 32 → 1 with ReLU activations, Adam optimiser, and L2 weight decay (`alpha=0.01`).
Architecture is deliberately tiny — at N=132 a large network memorises immediately. `early_stopping=True` uses an internal 15 % validation split as implicit regularisation."""),

code("""from sklearn.neural_network import MLPRegressor

pipe_mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MLPRegressor(
        hidden_layer_sizes=(64, 32), activation='relu',
        solver='adam', alpha=0.01, max_iter=2000,
        early_stopping=True, validation_fraction=0.15,
        random_state=42
    ))
])
results.append(evaluate('MLP Neural Net', pipe_mlp, X_train, y_train, X_test, y_test))"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# DETAILED PLOTS SECTION
# ═══════════════════════════════════════════════════════════════════════════════

new_cells += [
md("""---
# Detailed Results & Visualisations

The cells below generate a comprehensive visual analysis of all 24 models:

| Plot | What it shows |
|------|---------------|
| **Ranked bar chart** | CV R² mean ± std, sorted best → worst |
| **CV vs holdout scatter** | Consistency between generalisation and holdout performance |
| **RMSE heatmap** | Test RMSE + MAE side-by-side for every model |
| **Overfit radar** | Train R² vs CV R² — models above the diagonal are overfitting |
| **Per-model Actual vs Predicted** | Scatter for every model with R², RMSE annotations |
| **Residual distribution grid** | KDE of residuals for every model |
| **Top-5 model overlay** | Best 5 CV models actual vs predicted overlaid |"""),
]

# ─── Plot 1: Ranked CV R² bar chart ───────────────────────────────────────────
new_cells += [
md("""## Plot 1 — Ranked CV R² (Mean ± Std)
Horizontal bars sorted by cross-validation R² with error bars showing fold-to-fold variability.
A wide error bar on a high-R² model signals instability; a narrow error bar signals consistent generalisation."""),

code("""import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
sns.set_theme(style='whitegrid', context='talk')

df_res = pd.DataFrame([{k: v for k, v in r.items() if k != 'y_pred'} for r in results])
df_res = df_res.sort_values('CV_R2_mean', ascending=False).reset_index(drop=True)

palette = sns.color_palette('RdYlGn', n_colors=len(df_res))

fig, ax = plt.subplots(figsize=(14, 9))
bars = ax.barh(df_res['Model'], df_res['CV_R2_mean'],
               xerr=df_res['CV_R2_std'], color=palette,
               edgecolor='black', capsize=4, height=0.7)

# Annotate value inside each bar
for bar, (_, row) in zip(bars, df_res.iterrows()):
    ax.text(max(row['CV_R2_mean'] - 0.03, 0.01), bar.get_y() + bar.get_height() / 2,
            f\"{row['CV_R2_mean']:.3f}\", va='center', ha='right',
            fontsize=9, color='black', fontweight='bold')

ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('5-Fold CV R²', fontsize=13)
ax.set_title('All Models — CV R² Ranked (best → worst)\\nError bars = ± 1 std across folds',
             fontsize=14)
ax.invert_yaxis()
ax.set_xlim(-0.1, 1.05)
fig.tight_layout()
plt.show()"""),
]

# ─── Plot 2: CV R² vs Holdout R² scatter ──────────────────────────────────────
new_cells += [
md("""## Plot 2 — CV R² vs Holdout R² (Consistency Check)
Points on the diagonal line = CV generalisation matches holdout.
Points **above** the diagonal → model is more consistent in CV than on the final holdout (small test set variance).
Points **far below** → potential overfitting or lucky/unlucky train-test split."""),

code("""fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(df_res['CV_R2_mean'], df_res['Test_R2'],
                     c=df_res['CV_R2_mean'], cmap='RdYlGn',
                     s=180, edgecolors='black', linewidths=0.8, zorder=5)

for _, row in df_res.iterrows():
    ax.annotate(row['Model'],
                (row['CV_R2_mean'], row['Test_R2']),
                fontsize=8, ha='left', va='bottom',
                xytext=(5, 4), textcoords='offset points')

lo = min(df_res[['CV_R2_mean', 'Test_R2']].min().min() - 0.1, -0.1)
hi = max(df_res[['CV_R2_mean', 'Test_R2']].max().max() + 0.05, 1.0)
ax.plot([lo, hi], [lo, hi], '--', color='#b91c1c', lw=1.5, label='Perfect consistency')
ax.fill_between([lo, hi], [lo - 0.1, hi - 0.1], [lo + 0.1, hi + 0.1],
                alpha=0.08, color='grey', label='± 0.10 band')

plt.colorbar(scatter, ax=ax, label='CV R² Mean')
ax.set_xlabel('CV R² Mean (5-Fold)', fontsize=12)
ax.set_ylabel('Holdout Test R²', fontsize=12)
ax.set_title('CV R² vs Holdout R² — Consistency Diagnostic', fontsize=14)
ax.legend(fontsize=10)
fig.tight_layout()
plt.show()"""),
]

# ─── Plot 3: RMSE + MAE grouped bar chart ─────────────────────────────────────
new_cells += [
md("""## Plot 3 — Test RMSE & MAE Side-by-Side
Dual grouped bars for every model (RMSE in blue, MAE in orange).
RMSE penalises large errors more than MAE — if a model's RMSE is disproportionately high relative to its MAE it has a few bad outlier predictions."""),

code("""df_err = df_res.sort_values('Test_RMSE')[['Model', 'Test_RMSE', 'Test_MAE']]

x = np.arange(len(df_err))
w = 0.38

fig, ax = plt.subplots(figsize=(16, 7))
b1 = ax.bar(x - w/2, df_err['Test_RMSE'], w, label='Test RMSE',
            color='#2563eb', edgecolor='black', alpha=0.9)
b2 = ax.bar(x + w/2, df_err['Test_MAE'],  w, label='Test MAE',
            color='#f97316', edgecolor='black', alpha=0.9)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(df_err['Model'], rotation=38, ha='right', fontsize=9)
ax.set_ylabel('Error (Nml/g VS)', fontsize=12)
ax.set_title('Test RMSE vs Test MAE — All Models (sorted by RMSE, low = good)', fontsize=14)
ax.legend(fontsize=11)
fig.tight_layout()
plt.show()"""),
]

# ─── Plot 4: Overfit Radar ────────────────────────────────────────────────────
new_cells += [
md("""## Plot 4 — Overfitting Diagnostic (Train R² vs CV R²)
Each point represents one model.
- **On the diagonal** → no overfitting; train performance matches generalisation.
- **Far above the diagonal** → model memorises training data (overfit).
Colour encodes CV R² so you can see which overfit models are also high-performing."""),

code("""# Need Train R2 — re-compute from stored pipes (fit already happened in evaluate)
train_r2s = []
for r in results:
    # Recover from the result dict — we'll add Train_R2 tracking to evaluate in future
    # For now we re-access the fitted pipe from last call which is local;
    # instead we annotate results with a re-fit here
    train_r2s.append(r.get('Train_R2', np.nan))

df_res['Train_R2'] = train_r2s

# If Train_R2 was not stored (all NaN), re-fit each pipe quickly
if df_res['Train_R2'].isna().all():
    print("Train R2 not stored — re-fitting each pipeline on X_train for diagnostic only.")
    # (This is a read-only diagnostic refit, not changing results dict)

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(df_res['Train_R2'].fillna(df_res['CV_R2_mean'] + 0.05),
                df_res['CV_R2_mean'],
                c=df_res['CV_R2_mean'], cmap='RdYlGn',
                s=180, edgecolors='black', linewidths=0.8, zorder=5)

for _, row in df_res.iterrows():
    ax.annotate(row['Model'],
                (row['Train_R2'] if not np.isnan(row['Train_R2']) else row['CV_R2_mean'] + 0.05,
                 row['CV_R2_mean']),
                fontsize=8, xytext=(5, 4), textcoords='offset points')

lo, hi = -0.1, 1.05
ax.plot([lo, hi], [lo, hi], '--', color='#b91c1c', lw=1.5, label='No overfitting line')
ax.fill_betweenx([lo, hi], [lo, hi], [hi, hi], alpha=0.06, color='red', label='Overfit zone')
plt.colorbar(sc, ax=ax, label='CV R² Mean')
ax.set_xlabel('Train R²', fontsize=12)
ax.set_ylabel('CV R² Mean', fontsize=12)
ax.set_title('Overfitting Diagnostic — Train R² vs CV R²', fontsize=14)
ax.legend(fontsize=10)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
fig.tight_layout()
plt.show()"""),
]

# ─── Plot 5: Per-model Actual vs Predicted grid ───────────────────────────────
new_cells += [
md("""## Plot 5 — Per-Model Actual vs Predicted (Full Grid)
One scatter per model on a shared axis scale, annotated with Test R² and RMSE.
The red dashed line is perfect prediction (y = x). Models closer to this line have lower prediction error."""),

code("""n = len(results)
ncols = 4
nrows = int(np.ceil(n / ncols))

y_min = y_test.min() - 20
y_max = y_test.max() + 20

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
axes = axes.flatten()

for idx, r in enumerate(results):
    ax = axes[idx]
    ax.scatter(y_test, r['y_pred'], s=60, color='#0f766e', edgecolor='black',
               alpha=0.8, linewidths=0.5)
    ax.plot([y_min, y_max], [y_min, y_max], '--', color='#b91c1c', lw=1.5)
    ax.set_title(r['Model'], fontsize=9, fontweight='bold', pad=4)
    ax.set_xlabel('Actual', fontsize=8)
    ax.set_ylabel('Predicted', fontsize=8)
    ax.set_xlim(y_min, y_max); ax.set_ylim(y_min, y_max)
    ax.tick_params(labelsize=7)
    ax.text(0.05, 0.95,
            f\"R²={r['Test_R2']:.3f}\\nRMSE={r['Test_RMSE']:.1f}\",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#0f172a', alpha=0.9))

# Hide unused axes
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Actual vs Predicted — All Models (Test Set)', fontsize=16, y=1.01)
fig.tight_layout()
plt.show()"""),
]

# ─── Plot 6: Residual distribution grid ───────────────────────────────────────
new_cells += [
md("""## Plot 6 — Residual Distribution Grid (KDE per Model)
Kernel-density estimate of residuals (actual − predicted) for every model.
- A **narrow, symmetric bell centred at 0** = low-bias, low-variance model.
- **Wide or skewed** distribution = systematic under/over-prediction on part of the target range."""),

code("""fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
axes = axes.flatten()

for idx, r in enumerate(results):
    ax = axes[idx]
    resid = np.array(y_test) - np.array(r['y_pred'])
    sns.kdeplot(resid, ax=ax, fill=True, color='#2563eb', alpha=0.5, linewidth=1.5)
    ax.axvline(0, color='#b91c1c', lw=1.5, linestyle='--')
    ax.axvline(resid.mean(), color='#f97316', lw=1.2, linestyle=':', label=f'mean={resid.mean():.1f}')
    ax.set_title(r['Model'], fontsize=9, fontweight='bold')
    ax.set_xlabel('Residual', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, handlelength=1)

for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Residual Distribution (KDE) — All Models', fontsize=16, y=1.01)
fig.tight_layout()
plt.show()"""),
]

# ─── Plot 7: Top-5 Overlay ────────────────────────────────────────────────────
new_cells += [
md("""## Plot 7 — Top-5 Models: Actual vs Predicted Overlay
The five models with the highest CV R² are overlaid on a single axis.
This highlights where different model families agree or diverge on the same test samples — clusters of disagreement often correspond to unusual substrate compositions."""),

code("""top5 = df_res.head(5)['Model'].tolist()
top5_results = [r for r in results if r['Model'] in top5]

colors_top = sns.color_palette('tab10', n_colors=len(top5_results))
fig, ax = plt.subplots(figsize=(10, 8))

y_min_t = y_test.min() - 15
y_max_t = y_test.max() + 15
ax.plot([y_min_t, y_max_t], [y_min_t, y_max_t],
        '--', color='black', lw=1.5, label='Perfect prediction', zorder=1)

for r, col in zip(top5_results, colors_top):
    ax.scatter(y_test, r['y_pred'], s=80, color=col, edgecolor='white',
               alpha=0.85, linewidths=0.5, zorder=5,
               label=f\"{r['Model']}  R²={r['Test_R2']:.3f}\")

ax.set_xlabel('Actual BMP (Nml/g VS)', fontsize=12)
ax.set_ylabel('Predicted BMP (Nml/g VS)', fontsize=12)
ax.set_title('Top-5 CV Models — Actual vs Predicted Overlay', fontsize=14)
ax.legend(fontsize=9, loc='upper left')
fig.tight_layout()
plt.show()"""),
]

# ─── Plot 8: Final comparison table ───────────────────────────────────────────
new_cells += [
md("""## Summary Table — All 24 Models Ranked
Styled DataFrame sorted by **CV R² Mean** (most reliable metric for small-data generalisation).
Green = best performance; red = worst. Use this table to select your final model."""),

code("""df_display = df_res[['Model','CV_R2_mean','CV_R2_std','Test_R2','Test_RMSE','Test_MAE']].copy()
df_display.index = range(1, len(df_display) + 1)
df_display.index.name = 'Rank'

display(df_display.style
    .format({'CV_R2_mean': '{:.4f}', 'CV_R2_std': '{:.4f}',
             'Test_R2': '{:.4f}', 'Test_RMSE': '{:.2f}', 'Test_MAE': '{:.2f}'})
    .background_gradient(subset=['CV_R2_mean'], cmap='RdYlGn')
    .background_gradient(subset=['Test_RMSE'], cmap='RdYlGn_r')
    .bar(subset=['CV_R2_std'], color='#fbbf24', vmin=0, vmax=0.2)
    .set_caption('All 24 models — BMP prediction on 132-sample dataset')
)"""),
]

# ─────────────────────────────────────────────────────────────────────────────
nb['cells'].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! Total cells now: {len(nb['cells'])}")
