# ============================================================
#  UPGRADED: House Price Prediction — Reads ANY Dataset
#  from the Internet & Performs Full Analysis
# ============================================================
#  NEW IN THIS VERSION:
#   ✅ Load any CSV dataset from a URL
#   ✅ Auto EDA (shape, stats, missing values)
#   ✅ Auto data cleaning (fills missing values)
#   ✅ Multiple features (not just area)
#   ✅ Train/Test Split (80/20)
#   ✅ Model Evaluation (R², MAE, RMSE)
#   ✅ Professional 4-Panel Dashboard Graph
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

# ============================================================
#  SECTION 1 — CONFIGURATION
#  Change the URL below to use any CSV dataset from the internet
# ============================================================

# HOW TO USE:
#   Replace this URL with any CSV dataset URL you find online.
#   Good sources: Kaggle (download → upload to GitHub → get raw URL)
#                 UCI Machine Learning Repository
#                 data.world, datasetsearch.research.google.com
#
# EXAMPLE URLS YOU CAN TRY:
#   Titanic dataset:
#     https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
#   (change TARGET_COLUMN to 'Survived' and FEATURE_COLUMNS accordingly)

DATASET_URL = "https://raw.githubusercontent.com/dhruv-bamal/House-Price-Prediction/main/data.csv"

# TARGET_COLUMN = the column you want to PREDICT (the answer/output)
# In house price datasets, this is usually 'price', 'Price', 'SalePrice', etc.
TARGET_COLUMN = "price"

# FEATURE_COLUMNS = the columns the model uses to make predictions (the inputs)
# Leave as None to auto-select ALL numeric columns except the target
FEATURE_COLUMNS = None   # e.g. ['area', 'bedrooms', 'bathrooms']


# ============================================================
#  SECTION 2 — LOAD DATASET FROM INTERNET
# ============================================================

print("=" * 60)
print("   HOUSE PRICE PREDICTION — UPGRADED VERSION")
print("=" * 60)

print(f"\n📥 Loading dataset from:\n   {DATASET_URL}\n")

try:
    # pd.read_csv() can accept a URL directly — pandas handles the download
    df = pd.read_csv(DATASET_URL)
    print(f"✅ Dataset loaded successfully!")

except Exception as e:
    print(f"\n❌ ERROR: Could not load dataset from URL.")
    print(f"   Reason: {e}")
    print("\n💡 Possible fixes:")
    print("   1. Check if the URL is a direct CSV file link (ends with .csv)")
    print("   2. Try opening the URL in your browser — it should show raw CSV text")
    print("   3. For GitHub files, use the 'Raw' button and copy that URL")
    print("   4. Check your internet connection")
    sys.exit(1)


# ============================================================
#  SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
#  EDA = Looking at your data before doing anything with it
#  Like reading the question paper before writing the exam!
# ============================================================

print("\n" + "─" * 60)
print("  STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
print("─" * 60)

# Show basic info about the dataset
print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   → {df.shape[0]} houses, {df.shape[1]} properties per house\n")

print("📋 Column Names and Data Types:")
for col in df.columns:
    missing = df[col].isnull().sum()
    missing_info = f"  ⚠ {missing} missing" if missing > 0 else ""
    print(f"   {col:<20} {str(df[col].dtype):<12}{missing_info}")

print(f"\n📈 Statistical Summary:")
print(df.describe().round(2).to_string())


# ============================================================
#  SECTION 4 — DATA CLEANING
#  Real datasets from the internet always have problems:
#   - Missing values (empty cells)
#   - Wrong data types
#  We fix these before training the model
# ============================================================

print("\n" + "─" * 60)
print("  STEP 2: DATA CLEANING")
print("─" * 60)

total_missing_before = df.isnull().sum().sum()
print(f"\n🔍 Missing values BEFORE cleaning: {total_missing_before}")

# Keep only numeric columns (ML models only understand numbers)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df = df[numeric_cols]
print(f"📌 Keeping {len(numeric_cols)} numeric columns: {numeric_cols}")

# Fill missing values with the median of each column
# Median is better than average here — it's not affected by extreme values
df.fillna(df.median(), inplace=True)

total_missing_after = df.isnull().sum().sum()
print(f"✅ Missing values AFTER cleaning:  {total_missing_after}")
print(f"   → Fixed by filling with column medians")


# ============================================================
#  SECTION 5 — PREPARE X (inputs) AND y (output)
# ============================================================

print("\n" + "─" * 60)
print("  STEP 3: PREPARE DATA FOR MODEL")
print("─" * 60)

# Validate target column exists
if TARGET_COLUMN not in df.columns:
    print(f"\n❌ ERROR: Target column '{TARGET_COLUMN}' not found!")
    print(f"   Available columns: {list(df.columns)}")
    print(f"   → Set TARGET_COLUMN at the top of this file to the correct name.")
    sys.exit(1)

# Auto-select feature columns if not specified
if FEATURE_COLUMNS is None:
    # Use all numeric columns except the target
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
else:
    feature_cols = FEATURE_COLUMNS

print(f"\n🎯 Target (what we predict): '{TARGET_COLUMN}'")
print(f"📥 Features (what we use):   {feature_cols}")

# X = the input table (multiple columns, like area, bedrooms, etc.)
X = df[feature_cols]

# y = the output column (the price we want to predict)
y = df[TARGET_COLUMN]

print(f"\n   X shape: {X.shape}  (rows × feature columns)")
print(f"   y shape: {y.shape}  (one price per row)")


# ============================================================
#  SECTION 6 — TRAIN/TEST SPLIT
#  NEW: We now split data into Training set and Testing set
#
#  Why? Because we can't test a student's knowledge on the same
#  questions they already studied! We need NEW questions.
#
#  80% of data → used to TRAIN the model
#  20% of data → used to TEST the model (hidden during training)
# ============================================================

print("\n" + "─" * 60)
print("  STEP 4: TRAIN/TEST SPLIT (80% train, 20% test)")
print("─" * 60)

# random_state=42 means the split is always the same — reproducible results
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n   Training set:  {X_train.shape[0]} houses  → model learns from these")
print(f"   Testing set:   {X_test.shape[0]} houses  → model is tested on these")


# ============================================================
#  SECTION 7 — TRAIN THE MODEL
# ============================================================

print("\n" + "─" * 60)
print("  STEP 5: TRAINING THE MODEL")
print("─" * 60)

model = LinearRegression()
model.fit(X_train, y_train)    # Only trained on training data!
print("\n✅ Model trained successfully!")

# Show what the model learned — the coefficient of each feature
print("\n📊 What the model learned (Feature Coefficients):")
print("   (Positive = increases price | Negative = decreases price)")
coef_df = pd.Series(model.coef_, index=feature_cols).sort_values(ascending=False)
for feat, coef in coef_df.items():
    direction = "↑" if coef > 0 else "↓"
    print(f"   {direction} {feat:<20} {coef:+.4f}")
print(f"\n   Baseline (intercept): {model.intercept_:.2f}")


# ============================================================
#  SECTION 8 — EVALUATE THE MODEL
#  NEW: We now measure HOW GOOD the model is, using 3 metrics:
#
#   R² Score → How much of the price variation the model explains
#              1.0 = perfect, 0.0 = useless, negative = worse than useless
#              Good R² for house prices: > 0.75
#
#   MAE (Mean Absolute Error) → On average, how far off is the prediction?
#              e.g. MAE = 5 means predictions are off by ₹5 Lakhs on average
#
#   RMSE (Root Mean Square Error) → Like MAE but punishes big errors more
#              Lower is better
# ============================================================

print("\n" + "─" * 60)
print("  STEP 6: MODEL EVALUATION")
print("─" * 60)

y_pred = model.predict(X_test)   # Predict prices for the 20% test set

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n   R² Score:  {r2:.4f}  {'✅ Excellent' if r2 > 0.85 else '⚠ Needs improvement' if r2 > 0.6 else '❌ Poor fit'}")
print(f"   MAE:       {mae:.2f}   → avg prediction error")
print(f"   RMSE:      {rmse:.2f}   → penalized prediction error")


# ============================================================
#  SECTION 9 — MAKE A NEW PREDICTION
# ============================================================

print("\n" + "─" * 60)
print("  STEP 7: MAKE A PREDICTION ON A NEW HOUSE")
print("─" * 60)

# Create a new house using the median values of each feature
# (This gives a "typical" house from the dataset)
new_house_values = X_train.median().values.reshape(1, -1)
new_house = pd.DataFrame(new_house_values, columns=feature_cols)

predicted = model.predict(new_house)[0]

print(f"\n🏠 Predicting price for a house with these features:")
for feat in feature_cols:
    print(f"   {feat}: {new_house[feat].values[0]:.1f}")
print(f"\n💰 Predicted Price: {predicted:.2f} (in dataset units)")


# ============================================================
#  SECTION 10 — VISUALIZE: 4-PANEL DASHBOARD
# ============================================================

print("\n" + "─" * 60)
print("  STEP 8: GENERATING ANALYSIS DASHBOARD")
print("─" * 60)

# Color palette for dark dashboard
ACCENT = '#00d4ff'    # Blue — scatter points
GREEN  = '#00ff99'    # Green — positive impact features
ORANGE = '#ff6b35'    # Orange — negative impact features / reference lines
BG     = '#1a1d2e'    # Dark panel background
TEXT   = '#e0e0e0'    # Light text

fig = plt.figure(figsize=(14, 10), facecolor='#0f1117')
fig.suptitle(
    f'House Price Analysis Dashboard  |  Dataset: {len(df)} records  |  Features: {len(feature_cols)}',
    fontsize=16, color='white', fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

def style_ax(ax, title):
    """Apply consistent dark styling to each subplot"""
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2d3e')
    ax.set_title(title, color=TEXT, fontsize=11, pad=10, fontweight='bold')
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

# ─── CHART 1: Actual vs Predicted (how accurate is the model?) ─────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_pred, alpha=0.4, s=20, color=ACCENT, edgecolors='none')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax1.plot(lims, lims, color=ORANGE, lw=2, linestyle='--', label='Perfect Line')
ax1.legend(fontsize=8, labelcolor=TEXT, facecolor=BG, edgecolor='#2a2d3e')
ax1.set_xlabel(f'Actual {TARGET_COLUMN}')
ax1.set_ylabel(f'Predicted {TARGET_COLUMN}')
ax1.text(0.05, 0.92, f'R² = {r2:.3f}', transform=ax1.transAxes,
         color=GREEN, fontsize=10, fontweight='bold')
style_ax(ax1, '① Actual vs Predicted')

# ─── CHART 2: Feature Importance (which features matter most?) ─────────────
ax2 = fig.add_subplot(gs[0, 1])
coefs_sorted = coef_df.sort_values()
bar_colors = [GREEN if v > 0 else ORANGE for v in coefs_sorted.values]
bars = ax2.barh(coefs_sorted.index, coefs_sorted.values,
                color=bar_colors, edgecolor='none', height=0.6)
ax2.axvline(0, color='white', lw=0.8, alpha=0.4)
for bar, val in zip(bars, coefs_sorted.values):
    offset = abs(coefs_sorted.values).max() * 0.03
    ha = 'left' if val >= 0 else 'right'
    ax2.text(val + (offset if val >= 0 else -offset),
             bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}', va='center', ha=ha, color=TEXT, fontsize=8)
ax2.set_xlabel('Coefficient (Impact on Price)')
style_ax(ax2, '② Feature Impact (Coefficients)')

# ─── CHART 3: Price Distribution ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(df[TARGET_COLUMN], bins=35, color=ACCENT, alpha=0.75, edgecolor='none')
ax3.axvline(df[TARGET_COLUMN].mean(), color=ORANGE, lw=2, linestyle='--',
            label=f'Mean: {df[TARGET_COLUMN].mean():.1f}')
ax3.axvline(df[TARGET_COLUMN].median(), color=GREEN, lw=2, linestyle=':',
            label=f'Median: {df[TARGET_COLUMN].median():.1f}')
ax3.legend(fontsize=8, labelcolor=TEXT, facecolor=BG, edgecolor='#2a2d3e')
ax3.set_xlabel(TARGET_COLUMN)
ax3.set_ylabel('Count')
style_ax(ax3, '③ Price Distribution')

# ─── CHART 4: Residuals (prediction error distribution) ────────────────────
ax4 = fig.add_subplot(gs[1, 1])
residuals = y_test.values - y_pred
ax4.hist(residuals, bins=35, color=ORANGE, alpha=0.75, edgecolor='none')
ax4.axvline(0, color='white', lw=1.5, linestyle='--', alpha=0.8)
ax4.set_xlabel('Prediction Error (Actual − Predicted)')
ax4.set_ylabel('Count')
ax4.text(0.05, 0.90, 'Centered at 0 = Good!',
         transform=ax4.transAxes, color=GREEN, fontsize=9)
style_ax(ax4, '④ Residuals (Error Distribution)')

# ─── METRICS FOOTER BANNER ─────────────────────────────────────────────────
fig.text(
    0.5, 0.005,
    f'  R²: {r2:.3f}   |   MAE: {mae:.2f}   |   RMSE: {rmse:.2f}'
    f'   |   Train: {len(X_train)}   |   Test: {len(X_test)}  ',
    ha='center', va='bottom', fontsize=10, color='#0f1117',
    bbox=dict(boxstyle='round,pad=0.4', facecolor=ACCENT, alpha=0.9)
)

plt.savefig('dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("\n✅ Dashboard saved as 'dashboard.png'")
plt.show()

print("\n" + "=" * 60)
print("  ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\n  Dataset:    {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Features:   {feature_cols}")
print(f"  Target:     {TARGET_COLUMN}")
print(f"  R² Score:   {r2:.4f}")
print(f"  MAE:        {mae:.2f}")
print(f"  RMSE:       {rmse:.2f}")
print(f"  Prediction: {predicted:.2f}")
print("=" * 60)