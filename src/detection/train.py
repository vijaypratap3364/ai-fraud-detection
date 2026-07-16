"""
Detection Model Training - XGBoost with SMOTE

This script trains the fraud detection model (Phase 2: Detection Layer).
It handles the severe class imbalance using SMOTE and evaluates with proper metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'creditcard.csv'
MODEL_SAVE_PATH = PROJECT_ROOT / 'models' / 'detection'
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FRAUD DETECTION MODEL TRAINING - PHASE 2")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} transactions")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"\nFeatures: {X.shape[1]} columns")
print(f"Target distribution:")
print(f"  Non-Fraud (0): {(y == 0).sum():,} ({(y == 0).sum() / len(y) * 100:.3f}%)")
print(f"  Fraud (1):     {(y == 1).sum():,} ({(y == 1).sum() / len(y) * 100:.3f}%)")
print(f"  Imbalance ratio: 1:{int((y == 0).sum() / (y == 1).sum())}")

# ============================================================================
# 2. TRAIN/TEST SPLIT (before SMOTE!)
# ============================================================================
print("\n2. Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {len(X_train):,} samples")
print(f"Test set:  {len(X_test):,} samples")
print(f"\nTrain fraud rate: {(y_train == 1).sum() / len(y_train) * 100:.3f}%")
print(f"Test fraud rate:  {(y_test == 1).sum() / len(y_test) * 100:.3f}%")

# ============================================================================
# 3. FEATURE SCALING
# ============================================================================
print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled (StandardScaler fitted on train set only)")

# ============================================================================
# 4. APPLY SMOTE (only on training set!)
# ============================================================================
print("\n4. Applying SMOTE to balance training set...")
print("IMPORTANT: SMOTE is applied ONLY to training data, never to test data!")

smote = SMOTE(sampling_strategy=0.1, random_state=42)  # Minority class = 10% of majority
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"\nBefore SMOTE:")
print(f"  Non-Fraud: {(y_train == 0).sum():,}")
print(f"  Fraud:     {(y_train == 1).sum():,}")
print(f"\nAfter SMOTE:")
print(f"  Non-Fraud: {(y_train_resampled == 0).sum():,}")
print(f"  Fraud:     {(y_train_resampled == 1).sum():,}")
print(f"  New ratio: 1:{int((y_train_resampled == 0).sum() / (y_train_resampled == 1).sum())}")

# ============================================================================
# 5. TRAIN XGBOOST MODEL
# ============================================================================
print("\n5. Training XGBoost model...")

# XGBoost hyperparameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',  # PR-AUC, not ROC-AUC (better for imbalanced data)
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 1
}

model = xgb.XGBClassifier(**params)
model.fit(
    X_train_resampled,
    y_train_resampled,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

print("Model training complete!")

# ============================================================================
# 6. PREDICTIONS
# ============================================================================
print("\n6. Making predictions on test set...")

# Predict probabilities
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Predict classes with default threshold (0.5)
y_pred = model.predict(X_test_scaled)

# We'll also calculate optimal threshold based on PR curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

# Find threshold that maximizes F1 score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"\nOptimal threshold (max F1): {optimal_threshold:.4f}")

# Predict with optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# ============================================================================
# 7. EVALUATION (proper metrics for imbalanced data)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL EVALUATION - WHY THESE METRICS MATTER")
print("=" * 80)

print("\n--- WRONG METRIC: Accuracy ---")
accuracy = (y_pred == y_test).sum() / len(y_test)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("^ THIS IS MEANINGLESS! A 'always predict non-fraud' model gets 99.83% accuracy.")
print("  Accuracy is misleading for imbalanced datasets.")

print("\n--- RIGHT METRICS: Precision, Recall, F1, PR-AUC ---")
print("\nWith default threshold (0.5):")
print(f"  Precision: {precision_score(y_test, y_pred):.4f} (of flagged transactions, how many are actually fraud)")
print(f"  Recall:    {recall_score(y_test, y_pred):.4f} (of all fraud cases, how many did we catch)")
print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f} (harmonic mean of precision and recall)")

print(f"\nWith optimal threshold ({optimal_threshold:.4f}):")
print(f"  Precision: {precision_score(y_test, y_pred_optimal):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_optimal):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_optimal):.4f}")

print(f"\nPR-AUC (Precision-Recall Area Under Curve): {pr_auc:.4f}")
print("^ This is the MOST IMPORTANT metric for imbalanced classification!")
print("  Range: 0 to 1 (higher is better)")
print(f"  Random baseline for this dataset: {(y_test == 1).sum() / len(y_test):.4f}")

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC: {roc_auc:.4f} (less informative for imbalanced data)")

# Confusion Matrix
print("\n--- Confusion Matrix (with optimal threshold) ---")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()
print(f"True Negatives (correctly predicted non-fraud):  {tn:,}")
print(f"False Positives (non-fraud flagged as fraud):    {fp:,}")
print(f"False Negatives (fraud missed):                  {fn:,}")
print(f"True Positives (correctly detected fraud):       {tp:,}")

print(f"\nFalse Positive Rate: {fp / (fp + tn):.4f} ({fp / (fp + tn) * 100:.2f}%)")
print(f"False Negative Rate: {fn / (fn + tp):.4f} ({fn / (fn + tp) * 100:.2f}%)")

# ============================================================================
# 8. SAVE MODEL AND ARTIFACTS
# ============================================================================
print("\n8. Saving model and artifacts...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save trained model
model_path = MODEL_SAVE_PATH / f'xgboost_model_{timestamp}.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved: {model_path}")

# Save scaler (needed for inference)
scaler_path = MODEL_SAVE_PATH / f'scaler_{timestamp}.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved: {scaler_path}")

# Save evaluation metrics
metrics = {
    'timestamp': timestamp,
    'pr_auc': float(pr_auc),
    'roc_auc': float(roc_auc),
    'optimal_threshold': float(optimal_threshold),
    'precision_optimal': float(precision_score(y_test, y_pred_optimal)),
    'recall_optimal': float(recall_score(y_test, y_pred_optimal)),
    'f1_optimal': float(f1_score(y_test, y_pred_optimal)),
    'true_positives': int(tp),
    'false_positives': int(fp),
    'true_negatives': int(tn),
    'false_negatives': int(fn),
    'test_samples': int(len(y_test)),
    'fraud_in_test': int((y_test == 1).sum()),
}

metrics_df = pd.DataFrame([metrics])
metrics_path = MODEL_SAVE_PATH / f'metrics_{timestamp}.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"Metrics saved: {metrics_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE - KEY TAKEAWAYS")
print("=" * 80)
print(f"""
1. SMOTE was applied to training data only (never to test data)
   - Original train fraud rate: {(y_train == 1).sum() / len(y_train) * 100:.3f}%
   - After SMOTE: {(y_train_resampled == 1).sum() / len(y_train_resampled) * 100:.1f}%

2. Model Performance (PR-AUC is most important):
   - PR-AUC: {pr_auc:.4f} (primary metric for imbalanced data)
   - Optimal Threshold: {optimal_threshold:.4f}
   - Precision: {precision_score(y_test, y_pred_optimal):.4f} (of flagged cases, {precision_score(y_test, y_pred_optimal)*100:.1f}% are real fraud)
   - Recall: {recall_score(y_test, y_pred_optimal):.4f} (we catch {recall_score(y_test, y_pred_optimal)*100:.1f}% of all fraud)

3. Why Accuracy is Useless:
   - Accuracy: {accuracy:.4f} (looks good but meaningless!)
   - A dumb "always predict non-fraud" model gets 99.83% accuracy
   - We must use Precision/Recall/PR-AUC instead

4. Next Steps:
   - Generate synthetic customer profiles (Phase 3)
   - Build agent investigation layer (Phase 4)
   - The model is working, now we add the differentiator!

Model saved to: {model_path}
""")

print("=" * 80)
