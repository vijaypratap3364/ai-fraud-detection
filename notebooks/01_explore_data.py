"""
Data Exploration - Kaggle Credit Card Fraud Dataset

This script performs initial exploration of the credit card fraud dataset
to understand its structure, distribution, and characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_theme(style='darkgrid')

# Paths
DATA_PATH = Path(__file__).parent.parent / 'data' / 'raw' / 'creditcard.csv'
OUTPUT_PATH = Path(__file__).parent.parent / 'data' / 'processed'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("KAGGLE CREDIT CARD FRAUD DATASET - INITIAL EXPLORATION")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} transactions")

# Basic info
print("\n2. Dataset Structure:")
print("-" * 80)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes.value_counts()}")

# Memory usage
memory_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"\nMemory usage: {memory_mb:.2f} MB")

# Check for missing values
print("\n3. Missing Values:")
print("-" * 80)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values found")
else:
    print(f"Missing values:\n{missing[missing > 0]}")

# Class distribution (THE MOST IMPORTANT METRIC)
print("\n4. Class Distribution (Fraud vs Non-Fraud):")
print("-" * 80)
class_counts = df['Class'].value_counts()
fraud_count = class_counts[1]
non_fraud_count = class_counts[0]
fraud_rate = (fraud_count / len(df)) * 100

print(f"Non-Fraud (Class 0): {non_fraud_count:,} ({(non_fraud_count/len(df)*100):.3f}%)")
print(f"Fraud (Class 1):     {fraud_count:,} ({fraud_rate:.3f}%)")
print(f"\nImbalance Ratio: 1:{int(non_fraud_count/fraud_count)}")
print(f"\nWARNING: HIGHLY IMBALANCED DATASET - This is why accuracy is meaningless!")
print(f"    A model that predicts 'not fraud' for everything gets {(non_fraud_count/len(df)*100):.3f}% accuracy")
print(f"    but catches ZERO fraud. We need Precision/Recall/PR-AUC instead.")

# Feature statistics
print("\n5. Feature Statistics:")
print("-" * 80)
print("\nTime feature:")
print(f"  Range: {df['Time'].min():.0f} to {df['Time'].max():.0f} seconds")
print(f"  Duration: {df['Time'].max() / 3600:.1f} hours")

print("\nAmount feature (transaction values):")
amount_stats = df['Amount'].describe()
print(f"  Mean: ${amount_stats['mean']:.2f}")
print(f"  Median: ${amount_stats['50%']:.2f}")
print(f"  Std: ${amount_stats['std']:.2f}")
print(f"  Min: ${amount_stats['min']:.2f}")
print(f"  Max: ${amount_stats['max']:.2f}")

print("\nAmount by class:")
fraud_amount = df[df['Class'] == 1]['Amount'].describe()
non_fraud_amount = df[df['Class'] == 0]['Amount'].describe()
print(f"  Fraud transactions - Mean: ${fraud_amount['mean']:.2f}, Median: ${fraud_amount['50%']:.2f}")
print(f"  Non-fraud transactions - Mean: ${non_fraud_amount['mean']:.2f}, Median: ${non_fraud_amount['50%']:.2f}")

print("\nPCA features (V1-V28):")
print(f"  28 anonymized features from PCA transformation")
print(f"  These are the main features for fraud detection")

# Feature correlations with fraud
print("\n6. Feature Correlations with Fraud:")
print("-" * 80)
correlations = df.corr()['Class'].drop('Class').sort_values(ascending=False)
print("\nTop 10 positive correlations:")
print(correlations.head(10))
print("\nTop 10 negative correlations:")
print(correlations.tail(10))

# Time-based analysis
print("\n7. Temporal Analysis:")
print("-" * 80)
df['Hour'] = (df['Time'] / 3600).astype(int)
fraud_by_hour = df[df['Class'] == 1].groupby('Hour').size()
print(f"\nFraud transactions per hour:")
for hour, count in fraud_by_hour.items():
    print(f"  Hour {hour:2d}: {count:3d} frauds")

# Save summary statistics
print("\n8. Saving Summary Statistics:")
print("-" * 80)

summary = {
    'total_transactions': len(df),
    'fraud_count': int(fraud_count),
    'non_fraud_count': int(non_fraud_count),
    'fraud_rate_percent': float(fraud_rate),
    'imbalance_ratio': int(non_fraud_count / fraud_count),
    'features': list(df.columns),
    'time_range_hours': float(df['Time'].max() / 3600),
    'amount_mean': float(df['Amount'].mean()),
    'amount_std': float(df['Amount'].std()),
    'memory_mb': float(memory_mb)
}

summary_df = pd.DataFrame([summary])
summary_path = OUTPUT_PATH / 'dataset_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary to: {summary_path}")

# Key findings
print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)
print(f"""
1. Dataset is HIGHLY IMBALANCED:
   - Only {fraud_rate:.3f}% of transactions are fraud
   - Imbalance ratio: 1:{int(non_fraud_count/fraud_count)}
   - This is why we need SMOTE/class weighting and PR-AUC metrics

2. Dataset is CLEAN:
   - No missing values
   - All features are numerical (Time, Amount, V1-V28, Class)
   - Ready for ML without cleaning

3. PCA Features:
   - V1-V28 are anonymized via PCA (we can't interpret them)
   - These are the main signals for fraud detection
   - Time and Amount are the only interpretable features

4. Next Steps:
   Data exploration complete
   -> Train XGBoost with proper imbalance handling
   -> Generate synthetic customer profiles
   -> Build agent investigation layer
""")

print("\nData exploration complete!")
print("=" * 80)
