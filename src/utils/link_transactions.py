"""
Transaction-to-Customer Linker

Links Kaggle transactions to synthetic customers and builds transaction history.
This enables the agent investigation layer to query customer spending patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_SYNTHETIC = PROJECT_ROOT / 'data' / 'synthetic'
OUTPUT_PATH = DATA_SYNTHETIC

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_data():
    """Load Kaggle transactions and synthetic customers."""
    print("\n1. Loading datasets...")

    # Load Kaggle credit card fraud dataset
    kaggle_path = DATA_RAW / 'creditcard.csv'
    transactions_df = pd.read_csv(kaggle_path)
    print(f"   Loaded {len(transactions_df):,} Kaggle transactions")

    # Load synthetic customers
    customers_path = DATA_SYNTHETIC / 'customers.csv'
    customers_df = pd.read_csv(customers_path)
    print(f"   Loaded {len(customers_df):,} synthetic customers")

    return transactions_df, customers_df


def assign_customers_to_transactions(transactions_df, customers_df):
    """
    Assign each transaction to a customer.

    Strategy:
    - Distribute transactions across customers with variation
    - Some customers have many transactions, some have few (realistic)
    - Use weighted random assignment to create realistic distribution
    """
    print("\n2. Assigning transactions to customers...")

    num_transactions = len(transactions_df)
    num_customers = len(customers_df)

    # Create weighted customer pool
    # Some customers are more active than others
    # Use power law distribution: most customers have few transactions, some have many
    weights = np.random.power(0.5, num_customers)
    weights = weights / weights.sum()

    # Randomly assign each transaction to a customer
    customer_indices = np.random.choice(
        num_customers,
        size=num_transactions,
        replace=True,
        p=weights
    )

    # Add customer_id to transactions
    transactions_df['customer_id'] = customers_df.iloc[customer_indices]['customer_id'].values

    # Transactions per customer
    txn_per_customer = transactions_df.groupby('customer_id').size()
    print(f"   Transactions per customer:")
    print(f"     Mean: {txn_per_customer.mean():.1f}")
    print(f"     Median: {txn_per_customer.median():.1f}")
    print(f"     Min: {txn_per_customer.min()}")
    print(f"     Max: {txn_per_customer.max()}")

    return transactions_df


def calculate_customer_baselines(transactions_df, customers_df):
    """
    Calculate baseline spending patterns for each customer.

    These baselines enable the agent to detect deviations during investigation.
    """
    print("\n3. Calculating customer spending baselines...")

    # Group transactions by customer
    customer_stats = transactions_df.groupby('customer_id').agg({
        'Amount': ['mean', 'median', 'std', 'count'],
        'Class': 'sum'  # Number of fraud transactions
    }).reset_index()

    # Flatten column names
    customer_stats.columns = [
        'customer_id',
        'avg_transaction_amount',
        'median_transaction_amount',
        'std_transaction_amount',
        'total_transactions',
        'fraud_count'
    ]

    # Merge with customer profiles
    customers_enriched = customers_df.merge(customer_stats, on='customer_id', how='left')

    # Fill NaN for customers with no transactions (shouldn't happen but handle it)
    customers_enriched = customers_enriched.fillna({
        'avg_transaction_amount': 0,
        'median_transaction_amount': 0,
        'std_transaction_amount': 0,
        'total_transactions': 0,
        'fraud_count': 0
    })

    # Add derived metrics
    customers_enriched['has_fraud_history'] = customers_enriched['fraud_count'] > 0
    customers_enriched['avg_monthly_spend'] = customers_enriched['avg_transaction_amount'] * 30  # Rough estimate

    print(f"   Calculated baselines for {len(customers_enriched):,} customers")
    print(f"   Customers with fraud history: {customers_enriched['has_fraud_history'].sum():,}")

    return customers_enriched


def add_transaction_metadata(transactions_df):
    """
    Add metadata to transactions for agent investigation.

    Calculates:
    - Transaction sequence number for customer
    - Time since last transaction
    - Deviation from customer baseline
    """
    print("\n4. Adding transaction metadata...")

    # Sort by customer and time
    transactions_df = transactions_df.sort_values(['customer_id', 'Time'])

    # Add transaction sequence number for each customer
    transactions_df['transaction_number'] = transactions_df.groupby('customer_id').cumcount() + 1

    # Calculate time since last transaction (in hours)
    transactions_df['time_since_last_txn'] = transactions_df.groupby('customer_id')['Time'].diff() / 3600
    transactions_df['time_since_last_txn'] = transactions_df['time_since_last_txn'].fillna(0)

    # Calculate customer-specific baseline for deviation metric
    customer_baselines = transactions_df.groupby('customer_id')['Amount'].agg(['mean', 'std']).reset_index()
    customer_baselines.columns = ['customer_id', 'customer_avg_amount', 'customer_std_amount']

    transactions_df = transactions_df.merge(customer_baselines, on='customer_id', how='left')

    # Calculate deviation from baseline (in standard deviations)
    transactions_df['deviation_from_baseline'] = (
        (transactions_df['Amount'] - transactions_df['customer_avg_amount']) /
        (transactions_df['customer_std_amount'] + 1e-6)  # Avoid division by zero
    )

    # Clean up temporary columns
    transactions_df = transactions_df.drop(['customer_avg_amount', 'customer_std_amount'], axis=1)

    print(f"   Added metadata for {len(transactions_df):,} transactions")

    return transactions_df


def save_outputs(transactions_df, customers_df):
    """Save enriched datasets."""
    print("\n5. Saving outputs...")

    # Save transaction history with customer linkage
    txn_output = OUTPUT_PATH / 'transaction_history.csv'
    transactions_df.to_csv(txn_output, index=False)
    print(f"   Transaction history saved: {txn_output}")
    print(f"     {len(transactions_df):,} transactions")

    # Save enriched customer profiles
    customer_output = OUTPUT_PATH / 'customers_enriched.csv'
    customers_df.to_csv(customer_output, index=False)
    print(f"   Enriched customers saved: {customer_output}")
    print(f"     {len(customers_df):,} customers with baselines")


def main():
    """Link Kaggle transactions to synthetic customers."""
    print("=" * 80)
    print("TRANSACTION-TO-CUSTOMER LINKING")
    print("=" * 80)

    # Load data
    transactions_df, customers_df = load_data()

    # Assign customers to transactions
    transactions_df = assign_customers_to_transactions(transactions_df, customers_df)

    # Calculate customer baselines
    customers_enriched = calculate_customer_baselines(transactions_df, customers_df)

    # Add transaction metadata
    transactions_df = add_transaction_metadata(transactions_df)

    # Save outputs
    save_outputs(transactions_df, customers_enriched)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal transactions: {len(transactions_df):,}")
    print(f"Total customers: {len(customers_enriched):,}")
    print(f"Fraud transactions: {transactions_df['Class'].sum():,}")
    print(f"Customers with fraud history: {customers_enriched['has_fraud_history'].sum():,}")

    print("\nAgent investigation capabilities enabled:")
    print("  - Query customer transaction history")
    print("  - Calculate deviation from baseline spending")
    print("  - Identify spending pattern changes")
    print("  - Track transaction velocity (time since last txn)")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
