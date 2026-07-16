"""
Known Fraud Cases Generator

Generates a database of confirmed fraud cases for RAG similarity search.
These cases are embedded and stored in the vector database for the agent
to retrieve similar past fraud patterns when investigating new transactions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
import json

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_SYNTHETIC = PROJECT_ROOT / 'data' / 'synthetic'
OUTPUT_PATH = DATA_SYNTHETIC

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Fraud types and their characteristics
FRAUD_TYPES = {
    'card_theft': {
        'description': 'Physical card stolen and used for unauthorized purchases',
        'risk_factors': ['unusual_location', 'high_velocity', 'large_amount', 'unusual_merchant'],
        'typical_amount_range': (100, 5000),
        'pattern': 'Sudden high-value purchases in new locations shortly after theft'
    },
    'account_takeover': {
        'description': 'Fraudster gains access to victim\'s online account',
        'risk_factors': ['new_device', 'location_change', 'password_reset', 'new_payee'],
        'typical_amount_range': (50, 10000),
        'pattern': 'Login from new device/location followed by quick fund transfers or purchases'
    },
    'synthetic_identity': {
        'description': 'Fake identity created using real and fake information',
        'risk_factors': ['new_account', 'no_history', 'high_risk_merchant', 'velocity'],
        'typical_amount_range': (500, 15000),
        'pattern': 'New account with rapid escalation of transaction amounts'
    },
    'friendly_fraud': {
        'description': 'Customer disputes legitimate transaction',
        'risk_factors': ['dispute_filed', 'high_value_item', 'delivery_confirmation', 'chargeback'],
        'typical_amount_range': (50, 2000),
        'pattern': 'Customer claims non-receipt or unauthorized charge for legitimate purchase'
    },
    'card_not_present': {
        'description': 'Online/phone fraud without physical card',
        'risk_factors': ['online_merchant', 'high_velocity', 'different_billing_shipping', 'vpn_used'],
        'typical_amount_range': (20, 3000),
        'pattern': 'Multiple online purchases in short timeframe from different merchants'
    },
    'skimming': {
        'description': 'Card data stolen via skimming device at ATM or POS',
        'risk_factors': ['atm_withdrawal', 'nearby_location', 'small_then_large', 'time_pattern'],
        'typical_amount_range': (20, 2000),
        'pattern': 'Small test transactions followed by larger withdrawals at nearby ATMs'
    },
}

# Sample investigation notes for each fraud type
INVESTIGATION_NOTES = {
    'card_theft': [
        'Victim reported wallet stolen on {date}. Card used within 30 minutes.',
        'Multiple contactless payments under $100 limit at nearby merchants.',
        'Victim confirmed card was in wallet until reported stolen.',
    ],
    'account_takeover': [
        'Login from new device in different state at {time}.',
        'Password changed and email updated within 5 minutes of login.',
        'Two-factor authentication bypassed via SIM swap.',
    ],
    'synthetic_identity': [
        'Account opened with mix of real and fake PII.',
        'Credit history shows sudden appearance with no prior records.',
        'Identity linked to known synthetic identity ring.',
    ],
    'friendly_fraud': [
        'Customer claims item not received but delivery confirmation shows delivered.',
        'Chargeback filed for digital goods already consumed.',
        'Customer admits to family member making purchase.',
    ],
    'card_not_present': [
        'Multiple online purchases across different merchants in 2-hour window.',
        'Billing address differs from shipping address.',
        'Transactions from high-risk IP addresses or VPN exit nodes.',
    ],
    'skimming': [
        'Small $1 test transaction at gas station followed by $500 ATM withdrawals.',
        'Card used at compromised ATM identified in skimming investigation.',
        'Sequential withdrawals at multiple ATMs within 10-mile radius.',
    ],
}


def load_transaction_history():
    """Load the linked transaction history."""
    txn_path = DATA_SYNTHETIC / 'transaction_history.csv'
    df = pd.read_csv(txn_path)
    print(f"Loaded {len(df):,} transactions from history")
    return df


def load_customers_enriched():
    """Load enriched customer profiles."""
    cust_path = DATA_SYNTHETIC / 'customers_enriched.csv'
    df = pd.read_csv(cust_path)
    print(f"Loaded {len(df):,} enriched customers")
    return df


def select_fraud_transactions(transactions_df, num_cases_per_type=30):
    """
    Select actual fraud transactions from the dataset to use as known fraud cases.

    We'll use the actual fraud transactions from the Kaggle dataset and enhance them
    with fraud type classifications and investigation details.
    """
    print(f"\nSelecting fraud transactions for known cases...")

    # Get all fraud transactions
    fraud_txns = transactions_df[transactions_df['Class'] == 1].copy()
    print(f"Total fraud transactions available: {len(fraud_txns):,}")

    fraud_cases = []
    case_id = 0

    # Distribute fraud cases across types
    for fraud_type, config in FRAUD_TYPES.items():
        num_cases = num_cases_per_type

        # Randomly sample fraud transactions for this type
        if len(fraud_txns) >= num_cases:
            selected = fraud_txns.sample(n=num_cases, replace=False, random_state=RANDOM_SEED + hash(fraud_type) % 1000)
        else:
            selected = fraud_txns.sample(n=num_cases, replace=True, random_state=RANDOM_SEED + hash(fraud_type) % 1000)

        for _, txn in selected.iterrows():
            # Get customer info
            customer_id = txn['customer_id']

            # Generate investigation notes
            note_template = random.choice(INVESTIGATION_NOTES[fraud_type])
            investigation_note = note_template.format(
                date=datetime.now().strftime('%Y-%m-%d'),
                time=datetime.now().strftime('%H:%M')
            )

            # Create description for embedding
            description = (
                f"Fraud Type: {fraud_type}. "
                f"Pattern: {config['pattern']}. "
                f"Amount: ${txn['Amount']:.2f}. "
                f"Risk factors: {', '.join(config['risk_factors'])}. "
                f"Investigation: {investigation_note}"
            )

            # Create risk factors list
            risk_factors = config['risk_factors'].copy()
            # Add some transaction-specific risk factors
            if txn.get('deviation_from_baseline', 0) > 3:
                risk_factors.append('extreme_amount_deviation')
            if txn.get('time_since_last_txn', 0) < 1:
                risk_factors.append('high_velocity')
            if txn.get('transaction_number', 1) == 1:
                risk_factors.append('first_transaction')

            fraud_case = {
                'case_id': f"FRAUD_{case_id:06d}",
                'customer_id': txn['customer_id'],
                'transaction_id': f"TXN_{int(txn.get('transaction_number', case_id)):08d}",
                'fraud_type': fraud_type,
                'description': description,
                'risk_factors': json.dumps(risk_factors),
                'amount': float(txn['Amount']),
                'timestamp': txn.get('Time', 0),
                'confirmed_fraud': True,
                'investigation_notes': investigation_note,
                'original_class': int(txn['Class']),
                'deviation_from_baseline': float(txn.get('deviation_from_baseline', 0)),
                'time_since_last_txn': float(txn.get('time_since_last_txn', 0)),
            }

            fraud_cases.append(fraud_case)
            case_id += 1

            # Remove selected transaction from pool to avoid duplicates
            fraud_txns = fraud_txns[fraud_txns.index != txn.name]

    print(f"Created {len(fraud_cases)} known fraud cases")
    return fraud_cases


def generate_synthetic_fraud_cases(num_synthetic=50):
    """
    Generate additional synthetic fraud cases to supplement real ones.
    These fill in gaps and provide more variety for the RAG system.
    """
    print(f"\nGenerating {num_synthetic} synthetic fraud cases...")

    synthetic_cases = []
    base_id = 10000  # Start after real cases

    for i in range(num_synthetic):
        fraud_type = random.choice(list(FRAUD_TYPES.keys()))
        config = FRAUD_TYPES[fraud_type]

        # Generate realistic amount
        min_amt, max_amt = config['typical_amount_range']
        amount = random.uniform(min_amt, max_amt)

        # Generate timestamp within the 48-hour dataset window
        timestamp = random.uniform(0, 172800)

        # Create synthetic customer ID
        customer_id = f"CUST_{random.randint(0, 29999):06d}"

        note_template = random.choice(INVESTIGATION_NOTES[fraud_type])
        investigation_note = f"[SYNTHETIC] {note_template.format(date=datetime.now().strftime('%Y-%m-%d'), time=datetime.now().strftime('%H:%M'))}"

        description = (
            f"[SYNTHETIC] Fraud Type: {fraud_type}. "
            f"Pattern: {config['pattern']}. "
            f"Amount: ${amount:.2f}. "
            f"Risk factors: {', '.join(config['risk_factors'])}."
        )

        risk_factors = config['risk_factors'].copy()

        case = {
            'case_id': f"SYNTH_{base_id + i:06d}",
            'customer_id': f"CUST_{random.randint(0, 29999):06d}",
            'transaction_id': f"SYN_TXN_{base_id + i:08d}",
            'fraud_type': fraud_type,
            'description': description,
            'risk_factors': json.dumps(config['risk_factors']),
            'amount': round(amount, 2),
            'timestamp': round(random.uniform(0, 172800), 2),
            'confirmed_fraud': True,
            'investigation_notes': f"[SYNTHETIC CASE] Generated for training RAG system. {config['pattern']}",
            'original_class': 1,
            'deviation_from_baseline': round(random.uniform(2.0, 8.0), 2),
            'time_since_last_txn': round(random.uniform(0.1, 5.0), 2),
        }

        synthetic_cases.append(case)

    print(f"Generated {len(synthetic_cases)} synthetic fraud cases")
    return synthetic_cases


def save_fraud_cases(fraud_cases):
    """Save fraud cases to CSV and JSON for different use cases."""
    print("\nSaving fraud cases...")

    # Save as CSV for database import
    df = pd.DataFrame(fraud_cases)
    csv_path = OUTPUT_PATH / 'known_fraud_cases.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(fraud_cases):,} fraud cases to: {csv_path}")

    # Save as JSON for vector database ingestion
    json_path = OUTPUT_PATH / 'known_fraud_cases.json'
    with open(json_path, 'w') as f:
        json.dump(fraud_cases, f, indent=2, default=str)
    print(f"Saved JSON for vector DB: {json_path}")

    # Print summary by fraud type
    df_all = pd.DataFrame(fraud_cases)
    print("\nFraud cases by type:")
    for ftype, count in df_all['fraud_type'].value_counts().items():
        print(f"  {ftype}: {count}")

    # Save summary
    summary = {
        'total_cases': len(fraud_cases),
        'by_type': df_all['fraud_type'].value_counts().to_dict(),
        'generated_at': datetime.now().isoformat(),
        'description': 'Known fraud cases for RAG similarity search in agent investigation layer'
    }
    summary_path = OUTPUT_PATH / 'fraud_cases_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved: {summary_path}")


def main():
    """Generate known fraud cases database."""
    print("=" * 80)
    print("KNOWN FRAUD CASES GENERATION FOR RAG SIMILARITY SEARCH")
    print("=" * 80)

    # Load existing data
    transactions_df = load_transaction_history()
    customers_df = load_customers_enriched()

    # Select real fraud transactions as known cases
    real_cases = select_fraud_transactions(transactions_df, num_cases_per_type=25)

    # Generate synthetic cases to supplement
    synthetic_cases = generate_synthetic_fraud_cases(num_synthetic=50)

    # Combine all cases
    all_cases = real_cases + synthetic_cases

    # Save everything
    save_fraud_cases(all_cases)

    print("\n" + "=" * 80)
    print("KNOWN FRAUD CASES GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal cases: {len(real_cases) + len(synthetic_cases):,}")
    print(f"  Real fraud cases: {len(real_cases):,}")
    print(f"  Synthetic cases: {len(synthetic_cases):,}")
    print(f"\nFiles created:")
    print(f"  - data/synthetic/known_fraud_cases.csv")
    print(f"  - data/synthetic/known_fraud_cases.json")
    print(f"  - data/synthetic/fraud_cases_summary.json")
    print(f"\nNext: Embed these cases into vector database (Chroma) for RAG")


if __name__ == "__main__":
    main()