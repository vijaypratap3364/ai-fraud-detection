"""
Customer Name and ID Generator

Generates realistic synthetic customer profiles with names, IDs, and basic attributes.
This provides the foundation for linking Kaggle transactions to customers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'synthetic'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Sample data for realistic customer generation
FIRST_NAMES = [
    'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
    'William', 'Barbara', 'David', 'Elizabeth', 'Richard', 'Susan', 'Joseph', 'Jessica',
    'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher', 'Nancy', 'Daniel', 'Lisa',
    'Matthew', 'Betty', 'Anthony', 'Margaret', 'Mark', 'Sandra', 'Donald', 'Ashley',
    'Steven', 'Kimberly', 'Paul', 'Emily', 'Andrew', 'Donna', 'Joshua', 'Michelle',
    'Kenneth', 'Dorothy', 'Kevin', 'Carol', 'Brian', 'Amanda', 'George', 'Melissa',
    'Edward', 'Deborah', 'Ronald', 'Stephanie', 'Timothy', 'Rebecca', 'Jason', 'Sharon',
    'Jeffrey', 'Laura', 'Ryan', 'Cynthia', 'Jacob', 'Kathleen', 'Gary', 'Amy',
]

LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
    'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas',
    'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White',
    'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young',
    'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores',
    'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell',
    'Carter', 'Roberts', 'Gomez', 'Phillips', 'Evans', 'Turner', 'Diaz', 'Parker',
]

US_CITIES = [
    'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ',
    'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA', 'Dallas, TX', 'San Jose, CA',
    'Austin, TX', 'Jacksonville, FL', 'Fort Worth, TX', 'Columbus, OH', 'Charlotte, NC',
    'San Francisco, CA', 'Indianapolis, IN', 'Seattle, WA', 'Denver, CO', 'Washington, DC',
    'Boston, MA', 'Nashville, TN', 'Detroit, MI', 'Portland, OR', 'Las Vegas, NV',
    'Memphis, TN', 'Louisville, KY', 'Baltimore, MD', 'Milwaukee, WI', 'Albuquerque, NM',
]


def generate_customer_id(index: int) -> str:
    """Generate a unique customer ID."""
    return f"CUST_{index:06d}"


def generate_name() -> str:
    """Generate a realistic full name."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}"


def generate_age() -> int:
    """
    Generate customer age with realistic distribution.

    Age distribution:
    - 18-30: 25% (young adults)
    - 31-50: 40% (middle-aged, highest spending)
    - 51-65: 25% (older adults)
    - 66-80: 10% (seniors)
    """
    rand = random.random()
    if rand < 0.25:
        return random.randint(18, 30)
    elif rand < 0.65:
        return random.randint(31, 50)
    elif rand < 0.90:
        return random.randint(51, 65)
    else:
        return random.randint(66, 80)


def generate_location() -> str:
    """Generate a US city and state."""
    return random.choice(US_CITIES)


def generate_account_created_date() -> datetime:
    """
    Generate account creation date.

    Accounts created between 1-5 years ago.
    """
    days_ago = random.randint(365, 365 * 5)
    return datetime.now() - timedelta(days=days_ago)


def generate_customers(num_customers: int) -> pd.DataFrame:
    """
    Generate synthetic customer profiles.

    Args:
        num_customers: Number of customer profiles to generate

    Returns:
        DataFrame with customer profiles
    """
    print(f"\nGenerating {num_customers:,} synthetic customer profiles...")

    customers = []
    for i in range(num_customers):
        customer = {
            'customer_id': generate_customer_id(i),
            'name': generate_name(),
            'age': generate_age(),
            'location': generate_location(),
            'account_created': generate_account_created_date(),
        }
        customers.append(customer)

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1:,} customers...")

    df = pd.DataFrame(customers)
    print(f"Generated {len(df):,} customer profiles")

    return df


def main():
    """Generate synthetic customers and save to CSV."""
    print("=" * 80)
    print("SYNTHETIC CUSTOMER GENERATION")
    print("=" * 80)

    # We have 284,807 transactions in the Kaggle dataset
    # Assume each customer has ~10 transactions on average
    # So we need ~28,000 customers
    NUM_CUSTOMERS = 30000

    print(f"\nTarget: {NUM_CUSTOMERS:,} customers")
    print("Rationale: ~10 transactions per customer for 284,807 transactions")

    # Generate customers
    customers_df = generate_customers(NUM_CUSTOMERS)

    # Display sample
    print("\n" + "-" * 80)
    print("Sample customers:")
    print("-" * 80)
    print(customers_df.head(10))

    # Statistics
    print("\n" + "-" * 80)
    print("Statistics:")
    print("-" * 80)
    print(f"Total customers: {len(customers_df):,}")
    print(f"Age distribution:")
    print(f"  18-30: {len(customers_df[customers_df['age'] <= 30]):,}")
    print(f"  31-50: {len(customers_df[(customers_df['age'] > 30) & (customers_df['age'] <= 50)]):,}")
    print(f"  51-65: {len(customers_df[(customers_df['age'] > 50) & (customers_df['age'] <= 65)]):,}")
    print(f"  66-80: {len(customers_df[customers_df['age'] > 65]):,}")

    # Save to CSV
    output_file = OUTPUT_PATH / 'customers.csv'
    customers_df.to_csv(output_file, index=False)
    print(f"\nCustomers saved to: {output_file}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("- Link these customers to Kaggle transactions")
    print("- Generate transaction history for each customer")
    print("- Calculate spending patterns and baselines")


if __name__ == "__main__":
    main()
