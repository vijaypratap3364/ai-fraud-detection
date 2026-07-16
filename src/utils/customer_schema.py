"""
Customer Profile Schema

Defines the data structures for synthetic customer profiles.
This enables the agent investigation layer to query customer history.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class CustomerProfile:
    """
    Represents a synthetic customer with transaction history and spending patterns.

    This is used by the agent investigation layer to:
    - Fetch customer transaction history
    - Identify spending pattern deviations
    - Calculate baseline metrics for risk assessment
    """

    customer_id: str
    name: str
    age: int
    location: str  # City, State format
    account_created: datetime

    # Spending pattern characteristics
    avg_monthly_spend: float
    typical_transaction_amount: float
    transaction_frequency: float  # Transactions per day
    preferred_categories: List[str]

    # Risk indicators
    fraud_history: bool = False
    risk_score: float = 0.0

    # Metadata
    total_transactions: int = 0
    last_transaction_date: Optional[datetime] = None

    def __repr__(self) -> str:
        return f"Customer({self.customer_id}, {self.name}, ${self.avg_monthly_spend:.2f}/mo)"


@dataclass
class TransactionRecord:
    """
    Links a transaction from the Kaggle dataset to a synthetic customer.
    """

    transaction_id: str
    customer_id: str
    timestamp: datetime
    amount: float

    # Kaggle dataset features (V1-V28, Time, Amount, Class)
    kaggle_features: dict

    # Derived features for agent investigation
    is_fraud: bool
    deviation_from_baseline: float  # How many std deviations from customer's average
    time_since_last_transaction: Optional[float] = None  # Hours
    location_change: bool = False  # Did customer location change?

    def __repr__(self) -> str:
        fraud_marker = "[FRAUD]" if self.is_fraud else ""
        return f"Transaction({self.transaction_id}, ${self.amount:.2f} {fraud_marker})"


@dataclass
class KnownFraudCase:
    """
    Represents a confirmed fraud case for RAG similarity search.

    These cases are embedded and stored in the vector database.
    The agent retrieves similar cases when investigating a flagged transaction.
    """

    case_id: str
    customer_id: str
    transaction_id: str
    fraud_type: str  # e.g., "card_theft", "account_takeover", "synthetic_identity"

    # Pattern description (for embedding)
    description: str
    risk_factors: List[str]

    # Transaction details
    amount: float
    timestamp: datetime

    # Investigation outcome
    confirmed_fraud: bool
    investigation_notes: str

    def __repr__(self) -> str:
        return f"FraudCase({self.case_id}, {self.fraud_type}, ${self.amount:.2f})"


# Constants for customer generation
SPENDING_PATTERNS = {
    'low_spender': {'avg_monthly': 500, 'typical_txn': 25, 'freq': 2.0},
    'medium_spender': {'avg_monthly': 2000, 'typical_txn': 80, 'freq': 5.0},
    'high_spender': {'avg_monthly': 8000, 'typical_txn': 300, 'freq': 8.0},
}

TRANSACTION_CATEGORIES = [
    'groceries',
    'gas_station',
    'restaurant',
    'online_shopping',
    'electronics',
    'travel',
    'entertainment',
    'utilities',
    'healthcare',
]

FRAUD_TYPES = [
    'card_theft',           # Stolen physical card
    'account_takeover',     # Compromised online account
    'synthetic_identity',   # Fake identity using real/fake info
    'friendly_fraud',       # Customer disputes legitimate transaction
    'card_not_present',     # Online fraud without physical card
]
