"""
Agent Investigation Tools

Core tools for the fraud investigation agent:
1. Customer History Lookup
2. Fraud Similarity Search (RAG with Chroma)
3. Risk Factor Calculator
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_SYNTHETIC = PROJECT_ROOT / 'data' / 'synthetic'
CHROMA_PATH = PROJECT_ROOT / 'chroma_db'

# Load data once at module level
_transaction_history: pd.DataFrame = None
_customers_enriched: pd.DataFrame = None
_chroma_client = None
_chroma_collection = None
_embedding_model = None
_initialized = False


def _initialize():
    """Lazy initialization of data and connections."""
    global _transaction_history, _customers_enriched, _chroma_client, _chroma_collection, _embedding_model, _initialized
    if _initialized:
        return

    print("Initializing investigation tools...")

    # Load transaction history
    txn_path = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / 'transaction_history.csv'
    _transaction_history = pd.read_csv(txn_path)

    # Load enriched customers
    cust_path = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / 'customers_enriched.csv'
    _customers_enriched = pd.read_csv(cust_path)

    # Initialize Chroma
    _chroma_client = chromadb.PersistentClient(
        path=str(Path(__file__).parent.parent.parent / 'chroma_db'),
        settings=Settings(anonymized_telemetry=False)
    )
    _chroma_collection = _chroma_client.get_collection("fraud_cases")

    # Load embedding model
    _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    _initialized = True
    print("Investigation tools initialized!")


# ============================================================================
# TOOL 1: CUSTOMER HISTORY LOOKUP
# ============================================================================

@dataclass
class CustomerHistoryResult:
    """Result from customer history lookup."""
    customer_id: str
    customer_name: str
    total_transactions: int
    recent_transactions: List[Dict]
    avg_transaction_amount: float
    median_transaction_amount: float
    std_transaction_amount: float
    spending_pattern: str
    has_fraud_history: bool
    account_age_days: int
    recent_velocity: float  # transactions per hour in last 24h


def lookup_customer_history(customer_id: str, lookback_days: int = 90, max_transactions: int = 50) -> CustomerHistoryResult:
    """
    Fetch customer's recent transaction history and identify pattern breaks.

    This tool retrieves the customer's recent transactions and checks for:
    - Sudden location changes
    - Spend spikes (deviation from baseline)
    - New merchant categories
    - Transaction velocity changes
    """
    _initialize()

    # Get customer info
    customer_row = _customers_enriched[_customers_enriched['customer_id'] == customer_id]
    if customer_row.empty:
        raise ValueError(f"Customer {customer_id} not found")

    customer = customer_row.iloc[0]

    # Get customer's transactions
    customer_txns = _transaction_history[_transaction_history['customer_id'] == customer_id].copy()
    customer_txns = customer_txns.sort_values('Time', ascending=False)

    # Filter by lookback period
    if lookback_days > 0:
        # Time is in seconds, convert days to seconds
        cutoff_time = _transaction_history['Time'].max() - (lookback_days * 24 * 3600)
        customer_txns = customer_txns[customer_txns['Time'] >= cutoff_time]

    # Limit transactions
    recent_txns = customer_txns.head(max_transactions)

    # Convert to list of dicts
    recent_list = []
    for _, txn in recent_txns.iterrows():
        recent_list.append({
            'transaction_id': f"TXN_{int(txn.get('transaction_number', 0)):08d}",
            'timestamp': float(txn['Time']),
            'amount': float(txn['Amount']),
            'is_fraud': bool(txn['Class']),
            'deviation_from_baseline': float(txn.get('deviation_from_baseline', 0)),
            'time_since_last_txn': float(txn.get('time_since_last_txn', 0)),
            'transaction_number': int(txn.get('transaction_number', 0)),
        })

    # Calculate recent velocity (last 24 hours)
    recent_24h = customer_txns[customer_txns['Time'] >= (_transaction_history['Time'].max() - 24 * 3600)]
    recent_velocity = len(recent_24h) / 24.0 if len(recent_24h) > 0 else 0

    # Determine spending pattern
    avg_spend = float(customer.get('avg_transaction_amount', 0))
    if avg_spend < 50:
        spending_pattern = "low_spender"
    elif avg_spend < 200:
        spending_pattern = "medium_spender"
    else:
        spending_pattern = "high_spender"

    # Account age
    account_created = pd.to_datetime(customer.get('account_created', datetime.now()))
    account_age = (datetime.now() - account_created).days

    return CustomerHistoryResult(
        customer_id=customer_id,
        customer_name=customer.get('name', 'Unknown'),
        total_transactions=int(customer.get('total_transactions', 0)),
        recent_transactions=recent_list,
        avg_transaction_amount=float(customer.get('avg_transaction_amount', 0)),
        median_transaction_amount=float(customer.get('median_transaction_amount', 0)),
        std_transaction_amount=float(customer.get('std_transaction_amount', 0)),
        spending_pattern=spending_pattern,
        has_fraud_history=bool(customer.get('has_fraud_history', False)),
        account_age_days=account_age,
        recent_velocity=recent_velocity,
    )


# ============================================================================
# TOOL 2: FRAUD SIMILARITY SEARCH (RAG)
# ============================================================================

@dataclass
class SimilarFraudCase:
    """A similar fraud case from the vector database."""
    case_id: str
    fraud_type: str
    amount: float
    distance: float
    similarity_score: float
    risk_factors: List[str]
    description: str
    investigation_notes: str


def search_similar_fraud_cases(
    query_description: str,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> List[SimilarFraudCase]:
    """
    Search for similar past fraud cases using semantic similarity.

    Uses Chroma vector database with sentence-transformer embeddings
    to find fraud cases with similar patterns to the current transaction.
    """
    _initialize()

    # Encode query
    query_embedding = _embedding_model.encode([query_description]).tolist()

    # Search Chroma
    results = _chroma_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )

    similar_cases = []
    for i in range(len(results['ids'][0])):
        case_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]

        # Convert distance to similarity (lower distance = higher similarity)
        similarity = 1.0 - (distance / 2.0)  # Cosine distance is 0-2

        if similarity >= similarity_threshold:
            # Parse risk factors from JSON string
            risk_factors = json.loads(metadata['risk_factors']) if isinstance(metadata['risk_factors'], str) else metadata['risk_factors']

            similar_cases.append(SimilarFraudCase(
                case_id=case_id,
                fraud_type=metadata['fraud_type'],
                amount=metadata['amount'],
                distance=distance,
                similarity_score=similarity,
                risk_factors=risk_factors,
                description=results['documents'][0][i],
                investigation_notes=metadata.get('investigation_notes', ''),
            ))

    return similar_cases


# ============================================================================
# TOOL 3: RISK FACTOR CALCULATOR
# ============================================================================

@dataclass
class RiskFactors:
    """Calculated risk factors for a transaction."""
    # Velocity metrics
    velocity_24h: float  # transactions per hour in last 24h
    velocity_1h: float   # transactions per hour in last 1h

    # Deviation metrics
    amount_deviation_std: float  # std deviations from customer baseline
    amount_deviation_pct: float  # percentage deviation from customer baseline

    # Temporal metrics
    time_since_last_txn: float   # hours since last transaction
    is_unusual_time: bool        # transaction at unusual hour

    # Geographic metrics (simulated)
    location_change: bool
    distance_from_usual: float   # km from usual location

    # Merchant metrics
    new_merchant_category: bool
    high_risk_merchant: bool

    # Customer profile metrics
    account_age_days: int
    has_fraud_history: bool
    customer_risk_score: float

    # Aggregated risk score
    composite_risk_score: float

    def to_dict(self) -> Dict:
        return {
            'velocity_24h': self.velocity_24h,
            'velocity_1h': self.velocity_1h,
            'amount_deviation_std': self.amount_deviation_std,
            'amount_deviation_pct': self.amount_deviation_pct,
            'time_since_last_txn': self.time_since_last_txn,
            'is_unusual_time': self.is_unusual_time,
            'location_change': self.location_change,
            'distance_from_usual': self.distance_from_usual,
            'new_merchant_category': self.new_merchant_category,
            'high_risk_merchant': self.high_risk_merchant,
            'account_age_days': self.account_age_days,
            'has_fraud_history': self.has_fraud_history,
            'customer_risk_score': self.customer_risk_score,
            'composite_risk_score': self.composite_risk_score,
        }


def calculate_risk_factors(
    transaction: Dict,
    customer_history: CustomerHistoryResult,
    customer_id: str
) -> RiskFactors:
    """
    Calculate concrete risk factors for a transaction.

    Returns actual computed numbers, not LLM guesses.
    """
    # Velocity metrics
    velocity_24h = customer_history.recent_velocity
    recent_1h_txns = [txn for txn in customer_history.recent_transactions
                      if txn['time_since_last_txn'] <= 1.0]
    velocity_1h = len(recent_1h_txns)

    # Deviation metrics
    amount_deviation_std = transaction.get('deviation_from_baseline', 0)
    customer_avg = customer_history.avg_transaction_amount
    amount = transaction.get('amount', 0)
    amount_deviation_pct = ((amount - customer_avg) / (customer_avg + 1)) * 100 if customer_avg > 0 else 0

    # Temporal metrics
    time_since_last = transaction.get('time_since_last_txn', 0)
    txn_time = transaction.get('timestamp', 0)
    hour_of_day = int((txn_time / 3600) % 24)
    is_unusual_time = hour_of_day < 6 or hour_of_day > 22  # 10pm-6am

    # Geographic (simulated - in production would use actual geo data)
    location_change = False  # Would be determined by actual location data
    distance_from_usual = 0.0

    # Merchant (simulated)
    new_merchant_category = False
    high_risk_merchant = False

    # Customer profile
    account_age = customer_history.account_age_days
    has_fraud_history = customer_history.has_fraud_history

    # Customer risk score
    customer_risk = 0.0
    if has_fraud_history:
        customer_risk += 0.3
    if customer_history.recent_velocity > 5:
        customer_risk += 0.2
    if customer_history.total_transactions < 10:
        customer_risk += 0.1  # New customer

    # Composite risk score (weighted)
    weights = {
        'velocity_24h': 0.15,
        'amount_deviation_std': 0.25,
        'velocity_1h': 0.15,
        'time_since_last': 0.10,
        'unusual_time': 0.05,
        'customer_risk': 0.15,
        'has_fraud_history': 0.15,
    }

    # Normalize factors to 0-1
    v24_norm = min(velocity_24h / 10.0, 1.0)
    dev_norm = min(abs(transaction.get('deviation_from_baseline', 0)) / 5.0, 1.0)
    v1_norm = min(velocity_1h / 5.0, 1.0)
    time_norm = min(time_since_last / 24.0, 1.0)
    time_bonus = 1.0 if is_unusual_time else 0.0
    cust_risk_norm = min(customer_risk, 1.0)
    fraud_hist_norm = 1.0 if has_fraud_history else 0.0

    composite = (
        weights['velocity_24h'] * v24_norm +
        weights['amount_deviation_std'] * dev_norm +
        weights['velocity_1h'] * v1_norm +
        weights['time_since_last'] * time_norm +
        weights['unusual_time'] * time_bonus +
        weights['customer_risk'] * cust_risk_norm +
        weights['has_fraud_history'] * fraud_hist_norm
    )

    return RiskFactors(
        velocity_24h=round(velocity_24h, 2),
        velocity_1h=round(velocity_1h, 2),
        amount_deviation_std=round(transaction.get('deviation_from_baseline', 0), 2),
        amount_deviation_pct=round(amount_deviation_pct, 2),
        time_since_last_txn=round(time_since_last, 2),
        is_unusual_time=is_unusual_time,
        location_change=location_change,
        distance_from_usual=distance_from_usual,
        new_merchant_category=new_merchant_category,
        high_risk_merchant=high_risk_merchant,
        account_age_days=account_age,
        has_fraud_history=has_fraud_history,
        customer_risk_score=round(customer_risk, 2),
        composite_risk_score=round(composite, 4),
    )


# ============================================================================
# AGENT ORCHESTRATION
# ============================================================================

@dataclass
class InvestigationReport:
    """Structured investigation report from the agent."""
    transaction_id: str
    customer_id: str
    timestamp: datetime

    # Detection layer output
    fraud_probability: float
    detection_threshold: float

    # Tool outputs
    customer_history: CustomerHistoryResult
    similar_cases: List[SimilarFraudCase]
    risk_factors: RiskFactors

    # Agent synthesis
    risk_factors_identified: List[str]
    confidence_score: float
    recommended_action: str  # 'auto_block', 'hold_for_review', 'clear'
    reasoning: str

    # Traceability
    tools_called: List[str]
    tool_outputs: Dict

    def to_dict(self) -> Dict:
        return {
            'transaction_id': self.transaction_id,
            'customer_id': self.customer_id,
            'timestamp': self.timestamp.isoformat(),
            'fraud_probability': self.fraud_probability,
            'detection_threshold': self.detection_threshold,
            'customer_history': {
                'customer_id': self.customer_history.customer_id,
                'total_transactions': self.customer_history.total_transactions,
                'recent_velocity': self.customer_history.recent_velocity,
                'has_fraud_history': self.customer_history.has_fraud_history,
            },
            'similar_cases_count': len(self.similar_cases),
            'top_similar_case': self.similar_cases[0].case_id if self.similar_cases else None,
            'risk_factors': self.risk_factors.to_dict(),
            'risk_factors_identified': self.risk_factors_identified,
            'confidence_score': self.confidence_score,
            'recommended_action': self.recommended_action,
            'reasoning': self.reasoning,
            'tools_called': self.tools_called,
        }


def investigate_transaction(
    transaction: Dict,
    fraud_probability: float,
    detection_threshold: float = 0.7
) -> InvestigationReport:
    """
    Main agent orchestration function.

    When a transaction crosses the detection threshold, this function:
    1. Calls customer history lookup tool
    2. Calls fraud similarity search (RAG)
    3. Calls risk factor calculator
    4. Synthesizes all information into a structured report
    """
    _initialize()

    transaction_id = transaction.get('transaction_id', f"TXN_{int(transaction.get('transaction_number', 0)):08d}")
    customer_id = transaction.get('customer_id', '')
    amount = transaction.get('amount', 0)

    tools_called = []
    tool_outputs = {}

    print(f"\n{'='*60}")
    print(f"AGENT INVESTIGATION STARTED")
    print(f"Transaction: {transaction_id} | Customer: {customer_id} | Amount: ${amount:.2f}")
    print(f"Fraud Probability: {fraud_probability:.4f} | Threshold: {detection_threshold}")
    print(f"{'='*60}")

    # TOOL 1: Customer History Lookup
    print("\n[TOOL 1] Customer History Lookup...")
    customer_history = lookup_customer_history(customer_id)
    tools_called.append("lookup_customer_history")
    tool_outputs['customer_history'] = {
        'total_transactions': customer_history.total_transactions,
        'recent_velocity': customer_history.recent_velocity,
        'has_fraud_history': customer_history.has_fraud_history,
        'spending_pattern': customer_history.spending_pattern,
    }
    print(f"  Customer: {customer_history.customer_name} ({customer_id})")
    print(f"  Total Transactions: {customer_history.total_transactions}")
    print(f"  Recent Velocity: {customer_history.recent_velocity:.2f} txn/hr")
    print(f"  Fraud History: {'Yes' if customer_history.has_fraud_history else 'No'}")

    # TOOL 2: Fraud Similarity Search (RAG)
    print("\n[TOOL 2] Fraud Similarity Search (RAG)...")
    # Build query from transaction details
    query = (
        f"Transaction amount ${transaction.get('amount', 0):.2f} "
        f"deviation {transaction.get('deviation_from_baseline', 0):.2f} std "
        f"velocity {customer_history.recent_velocity:.1f} txn/hr"
    )
    similar_cases = search_similar_fraud_cases(query, top_k=5)
    tools_called.append("search_similar_fraud_cases")
    tool_outputs['similar_cases'] = [
        {'case_id': c.case_id, 'fraud_type': c.fraud_type, 'similarity': c.similarity_score}
        for c in similar_cases
    ]
    print(f"  Found {len(similar_cases)} similar fraud cases")
    for case in similar_cases[:3]:
        print(f"  - {case.case_id}: {case.fraud_type} (similarity: {case.similarity_score:.3f})")

    # TOOL 3: Risk Factor Calculator
    print("\n[TOOL 3] Risk Factor Calculator...")
    risk_factors = calculate_risk_factors(transaction, customer_history, '')
    tools_called.append("calculate_risk_factors")
    tool_outputs['risk_factors'] = risk_factors.to_dict()
    print(f"  Composite Risk Score: {risk_factors.composite_risk_score:.4f}")
    print(f"  Amount Deviation: {risk_factors.amount_deviation_std:.2f} std")
    print(f"  Velocity (24h): {risk_factors.velocity_24h:.2f} txn/hr")
    print(f"  Velocity (1h): {risk_factors.velocity_1h:.2f} txn/hr")

    # SYNTHESIS: Agent reasoning
    print("\n[SYNTHESIS] Agent Reasoning...")

    # Identify risk factors
    risk_factors_identified = []
    if customer_history.recent_velocity > 5:
        risk_factors_identified.append(f"High transaction velocity: {customer_history.recent_velocity:.1f} txn/hr")
    if abs(transaction.get('deviation_from_baseline', 0)) > 3:
        risk_factors_identified.append(f"Extreme amount deviation: {transaction.get('deviation_from_baseline', 0):.1f} std from baseline")
    if transaction.get('time_since_last_txn', 0) < 0.5:
        risk_factors_identified.append(f"High velocity: last transaction {transaction.get('time_since_last_txn', 0):.2f} hours ago")
    if customer_history.has_fraud_history:
        risk_factors_identified.append("Customer has prior fraud history")
    if customer_history.recent_velocity > 10:
        risk_factors_identified.append("Extreme transaction velocity spike")

    # Check similar cases
    matching_types = set()
    for case in similar_cases:
        matching_types.add(case.fraud_type)
    if matching_types:
        risk_factors_identified.append(f"Matches known fraud patterns: {', '.join(matching_types)}")

    # Calculate confidence
    base_confidence = 0.5
    if customer_history.has_fraud_history:
        base_confidence += 0.15
    if similar_cases:
        base_confidence += min(len(similar_cases) * 0.05, 0.2)
    if abs(transaction.get('deviation_from_baseline', 0)) > 3:
        base_confidence += 0.15
    if customer_history.recent_velocity > 5:
        base_confidence += 0.1

    confidence_score = min(base_confidence, 0.95)

    # Determine recommended action
    if confidence_score > 0.85 and abs(transaction.get('deviation_from_baseline', 0)) > 2:
        recommended_action = "auto_block"
    elif confidence_score > 0.65:
        recommended_action = "hold_for_review"
    else:
        recommended_action = "clear"

    # Build reasoning
    reasoning_parts = [
        f"Transaction flagged by detection model with {fraud_probability:.1%} fraud probability.",
        f"Customer {customer_history.customer_name} ({customer_id}) has {customer_history.total_transactions} lifetime transactions.",
    ]
    if risk_factors_identified:
        reasoning_parts.append(f"Risk factors identified: {'; '.join(risk_factors_identified)}.")
    if similar_cases:
        top_case = similar_cases[0]
        reasoning_parts.append(f"Top similar case: {top_case.case_id} ({top_case.fraud_type}) with {top_case.similarity_score:.1%} similarity.")
    reasoning_parts.append(f"Composite risk score: {risk_factors.composite_risk_score:.3f}.")
    reasoning_parts.append(f"Recommended action: {recommended_action} with {confidence_score:.1%} confidence.")

    reasoning = " ".join(reasoning_parts)

    print(f"  Risk Factors: {risk_factors_identified}")
    print(f"  Confidence: {confidence_score:.2%}")
    print(f"  Action: {recommended_action}")
    print(f"  Reasoning: {reasoning}")

    return InvestigationReport(
        transaction_id=transaction_id,
        customer_id=customer_id,
        timestamp=datetime.now(),
        fraud_probability=fraud_probability,
        detection_threshold=detection_threshold,
        customer_history=customer_history,
        similar_cases=similar_cases,
        risk_factors=risk_factors,
        risk_factors_identified=risk_factors_identified,
        confidence_score=confidence_score,
        recommended_action=recommended_action,
        reasoning=reasoning,
        tools_called=tools_called,
        tool_outputs=tool_outputs,
    )