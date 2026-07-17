"""
AI Fraud Detection System - Single App Deployment
Streamlit app with embedded agentic investigation (no separate API needed)
Deploys free on Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import investigation tools directly (no API calls needed)
try:
    from src.agent.tools.investigation_tools import (
        investigate_transaction,
        lookup_customer_history,
        search_similar_fraud_cases,
        calculate_risk_factors,
        _initialize as init_tools,
    )
    TOOLS_AVAILABLE = True
except ImportError as e:
    TOOLS_AVAILABLE = False
    IMPORT_ERROR = str(e)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .flagged-alert {
        background: #ffe6e6;
        border-left: 4px solid #ff4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .investigation-box {
        background: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .tool-call {
        background: #e8f4fd;
        border-left: 3px solid #1f77b4;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
    }
    .risk-high { color: #ff4444; font-weight: bold; }
    .risk-medium { color: #ff8800; font-weight: bold; }
    .risk-low { color: #00aa00; font-weight: bold; }
    .stButton>button { width: 100%; }
    .section-header { margin-top: 1.5rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INITIALIZATION
# ============================================================
@st.cache_resource
def initialize_tools():
    """Initialize investigation tools once (cached)."""
    if not TOOLS_AVAILABLE:
        raise RuntimeError(f"Investigation tools unavailable: {IMPORT_ERROR}")
    init_tools()
    return True

# Initialize on first load
if 'tools_initialized' not in st.session_state:
    with st.spinner("Loading investigation tools..."):
        try:
            initialize_tools()
            st.session_state.tools_initialized = True
            st.session_state.tools_error = None
        except Exception as e:
            st.session_state.tools_initialized = False
            st.session_state.tools_error = str(e)

# Session state defaults
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'investigations' not in st.session_state:
    st.session_state.investigations = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'total': 0, 'flagged': 0, 'tp': 0, 'fp': 0, 'fn': 0,
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'fpr': 0.0
    }
if 'simulator_running' not in st.session_state:
    st.session_state.simulator_running = False
if 'last_sim_txn' not in st.session_state:
    st.session_state.last_sim_txn = 0

# ============================================================
# DATA LOADING (for simulator)
# ============================================================
@st.cache_data
def load_transaction_data():
    """Load synthetic transaction data for simulation."""
    try:
        # FIX: download_data.py downloads to data/synthetic/transaction_history_demo.csv
        # The old code pointed at a nonexistent 'hf_data_staging' folder and a
        # fallback filename missing the '_demo' suffix, so this always failed
        # and the simulator had nothing to stream.
        data_path = PROJECT_ROOT / 'data' / 'synthetic' / 'transaction_history_demo.csv'
        if not data_path.exists():
            # fallback to non-demo filename if it's ever used instead
            data_path = PROJECT_ROOT / 'data' / 'synthetic' / 'transaction_history.csv'

        df = pd.read_csv(data_path)
        # Add fraud probability if not present (simulate model scores)
        if 'fraud_probability' not in df.columns:
            # Simple heuristic: higher deviation = higher fraud prob
            df['fraud_probability'] = np.clip(
                np.abs(df.get('deviation_from_baseline', 0)) * 0.15 +
                (df['Class'] * 0.7) +
                np.random.random(len(df)) * 0.1,
                0, 1
            )
        return df
    except Exception as e:
        st.error(f"Failed to load transaction data: {e}")
        return pd.DataFrame()

# ============================================================
# SIMULATOR FUNCTIONS
# ============================================================
def get_next_transaction(df, index):
    """Get next transaction from dataset."""
    if index >= len(df):
        return None
    row = df.iloc[index]

    # FIX: the old default expression `int(row.get('customer_id', 0))` was
    # evaluated eagerly even when customer_id already existed as a string
    # (e.g. "CUST_000001"), which raised ValueError on int(). Now we only
    # build the fallback id when customer_id is actually missing.
    raw_customer_id = row.get('customer_id', None)
    if raw_customer_id is None or (isinstance(raw_customer_id, float) and pd.isna(raw_customer_id)):
        customer_id = f"CUST_{index:06d}"
    else:
        customer_id = raw_customer_id

    return {
        'transaction_id': f"TXN_{int(row.get('transaction_number', index)):08d}",
        'customer_id': customer_id,
        'amount': float(row['Amount']),
        'timestamp': float(row['Time']),
        'fraud_probability': float(row.get('fraud_probability', 0)),
        'deviation_from_baseline': float(row.get('deviation_from_baseline', 0)),
        'time_since_last_txn': float(row.get('time_since_last_txn', 0)),
        'transaction_number': int(row.get('transaction_number', index)),
        'is_fraud': bool(row['Class']),
    }

def run_investigation(transaction, threshold=0.7):
    """Run agentic investigation on a transaction."""
    try:
        report = investigate_transaction(
            transaction=transaction,
            fraud_probability=transaction['fraud_probability'],
            detection_threshold=threshold
        )
        return report.to_dict()
    except Exception as e:
        st.error(f"Investigation failed: {e}")
        return None

def update_metrics(transaction, investigation=None):
    """Update running metrics."""
    m = st.session_state.metrics
    m['total'] += 1

    is_fraud = transaction['is_fraud']
    fraud_prob = transaction['fraud_probability']
    predicted_fraud = fraud_prob >= 0.7

    if predicted_fraud:
        m['flagged'] += 1

    if predicted_fraud and is_fraud:
        m['tp'] += 1
    elif predicted_fraud and not is_fraud:
        m['fp'] += 1
    elif not predicted_fraud and is_fraud:
        m['fn'] += 1

    # Recalculate
    if m['tp'] + m['fp'] > 0:
        m['precision'] = m['tp'] / (m['tp'] + m['fp'])
    if m['tp'] + m['fn'] > 0:
        m['recall'] = m['tp'] / (m['tp'] + m['fn'])
    if m['precision'] + m['recall'] > 0:
        m['f1'] = 2 * m['precision'] * m['recall'] / (m['precision'] + m['recall'])
    if m['fp'] + (m['total'] - m['tp'] - m['fp'] - m['fn']) > 0:
        m['fpr'] = m['fp'] / (m['fp'] + (m['total'] - m['tp'] - m['fp'] - m['fn']))

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🛡️ AI Fraud Detection")
    st.markdown("---")

    # System Status
    st.markdown("### 📊 System Status")
    st.success("🟢 Investigation Engine: Ready")
    st.caption("Tools: Customer History | RAG Similarity | Risk Calculator")

    st.markdown("---")

    # Controls
    st.markdown("### ⚙️ Controls")

    detection_threshold = st.slider(
        "Detection Threshold",
        min_value=0.5, max_value=0.95, value=0.7, step=0.05,
        help="Fraud probability threshold to trigger investigation"
    )

    stream_rate = st.slider(
        "Stream Rate (txns/sec)",
        min_value=1, max_value=20, value=5,
        help="Simulated transaction rate"
    )

    auto_investigate = st.checkbox("Auto-investigate flagged", value=True)

    st.markdown("---")

    # Simulator Controls
    st.markdown("### 🎮 Simulator")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", disabled=st.session_state.simulator_running, type="primary"):
            st.session_state.simulator_running = True
            st.rerun()
    with col2:
        if st.button("⏹️ Stop", disabled=not st.session_state.simulator_running):
            st.session_state.simulator_running = False
            st.rerun()

    if st.button("🔄 Clear All Data"):
        st.session_state.transactions = []
        st.session_state.investigations = {}
        st.session_state.metrics = {
            'total': 0, 'flagged': 0, 'tp': 0, 'fp': 0, 'fn': 0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'fpr': 0.0
        }
        st.session_state.last_sim_txn = 0
        st.rerun()

    st.markdown("---")

    # Quick Stats
    st.markdown("### 📈 Quick Stats")
    m = st.session_state.metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Processed", m['total'])
        st.metric("Flagged", m['flagged'])
    with col2:
        st.metric("Precision", f"{m['precision']:.1%}")
        st.metric("Recall", f"{m['recall']:.1%}")

    st.caption(f"TP: {m['tp']} | FP: {m['fp']} | FN: {m['fn']}")

# ============================================================
# MAIN DASHBOARD
# ============================================================
st.markdown('<h1 class="main-header">🛡️ AI Fraud Detection System</h1>', unsafe_allow_html=True)
st.caption("Real-time fraud detection with agentic investigation • XGBoost + Agent RAG • Streamlit Cloud Deployment")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Live Feed",
    "🔍 Investigations",
    "📈 Analytics",
    "⚙️ System"
])

# ============================================================
# TAB 1: LIVE FEED
# ============================================================
with tab1:
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.markdown("### 🔴 Live Transaction Feed")

        # Run simulator if active
        if st.session_state.simulator_running:
            df = load_transaction_data()
            if not df.empty and st.session_state.last_sim_txn < len(df):
                # Process batch based on rate
                batch_size = max(1, stream_rate // 2)
                for _ in range(batch_size):
                    if st.session_state.last_sim_txn >= len(df):
                        st.session_state.simulator_running = False
                        break

                    txn = get_next_transaction(df, st.session_state.last_sim_txn)
                    st.session_state.last_sim_txn += 1

                    if txn:
                        # Add to feed
                        txn['investigated'] = False
                        st.session_state.transactions.append(txn)

                        # Auto-investigate if flagged
                        if auto_investigate and txn['fraud_probability'] >= detection_threshold:
                            investigation = run_investigation(txn, detection_threshold)
                            if investigation:
                                txn['investigated'] = True
                                txn['investigation'] = investigation
                                st.session_state.investigations[txn['transaction_id']] = investigation

                        update_metrics(txn, txn.get('investigation'))

                st.rerun()
            elif df.empty:
                st.session_state.simulator_running = False
                st.error("No transaction data available — check that data/synthetic/transaction_history_demo.csv was downloaded.")

        # Display transactions (newest first)
        feed_container = st.container()
        with feed_container:
            if st.session_state.transactions:
                for txn in reversed(st.session_state.transactions[-50:]):
                    is_fraud = txn.get('is_fraud', False)
                    investigated = txn.get('investigated', False)

                    if investigated:
                        inv = txn.get('investigation', {})
                        action = inv.get('recommended_action', 'unknown')
                        confidence = inv.get('confidence_score', 0)

                        if action == 'auto_block':
                            alert_class = "flagged-alert"
                            icon = "🔴"
                            badge = f"BLOCKED ({confidence:.0%})"
                        elif action == 'hold_for_review':
                            alert_class = "flagged-alert"
                            icon = "🟡"
                            badge = f"REVIEW ({confidence:.0%})"
                        else:
                            alert_class = ""
                            icon = "🟢"
                            badge = f"CLEARED ({confidence:.0%})"
                    elif is_fraud:
                        alert_class = "flagged-alert"
                        icon = "⚠️"
                        badge = "FRAUD (Missed)"
                    else:
                        alert_class = ""
                        icon = "⚪"
                        badge = "Normal"

                    with st.container():
                        cols = st.columns([1, 2, 1, 1, 1, 2])
                        with cols[0]:
                            st.caption(f"{icon} {txn.get('transaction_id', 'N/A')}")
                        with cols[1]:
                            st.caption(f"${txn.get('amount', 0):.2f} | {txn.get('customer_id', 'N/A')}")
                        with cols[2]:
                            # FIX: key was 'fraud_prob' but transactions store 'fraud_probability'
                            prob = txn.get('fraud_probability', 0)
                            color = "risk-high" if prob > 0.7 else "risk-medium" if prob > 0.4 else "risk-low"
                            st.markdown(f"<span class='{color}'>{prob:.1%}</span>", unsafe_allow_html=True)
                        with cols[3]:
                            if investigated:
                                st.markdown(f"<span class='{color}'>{badge}</span>", unsafe_allow_html=True)
                            else:
                                st.caption("Pending")
                        with cols[4]:
                            # FIX: key was 'deviation' but transactions store 'deviation_from_baseline'
                            dev = txn.get('deviation_from_baseline', 0)
                            st.caption(f"Dev: {dev:.1f}σ")
                        with cols[5]:
                            if investigated and st.button("🔍", key=f"view_{txn.get('transaction_id')}", help="View investigation"):
                                st.session_state.selected_investigation = txn.get('transaction_id')
                                st.rerun()

                        if alert_class:
                            st.markdown(f'<div class="{alert_class}"></div>', unsafe_allow_html=True)
            else:
                st.info("No transactions yet. Click ▶️ Start in sidebar to begin simulation.")

    with col_right:
        st.markdown("### 📊 Live Scorecard")

        m = st.session_state.metrics
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Precision</h4>
                <h2>{m['precision']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Recall</h4>
                <h2>{m['recall']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)

        mcol3, mcol4 = st.columns(2)
        with mcol3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>F1 Score</h4>
                <h2>{m['f1']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with mcol4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>FPR</h4>
                <h2>{m['fpr']:.3%}</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption(f"TP: {m['tp']} | FP: {m['fp']} | FN: {m['fn']}")
        st.caption(f"Total: {m['total']} | Flagged: {m['flagged']}")

# ============================================================
# TAB 2: INVESTIGATIONS
# ============================================================
with tab2:
    st.markdown("### 🔍 Investigation Details")

    if 'selected_investigation' in st.session_state:
        txn_id = st.session_state.selected_investigation
        inv = st.session_state.investigations.get(txn_id, {})

        if inv:
            st.markdown(f"#### Investigation: {txn_id}")

            col1, col2 = st.columns([2, 1])

            with col1:
                # Agent Reasoning
                st.markdown("#### 🧠 Agent Reasoning")
                st.markdown(f"""
                <div class="investigation-box">
                {inv.get('reasoning', 'No reasoning available')}
                </div>
                """, unsafe_allow_html=True)

                # Tools Called
                st.markdown("#### 🔧 Tools Called")
                for tool in inv.get('tools_called', []):
                    st.markdown(f"""
                    <div class="tool-call">
                        <strong>{tool}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                # Risk Factors
                st.markdown("#### ⚠️ Risk Factors Identified")
                for rf in inv.get('risk_factors_identified', []):
                    st.markdown(f"- {rf}")

                if not inv.get('risk_factors_identified'):
                    st.caption("No significant risk factors identified")

            with col2:
                # Confidence & Action
                st.metric("Confidence", f"{inv.get('confidence_score', 0):.1%}")

                action = inv.get('recommended_action', 'clear')
                if action == 'auto_block':
                    st.error(f"🔴 **{action.upper()}**")
                elif action == 'hold_for_review':
                    st.warning(f"🟡 **{action.upper()}**")
                else:
                    st.success(f"🟢 **{action.upper()}**")

                st.markdown("---")

                # Similar Cases
                st.markdown("#### 🔗 Similar Fraud Cases")
                for case in inv.get('similar_cases', [])[:3]:
                    with st.expander(f"{case['case_id']} - {case['fraud_type']} ({case['similarity_score']:.1%})"):
                        st.caption(f"Amount: ${case['amount']:.2f}")
                        st.caption(f"Risk Factors: {', '.join(case['risk_factors'])}")
                        st.caption(f"Notes: {case['investigation_notes'][:100]}...")

                # Risk Factors Detail
                st.markdown("#### 📊 Risk Factors Detail")
                rf = inv.get('risk_factors', {})
                if rf:
                    for key, value in rf.items():
                        st.caption(f"{key}: {value}")
        else:
            st.warning("Investigation data not found")
    else:
        st.info("Click 🔍 on a flagged transaction in the Live Feed to view investigation details")

# ============================================================
# TAB 3: ANALYTICS
# ============================================================
with tab3:
    st.markdown("### 📈 Transaction Analytics")

    if len(st.session_state.transactions) > 10:
        df = pd.DataFrame(st.session_state.transactions)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fraud Probability Over Time', 'Amount Distribution',
                          'Fraud Rate by Hour', 'Investigation Actions'),
            specs=[[{"secondary_y": True}, {}], [{}, {}]]
        )

        # FIX: DataFrame column is 'fraud_probability', not 'fraud_prob' —
        # the old key name would raise KeyError here as soon as >10 txns existed.
        fig.add_trace(
            go.Scatter(x=list(range(len(df))), y=df['fraud_probability'],
                      mode='lines', name='Fraud Prob', line=dict(color='#1f77b4')),
            row=1, col=1
        )

        # Amount distribution
        fig.add_trace(
            go.Histogram(x=df['amount'], nbinsx=30, name='Amount', marker_color='#ff7f0e'),
            row=1, col=2
        )

        # Fraud rate by hour
        if 'timestamp' in df.columns:
            df['hour'] = (df['timestamp'] / 3600) % 24
            hourly_fraud = df.groupby('hour')['is_fraud'].mean().reset_index()
            fig.add_trace(
                go.Bar(x=hourly_fraud['hour'], y=hourly_fraud['is_fraud'], name='Fraud Rate', marker_color='#d62728'),
                row=2, col=1
            )

        # Investigation actions
        if 'investigation' in df.columns:
            actions = [txn.get('investigation', {}).get('action', 'none') for txn in st.session_state.transactions if 'investigation' in txn]
            if actions:
                action_counts = pd.Series(actions).value_counts()
                fig.add_trace(
                    go.Pie(labels=action_counts.index, values=action_counts.values, name="Actions"),
                    row=2, col=2
                )

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need more transactions for analytics. Start the simulator in the Live Feed tab.")

# ============================================================
# TAB 4: SYSTEM
# ============================================================
with tab4:
    st.markdown("### ⚙️ System Status & Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏗️ Architecture")
        st.markdown("""
        **Detection Layer**
        - XGBoost with SMOTE for class imbalance (1:577)
        - PR-AUC: 0.8588 | Precision: 92.9% | Recall: 79.6%
        - Threshold: 0.8878 (optimized for F1)

        **Agent Investigation Layer**
        - Tool 1: Customer History Lookup
        - Tool 2: Fraud Similarity Search (RAG)
        - Tool 3: Risk Factor Calculator
        - Full traceability (tools called, outputs, reasoning)

        **RAG System**
        - Chroma vector DB (200 fraud cases)
        - sentence-transformers/all-MiniLM-L6-v2
        - 6 fraud types: card_theft, account_takeover, synthetic_identity, friendly_fraud, card_not_present, skimming
        """)

    with col2:
        st.markdown("#### 🚀 Deployment Info")
        st.markdown("""
        **Single-App Streamlit Deployment**
        - No separate API service needed
        - Investigation tools run in-process
        - Deploys free on Streamlit Community Cloud
        - Auto-deploys from GitHub on push

        **Running Locally**
        ```bash
        pip install -r streamlit_app/requirements.txt
        streamlit run streamlit_app/app.py
        ```

        **Project Structure**
        ```
        ai-fraud-detection/
        ├── streamlit_app/
        │   ├── app.py          # This file
        │   └── requirements.txt
        ├── src/
        │   ├── detection/train.py
        │   ├── agent/tools/investigation_tools.py
        │   └── simulation/stream.py
        ├── data/synthetic/
        ├── models/detection/
        └── chroma_db/
        ```
        """)

    st.markdown("---")
    st.markdown("#### 📋 API Reference (Embedded Functions)")
    st.code("""
# Direct function calls (no HTTP needed)
from src.agent.tools.investigation_tools import (
    investigate_transaction,
    lookup_customer_history,
    search_similar_fraud_cases,
    calculate_risk_factors,
)

# Full investigation
report = investigate_transaction(
    transaction=txn_dict,
    fraud_probability=0.85,
    detection_threshold=0.7
)

# Individual tools
history = lookup_customer_history("CUST_000001")
similar = search_similar_fraud_cases("high amount deviation velocity", top_k=5)
factors = calculate_risk_factors(transaction_dict)
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("AI Fraud Detection System • XGBoost + Agent RAG + Streamlit • [GitHub](https://github.com/vijaypratap3364/ai-fraud-detection) • Deployed on Streamlit Community Cloud")
