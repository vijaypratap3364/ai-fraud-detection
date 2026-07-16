"""
Streamlit Dashboard for AI Fraud Detection System

Real-time dashboard showing:
- Live transaction feed
- Agent reasoning traces for flagged transactions
- Live evaluation scorecard
- Transaction analytics
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
API_BASE = "http://localhost:8000"
DATA_SYNTHETIC = Path(__file__).parent.parent.parent / 'data' / 'synthetic'

# Page config
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE = "http://localhost:8000"

# Session state initialization
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'investigations' not in st.session_state:
    st.session_state.investigations = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'total': 0, 'flagged': 0, 'tp': 0, 'fp': 0, 'fn': 0,
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'fpr': 0.0
    }
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# API Functions
@st.cache_data(ttl=5)
def check_api_health():
    try:
        response = requests.get(f"{API_BASE}/health", timeout=3)
        return response.status_code == 200, response.json()
    except:
        return False, None

def fetch_live_metrics():
    try:
        response = requests.get(f"{API_BASE}/metrics/live", timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def investigate_transaction(transaction, fraud_prob, threshold=0.7):
    try:
        response = requests.post(
            f"http://localhost:8000/investigate",
            json={
                "transaction": transaction,
                "fraud_probability": fraud_prob,
                "detection_threshold": 0.7
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# Sidebar
with st.sidebar:
    st.markdown("## 🛡️ AI Fraud Detection")
    st.markdown("---")

    # API Status
    health_ok, health_data = check_api_health()
    if health_ok:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Disconnected")
        st.caption("Start API: `python src/api/main.py`")

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

    auto_refresh = st.checkbox("Auto Refresh", value=True)

    if st.button("🔄 Clear Data", type="secondary"):
        st.session_state.transactions = []
        st.session_state.investigations = {}
        st.session_state.metrics = {
            'total': 0, 'flagged': 0, 'tp': 0, 'fp': 0, 'fn': 0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'fpr': 0.0
        }
        st.rerun()

    st.markdown("---")

    # Quick Stats
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Processed", st.session_state.metrics['total'])
        st.metric("Flagged", st.session_state.metrics['flagged'])
    with col2:
        st.metric("Precision", f"{st.session_state.metrics['precision']:.1%}")
        st.metric("Recall", f"{st.session_state.metrics['recall']:.1%}")

# Main Dashboard
st.markdown('<h1 class="main-header">🛡️ AI Fraud Detection System</h1>', unsafe_allow_html=True)
st.caption("Real-time fraud detection with agentic investigation • XGBoost + Agent RAG • Simulated live stream")

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

        # Transaction feed container
        feed_container = st.container()

        # Display transactions (newest first)
        if st.session_state.transactions:
            for txn in reversed(st.session_state.transactions[-50:]):  # Show last 50
                is_fraud = txn.get('is_fraud', False)
                investigated = txn.get('investigated', False)

                # Determine styling
                if investigated:
                    action = txn.get('investigation', {}).get('action', 'unknown')
                    confidence = txn.get('investigation', {}).get('confidence', 0)

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

                # Transaction row
                with st.container():
                    cols = st.columns([1, 2, 1, 1, 1, 2])
                    with cols[0]:
                        st.caption(f"{icon} {txn.get('transaction_id', 'N/A')}")
                    with cols[1]:
                        st.caption(f"${txn.get('amount', 0):.2f} | {txn.get('customer_id', 'N/A')}")
                    with cols[2]:
                        prob = txn.get('fraud_prob', 0)
                        color = "risk-high" if prob > 0.7 else "risk-medium" if prob > 0.4 else "risk-low"
                        st.markdown(f"<span class='{color}'>{prob:.1%}</span>", unsafe_allow_html=True)
                    with cols[3]:
                        if investigated:
                            st.markdown(f"<span class='{color}'>{badge}</span>", unsafe_allow_html=True)
                        else:
                            st.caption("Pending")
                    with cols[4]:
                        dev = txn.get('deviation', 0)
                        st.caption(f"Dev: {dev:.1f}σ")
                    with cols[5]:
                        if investigated and st.button("🔍", key=f"view_{txn.get('transaction_id')}", help="View investigation"):
                            st.session_state.selected_investigation = txn.get('transaction_id')
                            st.rerun()

                    if alert_class:
                        st.markdown('<div class="{}"></div>'.format(alert_class), unsafe_allow_html=True)
        else:
            st.info("No transactions yet. Start the stream simulator: `python src/simulation/stream.py`")

    with col_right:
        st.markdown("### 📊 Live Scorecard")

        # Metric cards
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Precision</h4>
                <h2>{st.session_state.metrics['precision']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Recall</h4>
                <h2>{st.session_state.metrics['recall']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)

        mcol3, mcol4 = st.columns(2)
        with mcol3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>F1 Score</h4>
                <h2>{st.session_state.metrics['f1']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with mcol4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>FPR</h4>
                <h2>{st.session_state.metrics['fpr']:.3%}</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption(f"TP: {st.session_state.metrics['tp']} | FP: {st.session_state.metrics['fp']} | FN: {st.session_state.metrics['fn']}")
        st.caption(f"Total: {st.session_state.metrics['total']} | Flagged: {st.session_state.metrics['flagged']}")

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

        # Time series of fraud probability
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fraud Probability Over Time', 'Amount Distribution',
                          'Fraud Rate by Hour', 'Investigation Actions'),
            specs=[[{"secondary_y": True}, {}], [{}, {}]]
        )

        # Fraud prob over time
        fig.add_trace(
            go.Scatter(x=list(range(len(df))), y=df['fraud_prob'],
                      mode='lines', name='Fraud Prob', line=dict(color='#1f77b4')),
            row=1, col=1
        )

        # Amount distribution
        fig.add_trace(
            go.Histogram(x=df['amount'], nbinsx=30, name='Amount', marker_color='#ff7f0e'),
            row=1, col=2
        )

        # Fraud rate by hour (if we have timestamps)
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
        st.info("Need more transactions for analytics. Run the stream simulator.")

# ============================================================
# TAB 4: SYSTEM
# ============================================================
with tab4:
    st.markdown("### ⚙️ System Status & Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏗️ Architecture")
        st.markdown("""
        **Detection Layer (Phase 2)**
        - XGBoost with SMOTE for class imbalance
        - PR-AUC: 0.8588 | Precision: 92.9% | Recall: 79.6%
        - Threshold: 0.8878 (optimized for F1)

        **Agent Investigation (Phase 4)**
        - Tool 1: Customer History Lookup
        - Tool 2: Fraud Similarity Search (RAG)
        - Tool 3: Risk Factor Calculator
        - Orchestration with full traceability

        **RAG System (Phase 4)**
        - Chroma vector DB (200 fraud cases)
        - sentence-transformers/all-MiniLM-L6-v2
        - 6 fraud types, semantic search
        """)

    with col2:
        st.markdown("#### 🚀 Running the System")
        st.code("""
# Terminal 1: Start API
python src/api/main.py

# Terminal 2: Start Stream Simulator
python src/simulation/stream.py --rate 5 --speed 1 --loop

# Terminal 3: Start Dashboard
streamlit run frontend/app.py
        """)

        st.markdown("#### 🔗 Endpoints")
        st.markdown("""
        - **API**: http://localhost:8000
        - **Dashboard**: http://localhost:8501
        - **Health**: GET /health
        - **Investigate**: POST /investigate
        - **Customer History**: GET /customer/{id}/history
        - **Similar Cases**: POST /similar-cases
        """)

    st.markdown("---")
    st.markdown("#### 📁 Project Structure")
    st.code("""
ai-fraud-detection/
├── data/
│   ├── raw/creditcard.csv
│   ├── synthetic/
│   │   ├── customers.csv
│   │   ├── transaction_history.csv
│   │   ├── customers_enriched.csv
│   │   └── known_fraud_cases.json
├── models/detection/xgboost_model.pkl
├── chroma_db/ (vector DB)
├── src/
│   ├── detection/train.py
│   ├── agent/tools/investigation_tools.py
│   ├── api/main.py
│   ├── simulation/stream.py
│   └── utils/*.py
├── frontend/app.py (this file)
├── chroma_db/
└── requirements.txt
    """)

# Auto-refresh logic
if auto_refresh:
    time.sleep(2)
    st.rerun()

# Footer
st.markdown("---")
st.caption("AI Fraud Detection System • Built with XGBoost + Agent RAG + Streamlit • [GitHub](https://github.com/vijaypratap3364/ai-fraud-detection)")