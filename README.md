# AI-Powered Financial Fraud Detection System

**An end-to-end fraud detection system with agentic investigation, live simulation, and visible reasoning.**

---

## What Makes This Different

This isn't just another fraud detection notebook. **95% of fraud detection projects online stop at the classifier** — train a model, report an accuracy number, done. This project builds the layers that actually matter:

### 🎯 The Differentiators (What Makes It Resume-Worthy)

1. **Agentic Investigation Layer** ⭐ **The core differentiator**
   - When a transaction is flagged, an AI agent investigates with real tool use
   - Fetches customer transaction history to identify pattern breaks
   - Performs similarity search over past fraud cases (RAG with embeddings)
   - Calculates concrete risk factors (not LLM guesses)
   - Produces structured, traceable reports with confidence scores

2. **Live Simulation with Visible Reasoning**
   - Replays Kaggle data as a simulated real-time transaction stream
   - Dashboard shows the agent thinking, not just final labels
   - Watch tool calls, retrieved context, and reasoning traces live

3. **Proper Evaluation Harness**
   - Precision/Recall/PR-AUC metrics (not naive accuracy)
   - Explicitly documents why accuracy is meaningless for fraud (0.1-0.5% base rate)
   - Live eval scorecard updating as transactions stream

4. **Actual Deployment**
   - Hosted publicly with a live demo link
   - Not just a notebook — a running system

---

## Architecture Overview

### Layer 1: Detection (Classical ML - The Plumbing)
- **XGBoost/LightGBM** classifier scoring each transaction
- Handles severe class imbalance with SMOTE/class weighting
- This is necessary but not what makes the project special

### Layer 2: Investigation Agent (The Differentiator) ⭐
When a transaction crosses the risk threshold, the agent investigates:

**Tool 1: Customer History Lookup**
- Fetches recent transactions from synthetic customer profiles
- Identifies pattern breaks: location changes, spend spikes, new merchant categories

**Tool 2: Fraud Case Similarity Search (RAG)**
- Embeds historical confirmed-fraud cases into a vector database (Chroma)
- Retrieves most similar past fraud patterns
- Enables pattern-based reasoning tied to real cases

**Tool 3: Risk Factor Calculator**
- Computes concrete metrics: transaction velocity, baseline deviation, geographic distance
- Returns actual numbers, not LLM guesses

**Synthesis: Structured Report**
- Risk factors identified
- Confidence score
- Recommended action: auto-block / hold for review / clear
- **Traceable** — every claim links back to tool outputs

### Layer 3: Real-Time Simulation
- Replays static Kaggle dataset as if transactions arrive live (one every few seconds)
- Standard practice for testing fraud systems before production
- **Honest framing**: "simulated real-time using historical data" (not claiming live production data)

### Layer 4: Live Dashboard
- **Transaction feed**: Incoming transactions scrolling in real-time
- **Agent reasoning trace**: Click a flagged transaction to see which tools ran and what they found
- **Live eval scorecard**: Precision/Recall/FPR updating as the stream runs
- Built with Streamlit or React

---

## How It Works (End-to-End Flow)

1. **Simulator** pushes a transaction from Kaggle dataset to API endpoint
2. **Detection model** (XGBoost) scores it → fraud probability 0.87
3. **If threshold crossed**, agent kicks in:
   - Fetches customer's synthetic transaction history
   - Runs similarity search over embedded fraud cases
   - Calculates risk factors (velocity, baseline deviation, geo jump)
   - Synthesizes structured report
4. **Dashboard updates** with flagged transaction and agent's reasoning trace
5. **Eval scorecard updates** (we secretly know true label from Kaggle data)

---

## Project Structure

```
ai-fraud-detection/
├── data/
│   ├── raw/                 # Kaggle credit card fraud dataset
│   ├── processed/           # Cleaned/preprocessed transactions
│   └── synthetic/           # Generated customer profiles & history
├── models/
│   ├── detection/           # Trained XGBoost/LightGBM models
│   └── embeddings/          # Vector embeddings for similarity search
├── src/
│   ├── detection/           # Detection layer (XGBoost training/inference)
│   ├── agent/
│   │   ├── tools/           # Agent tools (history, similarity, risk)
│   │   └── orchestration/   # Agent orchestration logic
│   ├── simulation/          # Transaction stream simulator
│   ├── api/                 # FastAPI backend
│   └── utils/               # Shared utilities
├── frontend/                # Dashboard (Streamlit/React)
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── .gitignore
├── README.md
├── requirements.txt
└── config.yaml
```

---

## Tech Stack

**Detection Layer:**
- XGBoost / LightGBM
- scikit-learn (SMOTE, metrics)
- pandas, numpy

**Agent Layer:**
- LangChain or custom orchestration
- Chroma (vector database for similarity search)
- OpenAI embeddings or sentence-transformers

**Backend:**
- FastAPI
- Pydantic for structured outputs

**Frontend:**
- Streamlit (fast prototyping) or React (polished)

**Deployment:**
- Railway / Render / Fly.io

---

## Setup Instructions

### 1. Clone and Install
```bash
git clone https://github.com/yourusername/ai-fraud-detection.git
cd ai-fraud-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Kaggle Data
- Get the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Place `creditcard.csv` in `data/raw/`

### 3. Run Setup Scripts
```bash
# Preprocess data and generate synthetic customer profiles
python src/detection/preprocess.py

# Train detection model
python src/detection/train.py

# Build vector database for fraud case similarity
python src/agent/tools/build_similarity_db.py
```

### 4. Start the System
```bash
# Terminal 1: Start FastAPI backend
python src/api/main.py

# Terminal 2: Start transaction simulator
python src/simulation/stream.py

# Terminal 3: Start dashboard
streamlit run frontend/app.py
```

### 5. Open Dashboard
Visit `http://localhost:8501` to watch live fraud detection with agent reasoning.

---

## Development Roadmap

**Phase 1: Foundation** ✅
- [x] Project structure
- [x] Git repository initialized
- [ ] Download & explore Kaggle data

**Phase 2: Detection Layer** (In Progress)
- [ ] Train XGBoost with proper imbalance handling
- [ ] Evaluate with Precision/Recall/PR-AUC
- [ ] Document why accuracy is meaningless

**Phase 3: Synthetic Customer Layer**
- [ ] Generate customer profiles
- [ ] Build transaction history database

**Phase 4: Agent Investigation** ⭐ **The Differentiator**
- [ ] Build customer history lookup tool
- [ ] Build fraud similarity search (RAG)
- [ ] Build risk factor calculator
- [ ] Orchestrate agent with tool calls

**Phase 5: Real-Time Simulation**
- [ ] Transaction stream simulator
- [ ] FastAPI backend endpoints

**Phase 6: Dashboard**
- [ ] Live transaction feed UI
- [ ] Agent reasoning trace viewer
- [ ] Live eval scorecard

**Phase 7: Deployment**
- [ ] Deploy to Railway/Render
- [ ] Public demo link

---

## Why "Simulated Real-Time"? (Honest Framing)

This project uses **historical Kaggle data replayed as a simulated live stream**. This is not a trick — it's standard practice in fraud detection engineering:

- **Reality**: Kaggle data is static; no one has access to actual live fraud data for a public project
- **Solution**: Replay the dataset as if transactions arrive in real-time (one every few seconds)
- **Why it's legitimate**: This is exactly how production fraud systems are tested before going live
- **Honest communication**: README and demos clearly state "simulated real-time using historical data"

Interviewers won't fault you for not having live production data. They'll respect the honesty and the fact that you built the system as it would operate in production.

---

## Contributing

This is a portfolio project, but feedback and suggestions are welcome! Open an issue or PR.

---

## License

MIT License - feel free to use this for learning or your own portfolio.

---

**Project started:** 2026-07-14  
**Status:** In active development