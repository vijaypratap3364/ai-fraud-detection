"""
Agent Orchestration Service

FastAPI service that coordinates the agent investigation layer
and provides REST API endpoints for the dashboard.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from src.agent.tools.investigation_tools import (
    investigate_transaction,
    lookup_customer_history,
    search_similar_fraud_cases,
    calculate_risk_factors,
    InvestigationReport,
    CustomerHistoryResult,
    SimilarFraudCase,
    RiskFactors,
)
from src.agent.tools.investigation_tools import _initialize as init_tools

# Initialize tools on startup
from src.agent.tools.investigation_tools import _initialize as init_tools
init_tools()

# Pydantic models for API
class TransactionInput(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    timestamp: float
    fraud_probability: float
    deviation_from_baseline: float = 0.0
    time_since_last_txn: float = 0.0
    transaction_number: int = 0

class InvestigationRequest(BaseModel):
    transaction: dict
    fraud_probability: float
    detection_threshold: float = 0.7

class InvestigationResponse(BaseModel):
    transaction_id: str
    customer_id: str
    timestamp: str
    fraud_probability: float
    detection_threshold: float
    customer_history: dict
    similar_cases: List[dict]
    risk_factors: dict
    risk_factors_identified: List[str]
    confidence_score: float
    recommended_action: str
    reasoning: str
    tools_called: List[str]

class CustomerHistoryResponse(BaseModel):
    customer_id: str
    customer_name: str
    total_transactions: int
    recent_velocity: float
    has_fraud_history: bool
    spending_pattern: str
    recent_transactions: List[dict]

class SimilarCaseResponse(BaseModel):
    case_id: str
    fraud_type: str
    amount: float
    distance: float
    similarity_score: float
    risk_factors: List[str]
    description: str
    investigation_notes: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: bool
    chroma_connected: bool


# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    from src.agent.tools.investigation_tools import _initialize
    _initialize()
    yield
    # Shutdown (if needed)
    pass


app = FastAPI(
    title="AI Fraud Detection API",
    description="Agent Investigation Layer API for Fraud Detection System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=True,
        chroma_connected=True,
    )


# Main investigation endpoint
@app.post("/investigate", response_model=InvestigationResponse)
async def investigate_endpoint(request: InvestigationRequest):
    """
    Main endpoint for agent investigation.
    Triggers full agent investigation when transaction crosses threshold.
    """
    try:
        from src.agent.tools.investigation_tools import investigate_transaction

        report = investigate_transaction(
            transaction=request.transaction,
            fraud_probability=request.fraud_probability,
            detection_threshold=request.detection_threshold,
        )

        return InvestigationResponse(
            transaction_id=report.transaction_id,
            customer_id=report.customer_id,
            timestamp=report.timestamp.isoformat(),
            fraud_probability=report.fraud_probability,
            detection_threshold=report.detection_threshold,
            customer_history={
                'customer_id': report.customer_history.customer_id,
                'customer_name': report.customer_history.customer_name,
                'total_transactions': report.customer_history.total_transactions,
                'recent_velocity': report.customer_history.recent_velocity,
                'has_fraud_history': report.customer_history.has_fraud_history,
                'spending_pattern': report.customer_history.spending_pattern,
            },
            similar_cases=[
                {
                    'case_id': c.case_id,
                    'fraud_type': c.fraud_type,
                    'amount': c.amount,
                    'distance': c.distance,
                    'similarity_score': c.similarity_score,
                    'risk_factors': c.risk_factors,
                    'description': c.description,
                    'investigation_notes': c.investigation_notes,
                }
                for c in report.similar_cases
            ],
            risk_factors=report.risk_factors.to_dict(),
            risk_factors_identified=report.risk_factors_identified,
            confidence_score=report.confidence_score,
            recommended_action=report.recommended_action,
            reasoning=report.reasoning,
            tools_called=report.tools_called,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Customer history endpoint
@app.get("/customer/{customer_id}/history", response_model=CustomerHistoryResponse)
async def get_customer_history(customer_id: str, lookback_days: int = 90, max_transactions: int = 50):
    """Get customer transaction history."""
    try:
        from src.agent.tools.investigation_tools import lookup_customer_history
        history = lookup_customer_history(customer_id, lookback_days, max_transactions)

        return CustomerHistoryResponse(
            customer_id=history.customer_id,
            customer_name=history.customer_name,
            total_transactions=history.total_transactions,
            recent_velocity=history.recent_velocity,
            has_fraud_history=history.has_fraud_history,
            spending_pattern=history.spending_pattern,
            recent_transactions=history.recent_transactions,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Similar fraud cases endpoint
@app.post("/similar-cases", response_model=List[dict])
async def get_similar_cases(query: str, top_k: int = 5, threshold: float = 0.7):
    """Search for similar fraud cases."""
    try:
        from src.agent.tools.investigation_tools import search_similar_fraud_cases
        cases = search_similar_fraud_cases(query, top_k=top_k)

        return [
            {
                'case_id': c.case_id,
                'fraud_type': c.fraud_type,
                'amount': c.amount,
                'distance': c.distance,
                'similarity_score': c.similarity_score,
                'risk_factors': c.risk_factors,
                'description': c.description,
                'investigation_notes': c.investigation_notes,
            }
            for c in cases
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Transaction stream endpoint (for simulation)
@app.post("/transactions/stream")
async def stream_transaction(transaction: dict, background_tasks: BackgroundTasks):
    """
    Endpoint for transaction stream simulator to push transactions.
    Returns investigation report if fraud probability exceeds threshold.
    """
    # This would be called by the simulator
    # For now, just return the investigation if fraud probability is high
    fraud_prob = transaction.get('fraud_probability', 0)
    threshold = 0.7

    if fraud_prob >= 0.7:
        from src.agent.tools.investigation_tools import investigate_transaction
        report = investigate_transaction(transaction, fraud_probability=fraud_prob)
        return {
            "investigated": True,
            "report": {
                "transaction_id": report.transaction_id,
                "confidence": report.confidence_score,
                "action": report.recommended_action,
                "reasoning": report.reasoning,
            }
        }
    else:
        return {"investigated": False, "reason": "Below threshold"}


# Metrics endpoint for dashboard
@app.get("/metrics/live")
async def get_live_metrics():
    """Get live evaluation metrics for dashboard."""
    # In production, this would query a metrics store
    return {
        "total_processed": 0,
        "flagged": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "false_positive_rate": 0.0,
        "last_updated": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)