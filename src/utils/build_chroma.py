"""
Chroma Vector Database Builder for Fraud Case RAG

Embeds known fraud cases into Chroma vector database for similarity search.
The agent investigation layer uses this to retrieve similar past fraud cases.
"""

import pandas as pd
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_SYNTHETIC = PROJECT_ROOT / 'data' / 'synthetic'
CHROMA_PATH = PROJECT_ROOT / 'chroma_db'

# Model for embeddings
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
COLLECTION_NAME = 'fraud_cases'


def load_fraud_cases():
    """Load fraud cases from JSON."""
    json_path = DATA_SYNTHETIC / 'known_fraud_cases.json'
    with open(json_path, 'r') as f:
        cases = json.load(f)
    print(f"Loaded {len(cases):,} fraud cases")
    return cases


def create_embeddings(cases, model):
    """Create embeddings for fraud case descriptions."""
    print("\nCreating embeddings...")
    descriptions = [case['description'] for case in cases]

    # Batch encode for efficiency
    embeddings = model.encode(
        descriptions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Created embeddings: {embeddings.shape}")
    return embeddings


def build_chroma_database(cases, embeddings):
    """Build Chroma vector database."""
    print("\nBuilding Chroma vector database...")

    # Initialize Chroma client
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # Reset if exists (for clean rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Known fraud cases for RAG similarity search"}
    )

    # Prepare data for insertion
    ids = [case['case_id'] for case in cases]
    documents = [case['description'] for case in cases]

    # Prepare metadata (Chroma requires flat metadata, no nested objects)
    metadatas = []
    for case in cases:
        meta = {
            'case_id': case['case_id'],
            'fraud_type': case['fraud_type'],
            'amount': case['amount'],
            'confirmed_fraud': case['confirmed_fraud'],
            'customer_id': case['customer_id'],
            'transaction_id': case['transaction_id'],
            'deviation_from_baseline': case['deviation_from_baseline'],
            'time_since_last_txn': case['time_since_last_txn'],
            # Store risk factors as JSON string
            'risk_factors': case['risk_factors'],
            'investigation_notes': case['investigation_notes'][:500],  # Truncate long notes
        }
        metadatas.append(meta)

    # Add to collection in batches
    batch_size = 100
    for i in range(0, len(cases), batch_size):
        end = min(i + batch_size, len(cases))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"  Added batch {i//batch_size + 1}: {i} to {end}")

    print(f"\nChroma database built successfully!")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total documents: {collection.count()}")
    print(f"Database path: {CHROMA_PATH}")

    return client, collection


def test_similarity_search(collection, model):
    """Test similarity search with a few queries."""
    print("\nTesting similarity search...")

    test_queries = [
        "Large transaction amount deviation from customer baseline with high velocity",
        "Account takeover with sudden location change and new device",
        "Small test transaction followed by large fraudulent purchase",
        "Customer disputes legitimate transaction claiming fraud",
        "Stolen card used at multiple gas stations in short time"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_embedding = model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )

        for i, (doc_id, doc, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"  {i+1}. {doc_id} | {metadata['fraud_type']} | Amount: ${metadata['amount']:.2f} | Distance: {distance:.4f}")
            print(f"     {doc[:100]}...")


def main():
    """Build Chroma vector database for fraud case RAG."""
    print("=" * 80)
    print("BUILDING CHROMA VECTOR DATABASE FOR FRAUD CASE RAG")
    print("=" * 80)

    # Load fraud cases
    cases = load_fraud_cases()

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Create embeddings
    embeddings = create_embeddings(cases, model)

    # Build Chroma database
    client, collection = build_chroma_database(cases, embeddings)

    # Test similarity search
    test_similarity_search(collection, model)

    print("\n" + "=" * 80)
    print("CHROMA VECTOR DATABASE BUILD COMPLETE")
    print("=" * 80)
    print(f"\nDatabase ready at: {CHROMA_PATH}")
    print(f"Collection: {COLLECTION_NAME} ({collection.count()} documents)")
    print("\nAgent can now retrieve similar fraud cases via semantic search!")
    print("\nNext: Build agent investigation layer with tool calling (Phase 4)")


if __name__ == "__main__":
    main()