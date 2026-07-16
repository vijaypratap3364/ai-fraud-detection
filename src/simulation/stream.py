"""
Transaction Stream Simulator

Replays the Kaggle dataset as a simulated real-time transaction stream.
Pushes transactions to the FastAPI backend one at a time with configurable timing.
"""

import pandas as pd
import numpy as np
import time
import requests
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import random

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_SYNTHETIC = PROJECT_ROOT / 'data' / 'synthetic'

# API Configuration
API_BASE = "http://localhost:8000"
STREAM_ENDPOINT = f"{API_BASE}/transactions/stream"
HEALTH_ENDPOINT = f"{API_BASE}/health"

# Default configuration
DEFAULT_RATE = 2.0  # transactions per second
DEFAULT_SPEED = 1.0  # 1.0 = real-time
DEFAULT_THRESHOLD = 0.7


class TransactionStreamer:
    """Simulates real-time transaction stream from historical data."""

    def __init__(
        self,
        api_base: str = API_BASE,
        rate: float = DEFAULT_RATE,
        speed: float = DEFAULT_SPEED,
        threshold: float = DEFAULT_THRESHOLD,
        shuffle: bool = True,
        loop: bool = False,
        verbose: bool = True
    ):
        self.api_base = api_base
        self.rate = rate  # transactions per second
        self.speed = speed  # replay speed multiplier
        self.threshold = threshold  # fraud detection threshold
        self.shuffle = shuffle
        self.loop = loop
        self.verbose = verbose

        self.interval = 1.0 / rate / speed  # seconds between transactions
        self.total_processed = 0
        self.total_investigated = 0
        self.total_flagged = 0

    def load_data(self):
        """Load and prepare transaction data."""
        print("\nLoading transaction data...")
        txn_path = PROJECT_ROOT / 'data' / 'synthetic' / 'transaction_history.csv'

        if not txn_path.exists():
            raise FileNotFoundError(f"Transaction history not found: {txn_path}")

        df = pd.read_csv(txn_path)

        # Sort by Time if not already sorted
        if 'Time' in df.columns:
            df = df.sort_values('Time').reset_index(drop=True)

        # Shuffle if requested
        if self.shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Shuffled {len(df):,} transactions")

        print(f"Loaded {len(df):,} transactions")
        print(f"Fraud rate: {(df['Class'] == 1).mean():.3%}")
        print(f"Time span: {df['Time'].max() / 3600:.1f} hours")

        return df

    def check_api_health(self) -> bool:
        """Check if API is reachable."""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def send_transaction(self, transaction: dict) -> Optional[dict]:
        """Send a single transaction to the API."""
        try:
            transaction['fraud_probability'] = float(transaction.get('Class', 0))

            response = requests.post(
                f"{self.api_base}/transactions/stream",
                json=transaction,
                timeout=5
            )

            if response.status_code == 200:
                return response.json()
            else:
                if self.verbose:
                    print(f"  API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            if self.verbose:
                print("  API timeout")
            return None
        except requests.exceptions.ConnectionError:
            if self.verbose:
                print("  Connection error - API may be down")
            return None
        except Exception as e:
            if self.verbose:
                print(f"  Error: {e}")
            return None

    def format_transaction(self, row: pd.Series) -> dict:
        """Format a transaction row for the API."""
        return {
            'transaction_id': f"TXN_{int(row.get('transaction_number', 0)):08d}",
            'customer_id': row.get('customer_id', 'UNKNOWN'),
            'timestamp': float(row.get('Time', 0)),
            'amount': float(row.get('Amount', 0)),
            'features': {col: float(row[col]) for col in row.index if col.startswith('V')},
            'class': int(row.get('Class', 0)),
        }

    def run(self):
        """Main simulation loop."""
        # Load data
        df = self.load_data()

        # Check API
        print(f"\nChecking API at {self.api_base}...")
        if not self.check_api_health():
            print("ERROR: API not reachable. Start the FastAPI server first:")
            print("  python src/api/main.py")
            return

        print("API is healthy!")

        print(f"\nStarting simulation...")
        print(f"  Rate: {self.rate} txns/sec (speed: {self.speed}x)")
        print(f"  Interval: {self.interval:.3f} seconds")
        print(f"  Threshold: {self.threshold}")
        print(f"  Press Ctrl+C to stop\n")

        last_time = time.time()
        start_time = time.time()

        try:
            while True:
                for _, row in df.iterrows():
                    loop_start = time.time()

                    # Format and send transaction
                    transaction = self.format_transaction(row)
                    result = self.send_transaction(transaction)

                    self.total_processed += 1

                    if result and result.get('investigated'):
                        self.total_investigated += 1
                        if self.verbose:
                            report = result.get('report', {})
                            print(f"  INVESTIGATED: {report.get('transaction_id')} | "
                                  f"Confidence: {report.get('confidence', 0):.2f} | "
                                  f"Action: {report.get('action')}")

                    # Print progress every 100 transactions
                    if self.total_processed % 100 == 0 and self.verbose:
                        elapsed = time.time() - start_time
                        rate = self.total_processed / elapsed if elapsed > 0 else 0
                        print(f"  Progress: {self.total_processed:,} processed | "
                              f"{rate:.1f} txns/sec | "
                              f"Investigated: {self.total_investigated}")

                    # Rate limiting
                    elapsed = time.time() - loop_start
                    sleep_time = self.interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                # End of dataset
                if not self.loop:
                    print("\nEnd of dataset reached. Stopping.")
                    break
                else:
                    print("\nEnd of dataset. Looping...")
                    if self.shuffle:
                        df = df.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)

        except KeyboardInterrupt:
            print("\n\nSimulation stopped by user")

        # Final stats
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total processed: {self.total_processed:,}")
        print(f"Total investigated: {self.total_investigated:,}")
        print(f"Elapsed time: {elapsed:.1f} seconds")
        print(f"Average rate: {self.total_processed/elapsed:.1f} txns/sec")


def main():
    parser = argparse.ArgumentParser(description="Transaction Stream Simulator")
    parser.add_argument('--rate', type=float, default=DEFAULT_RATE,
                        help='Transactions per second (default: 2.0)')
    parser.add_argument('--speed', type=float, default=DEFAULT_SPEED,
                        help='Replay speed multiplier (default: 1.0)')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='Fraud detection threshold (default: 0.7)')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Do not shuffle transactions')
    parser.add_argument('--loop', action='store_true',
                        help='Loop dataset continuously')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--api', type=str, default=API_BASE,
                        help=f'API base URL (default: {API_BASE})')

    args = parser.parse_args()

    streamer = TransactionStreamer(
        api_base=args.api,
        rate=args.rate,
        speed=args.speed,
        threshold=args.threshold,
        shuffle=not args.no_shuffle,
        loop=args.loop,
        verbose=not args.quiet
    )

    streamer.run()


if __name__ == "__main__":
    main()