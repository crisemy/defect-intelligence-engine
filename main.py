from pathlib import Path
import pandas as pd

# Import modules
from src.preprocessing.preprocessing import preprocess_dataset
from src.analysis.kpi_engine import (
    average_lead_time,
    average_lead_time_by_priority,
    blocker_lead_time,
    priority_distribution,
    quarterly_velocity,
    lead_time_percentiles,
    lead_time_percentiles_by_priority
)

MASKED_FILENAME = "masked_dataset.csv"
PROCESSED_FILENAME = "processed_dataset.csv"

PROCESSED_PATH = Path("data/processed") / PROCESSED_FILENAME

def run_pipeline():
    print("Running preprocessing...")
    preprocess_dataset(MASKED_FILENAME, PROCESSED_FILENAME)

    print("Loading processed dataset...")
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["created", "updated"])

    print("\n===== KPI RESULTS =====\n")

    avg_lt = average_lead_time(df)
    print(f"Average Lead Time (global): {avg_lt:.2f} days")

    blocker_lt = blocker_lead_time(df)
    print(f"Average Lead Time (Blockers): {blocker_lt:.2f} days")

    print("\nLead Time by Priority:")
    print(average_lead_time_by_priority(df))

    print("\nPriority Distribution (%):")
    print(priority_distribution(df))

    print("\nQuarterly Velocity:")
    print(quarterly_velocity(df))
    
    print("\nLead Time Percentiles (Global):")
    print(lead_time_percentiles(df))

    print("\nLead Time Percentiles by Priority:")
    print(lead_time_percentiles_by_priority(df))

if __name__ == "__main__":
    run_pipeline()