import pandas as pd
import numpy as np
from pathlib import Path

MASKED_PATH = Path("data/masked")
PROCESSED_PATH = Path("data/processed")

DATE_COLUMNS = ["Created", "Updated", "Due date"]

def combine_multivalue_columns(df: pd.DataFrame, base_name: str) -> pd.DataFrame:
    """
    Combine columns like:
    fix_versions, fix_versions.1, fix_versions.2
    into a single column: fix_versions_combined
    """

    cols = [col for col in df.columns if col.startswith(base_name)]

    if not cols:
        return df

    df[f"{base_name}_combined"] = (
        df[cols]
        .apply(lambda row: " | ".join(
            [str(val) for val in row if pd.notna(val)]
        ), axis=1)
    )

    df = df.drop(columns=cols)

    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names:
    - lowercase
    - replace spaces with underscores
    - remove parentheses
    """
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df

def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date columns to datetime.
    """
    for col in DATE_COLUMNS:
        col_normalized = col.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if col_normalized in df.columns:
            df[col_normalized] = pd.to_datetime(
                df[col_normalized],
                format="%d/%b/%y %I:%M %p",
                errors="coerce"
            )
    return df

def filter_closed_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only Release Ready issues.
    """
    return df[df["status"] == "Release Ready"].copy()

def calculate_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lead time = Updated - Created in days.
    """
    total_seconds_in_day = 86400  # seconds in a day 
    df["lead_time_days"] = (
        (df["updated"] - df["created"]).dt.total_seconds() / total_seconds_in_day
    )
    
    # Remove negative lead times (if any)
    df = df[df["lead_time_days"] >= 0]
    
    # Remove extreme outliers (e.g., lead time > 365 days)
    df = df[df["lead_time_days"] <= 365]
    
    return df

def preprocess_dataset(input_filename: str, output_filename: str) -> None:

    input_path = MASKED_PATH / input_filename
    output_path = PROCESSED_PATH / output_filename

    df = pd.read_csv(input_path)

    df = clean_column_names(df)
    df = convert_dates(df)
    df = filter_closed_issues(df)
    df = calculate_lead_time(df)
    df = combine_multivalue_columns(df, "fix_versions")
    df = combine_multivalue_columns(df, "affects_versions")
    df = combine_multivalue_columns(df, "components")

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Processed dataset saved to {output_path}")

if __name__ == "__main__":
    preprocess_dataset("masked_dataset.csv", "processed_dataset.csv")
