import pandas as pd

def _resolved_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter Release Ready tickets only
    """
    return df[df['status'].str.lower() == 'release ready'].copy()

def average_lead_time(df: pd.DataFrame) -> float:
    """
    Lead Time global average. Only resolved tickets.
    """
    resolved = _resolved_only(df)
    return resolved['lead_time_days'].mean()

def lead_time_percentiles(df: pd.DataFrame) -> pd.Series:
    """
    Global Lead Time percentiles (P50, P75, P90).
    Only resolved tickets.
    """
    resolved = _resolved_only(df)

    percentiles = resolved['lead_time_days'].quantile([0.5, 0.75, 0.9])
    percentiles.index = ['P50', 'P75', 'P90']

    return percentiles

def average_lead_time_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lead Time average grouped by priority.
    """
    resolved = _resolved_only(df)
    return (
        resolved
        .groupby('priority')['lead_time_days']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

def lead_time_percentiles_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lead Time percentiles grouped by priority.
    """
    resolved = _resolved_only(df)

    return (
        resolved
        .groupby('priority')['lead_time_days']
        .quantile([0.5, 0.75, 0.9])
        .unstack()
        .rename(columns={
            0.5: 'P50',
            0.75: 'P75',
            0.9: 'P90'
        })
        .reset_index()
    )

def blocker_lead_time(df: pd.DataFrame) -> float:
    """
    Lead Time average only for Blockers.
    """
    resolved = _resolved_only(df)
    blockers = resolved[resolved['priority'].str.lower() == 'blocker']
    return blockers['lead_time_days'].mean()

def priority_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
   Distribution of priorities in percentage.
    """
    distribution = (
        df['priority']
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
    )

    distribution.columns = ['priority', 'percentage']
    return distribution

def quarterly_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantity of tickets resolved per quarter.
    """
    resolved = _resolved_only(df).copy()

    resolved['quarter'] = resolved['updated'].dt.to_period('Q')

    velocity = (
        resolved
        .groupby('quarter')
        .size()
        .reset_index(name='tickets_resolved')
        .sort_values('quarter')
    )

    velocity['quarter'] = velocity['quarter'].astype(str)

    return velocity