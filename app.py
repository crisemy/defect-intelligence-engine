import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
from src.modeling.ml_model import train_lead_time_model

from src.analysis.kpi_engine import (
    average_lead_time,
    blocker_lead_time,
    lead_time_percentiles,
    average_lead_time_by_priority,
    quarterly_velocity
)

PROCESSED_PATH = Path("data/processed/processed_dataset.csv")

st.set_page_config(page_title="Defect Intelligence Engine", layout="wide")

st.title("Defect Intelligence Engine")
st.markdown("Analytical dashboard for QA performance and defect resolution behavior.")

# Load dataset
df = pd.read_csv(PROCESSED_PATH, parse_dates=["created", "updated"])

# ======================
# Sidebar Filters
# ======================

st.sidebar.header("Filters")

priorities = st.sidebar.multiselect(
    "Select Priority",
    options=sorted(df["priority"].unique()),
    default=sorted(df["priority"].unique())
)

date_range = st.sidebar.date_input(
    "Updated Date Range",
    value=(df["updated"].min(), df["updated"].max())
)

filtered_df = df[
    (df["priority"].isin(priorities)) &
    (df["updated"].dt.date >= date_range[0]) &
    (df["updated"].dt.date <= date_range[1])
]

# ======================
# KPI Metrics
# ======================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Lead Time",
        f"{average_lead_time(filtered_df):.2f} days",
        help="Mean resolution time. Sensitive to extreme outliers."
    )

with col2:
    percentiles = lead_time_percentiles(filtered_df)
    st.metric(
        "P50 (Median)",
        f"{percentiles['P50']:.2f} days",
        help="Typical resolution time. 50% of tickets resolve faster than this."
    )

with col3:
    st.metric(
        "P90",
        f"{percentiles['P90']:.2f} days",
        help="90% of tickets resolve before this value. Indicates process stability."
    )

with col4:
    st.metric(
        "Blocker Avg",
        f"{blocker_lead_time(filtered_df):.2f} days",
        help="Average resolution time specifically for Blocker priority."
    )

# ======================
# Distribution Chart
# ======================

st.subheader("Lead Time Distribution")

fig = px.histogram(
    filtered_df,
    x="lead_time_days",
    nbins=40,
    title="Distribution of Lead Time (Days)"
)

st.plotly_chart(fig, use_container_width=True)

# ======================
# Boxplot by Priority
# ======================

st.subheader("Lead Time by Priority")

fig2 = px.box(
    filtered_df,
    x="priority",
    y="lead_time_days",
    title="Lead Time Dispersion by Priority"
)

st.plotly_chart(fig2, use_container_width=True)

# ======================
# Quarterly Velocity
# ======================

st.subheader("Quarterly Resolution Velocity")

velocity = quarterly_velocity(filtered_df)

fig3 = px.bar(
    velocity,
    x="quarter",
    y="tickets_resolved",
    title="Tickets Resolved per Quarter"
)

st.plotly_chart(fig3, use_container_width=True)

# ======================
# Interpretation Section
# ======================

st.markdown("## Interpretation Guide")

st.markdown("""
- **Mean vs Median**: A large gap indicates long-tail delays.
- **P90**: Strong indicator of structural inefficiencies.
- **Boxplot**: Reveals dispersion and extreme cases.
- **Velocity Trend**: Helps identify productivity shifts over time.
""")

# ======================
# Prediction Section
# ======================

st.markdown("---")
st.header("Predictive Lead Time Model")

with st.spinner("Training model..."):

    model, metrics, y_test, y_pred = train_lead_time_model(df)

col1, col2, col3 = st.columns(3)

col1.metric("Model MAE (days)", metrics["MAE"])
col2.metric("Baseline MAE (days)", metrics["Baseline_MAE"])
col3.metric("R² Score", metrics["R2"])

st.markdown("""
**Interpretation:**
- MAE: Average prediction error in days.
- Baseline MAE: Error if predicting historical mean.
- R²: Variance explained by the model.
""")

improvement = round(
    (metrics["Baseline_MAE"] - metrics["MAE"]) / metrics["Baseline_MAE"] * 100,
    1
)

st.success(f"Model improves baseline error by {improvement}%")

# ======================
# Errors Section
# ======================

errors = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

errors["Error"] = errors["Actual"] - errors["Predicted"]

fig = px.histogram(
    errors,
    x="Error",
    nbins=40,
    title="Prediction Error Distribution"
)

st.plotly_chart(fig, use_container_width=True)
