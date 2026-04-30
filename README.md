![QA Architect header](./images/github_header.png)

# Defect Intelligence & Lead Time Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3-orange)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a full analytics and predictive pipeline over Jira defect data.

It provides:

- Data preprocessing pipeline
- KPI computation engine
- Interactive Streamlit dashboard
- Baseline machine learning model for lead time prediction
- Error distribution analysis and model evaluation

The objective is to transform raw defect lifecycle data into actionable operational insights.

---

## Problem Statement

Engineering organizations often lack quantitative visibility into:

- Defect lead time behavior
- Priority-based resolution patterns
- Resolution velocity trends
- Predictability of defect lifecycle duration

This project addresses those gaps using data analytics and baseline machine learning.

---

## Architecture
```bash
Defect-intelligence-engine/
│
├── main.py
├── app.py
├── kpi_engine.py
├── modeling/
│ ├── __init__.py
│ └── ml_model.py
│
├── data/
│ ├── raw/
│ └── processed/
│
├── data_ingestion/ (reserved for future extension)
├── feature_engineering/ (reserved for future extension)
│
├── requirements.txt
└── README.md
```
---

## KPI Engine

The system computes:

- Average Lead Time (global)
- Average Lead Time by Priority
- Lead Time Percentiles (P50, P75, P90)
- Lead Time Percentiles by Priority
- Priority Distribution
- Quarterly Resolution Velocity

---

## Predictive Modeling

A baseline Random Forest Regressor is trained to predict:

> lead_time_days

Features used:

- Priority (categorical)
- Created Month
- Created Weekday
- Created Quarter

Evaluation metrics:

- MAE (Mean Absolute Error)
- Baseline MAE (historical mean predictor)
- R² Score
- Error distribution analysis

---

## Results

The baseline ML model underperformed compared to the historical mean baseline.

Key observations:

- High variance in lead time
- Presence of extreme outliers
- Limited predictive signal in available features
- Error distribution centered but widely dispersed

This indicates that defect lead time is influenced by organizational and contextual variables not present in the dataset (e.g., workload, dependencies, team capacity).

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly

---

## How to Run

### 1. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run preprocessing pipeline
```bash
python main.py
```
### 4. Launch dashboard
```bash
streamlit run app.py
```
---

## Project Scope

This project is intentionally scoped as a foundational data analytics and ML baseline implementation.

It demonstrates:

- End-to-end pipeline development
- Modular architecture
- Statistical reasoning
- Critical evaluation of model performance

Future iterations may include:

- Feature enrichment
- Time-based cross-validation
- Advanced models (XGBoost)
- SHAP explainability
- MLOps integration

# Author

Cristian N.

QA Engineer with 20+ years of experience in software testing and automation.

MSc Candidate in Data Science & Artificial Intelligence.

Research interests include:

* Experimental QA engineering
* QA Architecture
* Reliability testing
* AI-assisted quality assurance
* Data-driven software stability analysis
