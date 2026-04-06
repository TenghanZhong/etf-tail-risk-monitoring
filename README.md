# ETF Tail-Risk Monitoring

Code and experiments for the paper **"Robust ETF Tail-Risk Monitoring under Data Degradation and Market Uncertainty."**

## Overview

This repository implements a **quality-aware** and **uncertainty-aware** ETF tail-risk monitoring framework for next-day risk surveillance.

Instead of treating ETF tail-risk prediction as a stand-alone forecasting task, this project formulates it as a **service problem**. The system combines:

- service-time data quality checks
- lower-tail risk prediction
- uncertainty diagnostics
- conservative fallback and alert generation

The goal is not only to produce a next-day lower-tail risk estimate, but also to make monitoring outputs more reliable when **inputs degrade**, **market regimes shift**, or **model confidence weakens**.

## Main Features

- Multi-ETF daily risk monitoring pipeline
- Quality-control layer for missing fields, stale prices, invalid OHLC relations, and other service-time issues
- Lower-tail prediction at the **5% VaR-type** level
- Uncertainty layer based on:
  - ensemble dispersion
  - out-of-distribution diagnostics
  - recent monitoring deterioration
- Conservative fallback mechanism for safer operational outputs
- Alert generation for daily monitoring
- Validation under **synthetic input degradation**
- Cross-asset and stress-period evaluation

## Method Summary

The framework contains four linked components:

1. **Data and quality-control layer**  
   Screens incoming records and computes quality states from service-time observable diagnostics.

2. **Risk prediction layer**  
   Produces next-day lower-tail risk estimates for ETF returns.

3. **Uncertainty layer**  
   Aggregates model disagreement, regime distance / OOD behavior, and recent drift into an uncertainty score.

4. **Safe output layer**  
   Combines the model-based estimate with conservative fallback logic to produce a safer operational VaR output and daily alert state.

## Repository Structure

A suggested structure is:

```text
.
├── data/
│   ├── multiasset_daily_10y_panel_model.csv
│   ├── VIXCLS.csv
│   └── zero_coupon_yield.csv
├── outputs/
│   ├── daily_service_outputs.csv
│   ├── evaluation_summary.csv
│   ├── latest_service_snapshot.csv
│   ├── monthly_service_summary.csv
│   └── run_config.json
├── figures/
│   ├── figure1_pipeline_diagram.png
│   ├── figure3_stress_vs_nonstress.png
│   ├── figure4a_monthly_alerts.png
│   └── figure4b_uncertainty_heatmap.png
├── src/
│   ├── data_processing.py
│   ├── quality_layer.py
│   ├── model.py
│   ├── uncertainty.py
│   ├── fallback.py
│   ├── evaluation.py
│   └── plots.py
├── main.py
├── requirements.txt
└── README.md
