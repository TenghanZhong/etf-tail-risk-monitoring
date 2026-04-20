# ETF Tail-Risk Monitoring

Code and experimental materials for the paper:

**Reliability-Aware ETF Tail-Risk Monitoring under Data Degradation and Market Uncertainty**

## Overview

This repository implements a quality-aware and uncertainty-aware framework for next-day ETF tail-risk monitoring.

Unlike a standard stand-alone VaR forecasting pipeline, this project treats ETF downside-risk estimation as a **monitoring and reliability problem**. In addition to producing a next-day lower-tail risk estimate, the framework evaluates service-time input quality, diagnoses predictive uncertainty, applies conservative fallback logic, and generates operational alert states.

The framework is designed to remain usable under:

- degraded or partially missing inputs
- stale or inconsistent price records
- changing market conditions
- elevated predictive uncertainty

## Repository Structure

- `run_tail_risk_monitoring.py`  
  Main implementation of the proposed ETF tail-risk monitoring framework, including walk-forward evaluation, quality-aware monitoring, uncertainty diagnostics, fallback logic, and evaluation summaries.

- `compare_gjr_garch.py`  
  Benchmark comparison script that evaluates the proposed framework against a GJR-GARCH(1,1)-t style VaR baseline.

- `quality-layer.py`  
  Quality-validation and corruption-analysis script for testing the effect of degraded service-time inputs on monitoring reliability.

- `plot.py`  
  Figure-generation utilities for producing paper-ready plots and diagrams from exported result files.

- `multiasset_daily_10y_panel_model.csv`  
  Main multi-ETF panel dataset used by the framework.

- `VIXCLS.csv`  
  VIX data used for market-state and stress-related features.

- `zero_coupon_yield.csv`  
  Zero-coupon yield curve data used in macro-financial feature construction.

- `VXVCLS.csv`  
  Optional auxiliary volatility index input.

- `Result/`  
  Example output folder containing representative result tables and figures.

## Main Framework

The monitoring framework contains four linked layers:

### 1. Quality-Control Layer
Screens incoming records using service-time observable diagnostics, including:

- missing values
- stale prices
- invalid OHLC relations
- jump-like abnormal returns
- abnormal trading activity

### 2. Risk Prediction Layer
Produces next-day lower-tail risk estimates for ETF returns, targeting a 5% VaR-type threshold.

### 3. Uncertainty Layer
Builds an uncertainty score from multiple sources, including:

- model dispersion
- out-of-distribution diagnostics
- recent monitoring deterioration

### 4. Safe Output Layer
Applies conservative fallback logic to generate a safer reported risk estimate together with an operational alert state.

## Main Features

- Multi-ETF daily monitoring pipeline
- Next-day lower-tail risk estimation
- Service-time input quality checks
- Reliability-oriented uncertainty diagnostics
- Conservative fallback mechanism
- Daily operational alert generation
- Synthetic input degradation experiments
- Cross-asset evaluation under stressed conditions
- Benchmark comparison against GJR-GARCH-style VaR

## Data Requirements

Place the following files in the repository root, or in a directory referenced by an environment variable if you later modify the path configuration:

- `multiasset_daily_10y_panel_model.csv`
- `VIXCLS.csv`
- `zero_coupon_yield.csv`

Optional:

- `VXVCLS.csv`

## Environment Setup

Create a clean Python environment and install dependencies:

```bash
python -m venv .venv
