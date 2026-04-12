# ETF Tail-Risk Monitoring

Code and experimental materials for the paper **“Robust ETF Tail-Risk Monitoring under Data Degradation and Market Uncertainty.”**

## Overview

This repository implements a quality-aware and uncertainty-aware framework for **next-day ETF tail-risk monitoring**.

The project treats ETF tail-risk surveillance as a **monitoring and reliability problem**, rather than only a stand-alone forecasting task. In addition to producing a next-day lower-tail risk estimate, the framework evaluates the condition of service-time inputs, diagnoses model uncertainty, and applies conservative fallback logic when reliability is weakened.

The framework is designed to remain operational under:
- degraded or partially missing inputs
- stale or inconsistent price records
- changing market conditions
- elevated predictive uncertainty

## Repository Structure

- `main4.py`  
  Main implementation of the proposed ETF tail-risk monitoring framework, including walk-forward evaluation, quality-aware monitoring, uncertainty diagnostics, conservative fallback rules, and quality-validation experiments.

- `Comaprision_grach.py`  
  Comparison script that extends the main framework with a **GJR-GARCH(1,1)-t VaR baseline** for benchmark evaluation.

## Main Components

The framework contains four linked layers:

1. **Quality-control layer**  
   Screens incoming records using service-time observable diagnostics such as missing values, stale prices, invalid OHLC relations, and anomalous trading activity.

2. **Risk prediction layer**  
   Produces next-day lower-tail risk estimates for ETF returns at the 5% VaR-type level.

3. **Uncertainty layer**  
   Aggregates model dispersion, out-of-distribution diagnostics, and recent monitoring deterioration into an uncertainty score.

4. **Safe output layer**  
   Applies conservative fallback logic to generate a safer reported VaR estimate together with an operational alert state.

## Main Features

- Multi-ETF daily monitoring pipeline
- Next-day lower-tail risk prediction
- Service-time input quality checks
- Uncertainty diagnostics for deployment reliability
- Conservative fallback mechanism for safer outputs
- Daily alert generation
- Synthetic input degradation experiments
- Cross-asset evaluation under stressed conditions
- Benchmark comparison against GJR-GARCH(1,1)-t VaR

## Data Requirements

The code is designed to use the following input files:

- `multiasset_daily_10y_panel_model.csv`
- `VIXCLS.csv`
- `zero_coupon_yield.csv`

Optional:
- `VXVCLS.csv`

## Running the Code

Update the base data directory in the configuration or set the corresponding environment path, then run:

```bash
python main4.py
