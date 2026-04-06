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

