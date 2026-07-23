# Reliability-Aware ETF Tail-Risk Monitoring

Code and experimental materials for **Reliability-Aware ETF Tail-Risk Monitoring**, accepted at IEEE SMC 2026.

The repository implements a rolling walk-forward service for next-day ETF lower-tail risk monitoring. The pipeline combines input-quality diagnostics, bootstrap quantile prediction, uncertainty scoring, conservative fallback logic, and operational alerts. Additional experiments evaluate prediction-time data degradation and compare the service output with standard VaR benchmarks.

## Repository structure

- `run_tail_risk_monitoring.py`: main experiment and component ablations.
- `compare_gjr_garch.py`: GJR-GARCH(1,1)-t benchmark comparison.
- `quality-layer.py`: focused input-quality validation experiment.
- `plot.py`: figure-generation utilities.
- `src/`: experiment implementation modules.
- `Result/`: representative result tables and figures.

## Required data

Place these files in the repository root or in the directory specified by `ETF_TAIL_RISK_DATA_DIR`:

- `multiasset_daily_10y_panel_model.csv`
- `VIXCLS.csv`
- `zero_coupon_yield.csv`

Optional input:

- `VXVCLS.csv`

The ETF panel must contain `date`, `symbol`, OHLC prices, volume, returns, rolling volatility variables, range-based volatility proxies, and drawdown variables used by the source modules.

## Environment setup

Python 3.10 or later is recommended.

### Linux and macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data-directory configuration

Without an environment variable, the scripts read data from the repository root. A separate data directory can be supplied as follows.

Linux and macOS:

```bash
export ETF_TAIL_RISK_DATA_DIR=/path/to/data
```

Windows PowerShell:

```powershell
$env:ETF_TAIL_RISK_DATA_DIR = "C:\path\to\data"
```

## Running the experiments

Main monitoring experiment:

```bash
python run_tail_risk_monitoring.py
```

GJR-GARCH comparison:

```bash
python compare_gjr_garch.py
```

Quality-layer validation:

```bash
python quality-layer.py
```

Default output directories:

- `results/`
- `results_gjr_garch/`
- `results_quality_validation/`

## Generating figures

`plot.py` reads `daily_service_outputs.csv` and `monthly_service_summary.csv` from the configured data directory.

```bash
python plot.py
```

Figures are written to the `figures/` subdirectory.

## Main configuration

The experiment modules use the following baseline settings:

- 5% lower-tail target
- 756-trading-day training window
- 63-trading-day calibration window
- 63-trading-day retraining interval
- five bootstrap quantile models
- 252-day and 63-day historical VaR windows
- 15% row-level corruption probability in degradation experiments

All model, fallback, alert, and corruption parameters are specified in the source modules.

## Reproducibility notes

- Training, calibration, and prediction periods are separated chronologically.
- Calibration offsets use observations strictly preceding each prediction period.
- Symbol-level uncertainty states use prior evaluable uncertainty scores only.
- Synthetic faults are injected only into prediction-time observable inputs.
- Corruption experiments use fixed seeds and fixed design parameters.

## Output interpretation

Safe VaR is a lower-tail return threshold. More negative values indicate a more conservative threshold. A breach occurs when the next-day realized return falls below the reported threshold. Alert states are operational diagnostics and are distinct from the low/elevated uncertainty split used in the empirical analysis.
