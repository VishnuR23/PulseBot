
# PulseTrader‑TF — TensorFlow Trend & Pulse Predictor (Simple Trading Bot)

**PulseTrader‑TF** is a small, educational project that trains a **TensorFlow** model to
predict (1) short‑term **trend** (up vs. down) and (2) sudden **pulses** (spikes) using OHLCV time‑series.
It includes data prep, training, inference, and a basic walk‑forward **backtester**.

---

## ✨ What it does
- Builds features (returns, rolling vol, RSI, moving averages, z‑scores).
- Trains a **Conv1D + LSTM** model with two heads:
  - **Trend head** → probability the price will be higher after `horizon` steps.
  - **Pulse head** → probability of an abnormal move in the next `pulse_window`.
- Runs a simple **backtest** to evaluate signals & PnL.
- Ships with **synthetic** OHLCV so it works offline. You can swap in real CSVs.

**CSV schema expected:** `timestamp,open,high,low,close,volume` (header row, any frequency).

---

## 🧪 Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Train
python src/train.py --data data/synthetic_prices.csv --model outputs/pulse_trader.h5 --seq-len 64 --horizon 12

# 2) Backtest (plots saved to outputs/)
python src/backtest.py --data data/synthetic_prices.csv --model outputs/pulse_trader.h5 --plot

# 3) Inference on most recent window
python src/infer.py --data data/synthetic_prices.csv --model outputs/pulse_trader.h5
```

### Using your own data
Place a CSV at `data/your_prices.csv` with columns: `timestamp,open,high,low,close,volume`.
Then pass `--data data/your_prices.csv` to the commands above.

---

## 🛠 Tech
- **TensorFlow / Keras** (Conv1D + LSTM)
- **pandas / numpy / matplotlib**
- Pure‑Python indicators (no external TA dependency)

---

## 📁 Structure
```
src/
  data.py        # load CSV, build features & sliding windows
  features.py    # indicators (RSI, MAs, zscores, rolling vol, etc.)
  model.py       # TF model factory (Conv1D + LSTM, dual-head)
  train.py       # training script
  infer.py       # single-window inference
  backtest.py    # walk-forward backtest + metrics + plot
  utils.py       # metrics (sharpe, drawdown), plotting helpers
data/
  synthetic_prices.csv  # generated GBM-like series with regime "pulses"
outputs/
  (models & plots land here)
```

---
