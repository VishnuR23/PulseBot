
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sharpe(returns, freq=252):
    r = np.array(returns, dtype=float)
    if r.size == 0: return 0.0
    mu, sd = np.mean(r), np.std(r, ddof=1) + 1e-12
    return (mu * freq) / (sd * (np.sqrt(freq)))

def max_drawdown(equity_curve):
    ec = np.array(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(ec)
    dd = (ec - peaks) / peaks
    return dd.min()

def plot_backtest(ts, prices, equity, signals_idx, path):
    plt.figure(figsize=(10,6))
    ax1 = plt.gca()
    ax1.plot(ts, prices, label="Close")
    ax1.set_ylabel("Price")
    ax2 = ax1.twinx()
    ax2.plot(ts, equity, color="tab:green", alpha=0.6, label="Equity")
    for t in signals_idx:
        ax1.axvline(ts[t], color="tab:orange", alpha=0.1)
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.title("Backtest — Price & Equity")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
