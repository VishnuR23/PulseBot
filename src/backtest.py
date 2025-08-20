
import argparse, numpy as np
import tensorflow as tf
from .data import load_prices, make_dataset
from .utils import sharpe, max_drawdown, plot_backtest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--pulse-window", type=int, default=12)
    ap.add_argument("--pulse-k", type=float, default=2.5)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="outputs/backtest.png")
    args = ap.parse_args()

    df = load_prices(args.data)
    X, y, idx = make_dataset(df, seq_len=args.seq_len, horizon=args.horizon,
                             pulse_window=args.pulse_window, pulse_k=args.pulse_k)
    closes = df["close"].values
    ts = df["timestamp"].values

    model = tf.keras.models.load_model(args.model)
    tp, pp = model.predict(X, verbose=0)
    tp, pp = tp.squeeze(), pp.squeeze()

    # simple strategy: enter long if trend>0.6 and pulse<0.5; enter short if trend<0.4 and pulse<0.5; else flat.
    pos = np.zeros_like(tp)
    pos[tp>0.6] = 1
    pos[tp<0.4] = -1
    # avoid trading when pulse risk high
    pos[pp>0.5] = 0

    # compute returns on horizon step ahead (approx): r_{t+h} applied to position at t
    # For simplicity use 1-step log return as proxy
    logp = np.log(closes)
    r = np.diff(logp, prepend=logp[0])
    strat = pos * r[-len(pos):]
    equity = (1 + strat).cumprod()

    sr = sharpe(strat, freq=252*390)  # minute-ish
    mdd = max_drawdown(equity)
    print({"sharpe": float(sr), "max_drawdown": float(mdd), "final_equity": float(equity[-1])})

    if args.plot:
        plot_backtest(ts[-len(equity):], closes[-len(equity):], equity, idx, args.out)
        print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
