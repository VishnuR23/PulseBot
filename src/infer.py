
import argparse, numpy as np
from .data import load_prices, make_dataset
import tensorflow as tf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--pulse-window", type=int, default=12)
    ap.add_argument("--pulse-k", type=float, default=2.5)
    args = ap.parse_args()

    df = load_prices(args.data)
    X, y, idx = make_dataset(df, seq_len=args.seq_len, horizon=args.horizon,
                             pulse_window=args.pulse_window, pulse_k=args.pulse_k)
    X_last = X[-1:]
    model = tf.keras.models.load_model(args.model)
    trend_prob, pulse_prob = model.predict(X_last, verbose=0)
    t = float(trend_prob[0][0]); p = float(pulse_prob[0][0])

    # naive rule
    if t > 0.6 and p < 0.5:
        signal = "BUY"
    elif t < 0.4 and p < 0.5:
        signal = "SELL"
    else:
        signal = "NO_TRADE"
    print({"trend_prob": round(t,4), "pulse_prob": round(p,4), "signal": signal})

if __name__ == "__main__":
    main()
