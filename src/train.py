
import argparse, os, json
import numpy as np
from .data import load_prices, make_dataset, train_val_split
from .model import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="outputs/pulse_trader.h5")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--pulse-window", type=int, default=12)
    ap.add_argument("--pulse-k", type=float, default=2.5)
    args = ap.parse_args()

    df = load_prices(args.data)
    X, y, idx = make_dataset(df, seq_len=args.seq_len, horizon=args.horizon,
                             pulse_window=args.pulse_window, pulse_k=args.pulse_k)
    Xtr, ytr, idxtr, Xva, yva, idxva = train_val_split(X, y, idx, val_frac=0.2)
    model = build_model(seq_len=X.shape[1], n_features=X.shape[2])
    hist = model.fit(Xtr, [ytr["trend"], ytr["pulse"]],
                     validation_data=(Xva, [yva["trend"], yva["pulse"]]),
                     epochs=8, batch_size=64, verbose=2)
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    model.save(args.model)
    print(f"Saved model to {args.model}")

if __name__ == "__main__":
    main()
