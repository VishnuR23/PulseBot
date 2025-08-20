
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from .features import add_indicators, feature_matrix

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def make_dataset(df: pd.DataFrame, seq_len: int = 64, horizon: int = 12, pulse_window: int = 12, pulse_k: float = 2.5):
    df = add_indicators(df)
    Xdf = feature_matrix(df)
    closes = df["close"].values.astype("float32")
    rets = np.diff(np.log(closes), prepend=np.log(closes[0]))
    # rolling vol
    vol = pd.Series(rets).rolling(pulse_window).std().fillna(method="bfill").values
    n = len(df)
    X, y_trend, y_pulse, idx = [], [], [], []
    for t in range(seq_len, n - horizon - 1):
        # input window [t-seq_len, t)
        X.append(Xdf.iloc[t-seq_len:t].values)
        # trend label: +1 if future close higher than now by horizon
        future_ret = np.log(closes[t+horizon]/closes[t])
        y_trend.append(1.0 if future_ret > 0 else 0.0)
        # pulse label: if any |ret| in next window exceeds k * current vol
        window_rets = np.abs(np.diff(np.log(closes[t:t+pulse_window+1])))
        y_pulse.append(1.0 if window_rets.max() > pulse_k*vol[t] else 0.0)
        idx.append(t)
    X = np.array(X, dtype="float32")
    y_trend = np.array(y_trend, dtype="float32")
    y_pulse = np.array(y_pulse, dtype="float32")
    return X, {"trend": y_trend, "pulse": y_pulse}, idx

def train_val_split(X, y_dict, idx, val_frac=0.2):
    n = len(X)
    split = int(n*(1-val_frac))
    return (X[:split], {k:v[:split] for k,v in y_dict.items()}, idx[:split],
            X[split:], {k:v[split:] for k,v in y_dict.items()}, idx[split:])
