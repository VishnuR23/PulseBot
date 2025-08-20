
import numpy as np
import pandas as pd

def rolling_zscore(x, window=32):
    r = x.rolling(window)
    mu = r.mean()
    sd = r.std(ddof=0)
    return (x - mu) / (sd.replace(0, np.nan))

def rsi(series, period=14):
    delta = series.diff()
    up = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    down = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def ema(x, span):
    return x.ewm(span=span, adjust=False).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # log returns & volatility
    out["ret_1"] = np.log(out["close"]).diff()
    out["ret_5"] = np.log(out["close"]).diff(5)
    out["vol_20"] = out["ret_1"].rolling(20).std().fillna(method="bfill")
    # moving averages & slopes
    out["ema_10"] = ema(out["close"], 10)
    out["ema_20"] = ema(out["close"], 20)
    out["ema_slope"] = out["ema_10"].diff()
    # rsi and zscores
    out["rsi_14"] = rsi(out["close"], 14)
    out["z_close_32"] = rolling_zscore(out["close"], 32)
    out["z_ret_32"] = rolling_zscore(out["ret_1"], 32)
    # normalize a few
    for c in ["ema_slope", "rsi_14", "z_close_32", "z_ret_32", "vol_20", "ret_1", "ret_5"]:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ret_1","ret_5","vol_20","ema_slope","rsi_14","z_close_32","z_ret_32"]
    return df[cols].astype("float32")
