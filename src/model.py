
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers as L, models as M

def build_model(seq_len: int, n_features: int, lstm_units: int = 32, conv_filters: int = 32) -> tf.keras.Model:
    inp = L.Input(shape=(seq_len, n_features), name="features")
    x = L.Conv1D(conv_filters, kernel_size=3, padding="causal", activation="relu")(inp)
    x = L.BatchNormalization()(x)
    x = L.Conv1D(conv_filters, kernel_size=5, padding="causal", activation="relu")(x)
    x = L.MaxPooling1D(pool_size=2)(x)
    x = L.LSTM(lstm_units, return_sequences=False)(x)
    x = L.Dropout(0.2)(x)
    # heads
    trend = L.Dense(32, activation="relu")(x)
    trend = L.Dense(1, activation="sigmoid", name="trend")(trend)
    pulse = L.Dense(32, activation="relu")(x)
    pulse = L.Dense(1, activation="sigmoid", name="pulse")(pulse)
    model = M.Model(inp, [trend, pulse], name="pulse_trader")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"trend":"binary_crossentropy", "pulse":"binary_crossentropy"},
        metrics={"trend":[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")],
                 "pulse":[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")]},
    )
    return model
