import os
import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DATA_PATH = os.getenv("DATA_PATH", "data/housing.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_FEATURES = [
    'LGFA','LAGE','FL','LCENTRAL','E','S','W','N','NE','SE','SW','NW','DM','GSI'
]

TARGET = 'LRP'
DATE_COL = 'Date'

def safe_log(x: pd.Series, eps: float = 1e-9) -> pd.Series:
    return np.log(x.clip(lower=eps))

def ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET not in df.columns:
        price_candidates = [c for c in df.columns if c.lower() in {"price","real_price","sale_price"}]
        if price_candidates:
            df[TARGET] = safe_log(df[price_candidates[0]].astype(float))
            print(f"[Info] Created {TARGET} from '{price_candidates[0]}' using natural log.")
        else:
            raise ValueError(f"Dataset must contain '{TARGET}' or a price column (Price/Real_Price). Columns: {list(df.columns)}")
    return df

def maybe_engineer_logs(df: pd.DataFrame) -> pd.DataFrame:
    if 'LGFA' not in df.columns:
        for c in df.columns:
            if c.lower() in {"gfa","area","gross_floor_area","size_sqft","sqft"}:
                df['LGFA'] = safe_log(df[c].astype(float))
                print(f"[Info] Created LGFA from '{c}' (log).")
                break
    if 'LAGE' not in df.columns:
        for c in df.columns:
            if c.lower() in {"age","building_age","years_old"}:
                df['LAGE'] = safe_log(df[c].astype(float) + 1.0)
                print(f"[Info] Created LAGE from '{c}' (log(age+1)).")
                break
    return df

def complete_directionals(df: pd.DataFrame) -> pd.DataFrame:
    directional = ['E','S','W','N','NE','SE','SW','NW']
    for d in directional:
        if d not in df.columns:
            df[d] = 0
    return df

def sanitize_features(df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = complete_directionals(df)
    features = [f for f in feature_list if f in df.columns]
    missing = sorted(list(set(feature_list) - set(features)))
    if missing:
        print(f"[Warn] Missing features skipped: {missing}")
    return df, features

def compute_correlations(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    corr = df[cols].corr()
    corr.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix.csv'))
    plt.figure(figsize=(10,8))
    plt.imshow(corr, interpolation='nearest')
    plt.title('Correlation Matrix')
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=200)
    plt.close()
    return corr

@dataclass
class DataSplits:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: Optional[StandardScaler]
    feature_names: List[str]

def prepare_tabular(df: pd.DataFrame, feature_list: List[str], test_size: float = 0.2) -> DataSplits:
    X = df[feature_list].astype(float).values
    y = df[TARGET].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return DataSplits(X_train, X_test, y_train, y_test, scaler, feature_list)

def evaluate_and_plot(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=10)
    min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v])
    plt.xlabel('Actual LRP')
    plt.ylabel('Predicted LRP')
    plt.title(f'{name}: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_pred_vs_actual.png'), dpi=200)
    plt.close()

    residuals = y_true - y_pred
    plt.figure(figsize=(7,4))
    plt.scatter(y_pred, residuals, s=8)
    plt.axhline(0)
    plt.xlabel('Predicted LRP')
    plt.ylabel('Residual (Actual - Pred)')
    plt.title(f'{name}: Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_residuals.png'), dpi=200)
    plt.close()

    return {
        'model': name,
        'R2': r2,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def train_svr_linear(splits: DataSplits) -> Dict:
    model = SVR(kernel='linear', C=1.0)
    model.fit(splits.X_train, splits.y_train)
    y_pred = model.predict(splits.X_test)
    return evaluate_and_plot('SVR_linear', splits.y_test, y_pred)

def train_sgd(splits: DataSplits) -> Dict:
    model = SGDRegressor(random_state=RANDOM_SEED, max_iter=2000, tol=1e-4)
    model.fit(splits.X_train, splits.y_train)
    y_pred = model.predict(splits.X_test)
    return evaluate_and_plot('SGDRegressor', splits.y_test, y_pred)

def train_rf(splits: DataSplits) -> Dict:
    model = RandomForestRegressor(
        n_estimators=400, random_state=RANDOM_SEED, n_jobs=-1, max_depth=None
    )
    model.fit(splits.X_train, splits.y_train)
    y_pred = model.predict(splits.X_test)
    metrics = evaluate_and_plot('RandomForest', splits.y_test, y_pred)

    importances = model.feature_importances_
    plt.figure(figsize=(8,4))
    order = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[order])
    plt.xticks(range(len(importances)), [splits.feature_names[i] for i in order], rotation=60)
    plt.title('RandomForest: Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'RandomForest_feature_importances.png'), dpi=200)
    plt.close()

    return metrics

def train_gbm(splits: DataSplits) -> Dict:
    model = GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=500, learning_rate=0.05, max_depth=3)
    model.fit(splits.X_train, splits.y_train)
    y_pred = model.predict(splits.X_test)
    metrics = evaluate_and_plot('GradientBoosting', splits.y_test, y_pred)
    return metrics

def build_ann(input_dim: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

def build_cnn1d(n_features: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

def build_lstm(input_steps: int, n_features: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_steps, n_features)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

def train_deep_model(name: str, model: tf.keras.Model, X_train, y_train, X_val, y_val) -> Tuple[tf.keras.Model, Dict]:
    es = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300, batch_size=64, verbose=0,
        callbacks=[es, rlrop]
    )

    plt.figure(figsize=(7,4))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'{name} Training')
    plt.legend(['train','val'])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'deep_{name}_training.png'), dpi=200)
    plt.close()

    y_pred = model.predict(X_val, verbose=0).squeeze()
    metrics = evaluate_and_plot(f'DEEP_{name}', y_val, y_pred)
    return model, metrics

def make_sequences_from_time(df: pd.DataFrame, features: List[str], target: str, date_col: str, steps: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    d = df[[date_col] + features + [target]].dropna().copy()
    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    d = d.sort_values(date_col)

    X_raw = d[features].astype(float).values
    y_raw = d[target].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_seq = [], []
    for i in range(len(d) - steps):
        X_seq.append(X_scaled[i:i+steps, :])
        y_seq.append(y_raw[i+steps])
    return np.array(X_seq), np.array(y_seq)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DATA_PATH, help='Path to CSV file')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--lstm_steps', type=int, default=12)
    args = parser.parse_args([])

    df = pd.read_csv(args.data)
    df = ensure_target(df)
    df = maybe_engineer_logs(df)
    df, features = sanitize_features(df, DEFAULT_FEATURES)

    corr = compute_correlations(df, [TARGET] + features)
    print(corr[[TARGET]].sort_values(by=TARGET, ascending=False).head(10))

    splits = prepare_tabular(df, features, test_size=args.test_size)

    records = []
    records.append(train_svr_linear(splits))
    records.append(train_sgd(splits))
    records.append(train_rf(splits))
    records.append(train_gbm(splits))

    X_tr, X_val, y_tr, y_val = train_test_split(
        splits.X_train, splits.y_train, test_size=0.2, random_state=RANDOM_SEED
    )

    ann = build_ann(input_dim=X_tr.shape[1])
    _, m_ann = train_deep_model('ANN', ann, X_tr, y_tr, X_val, y_val)
    records.append(m_ann)
    Xtr_cnn = np.expand_dims(X_tr, axis=-1)
    Xval_cnn = np.expand_dims(X_val, axis=-1)
    cnn = build_cnn1d(n_features=X_tr.shape[1])
    _, m_cnn = train_deep_model('CNN1D', cnn, Xtr_cnn, y_tr, Xval_cnn, y_val)
    records.append(m_cnn)
    if DATE_COL in df.columns:
        try:
            X_seq, y_seq = make_sequences_from_time(df, features, TARGET, DATE_COL, steps=args.lstm_steps)
            if len(X_seq) > 100:
                Xs_tr, Xs_val, ys_tr, ys_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=RANDOM_SEED)
                lstm = build_lstm(input_steps=Xs_tr.shape[1], n_features=Xs_tr.shape[2])
                _, m_lstm = train_deep_model('LSTM', lstm, Xs_tr, ys_tr, Xs_val, ys_val)
                records.append(m_lstm)
            else:
                print(f"[Warn] Not enough sequential samples for LSTM (have {len(X_seq)}). Skipping.")
        except Exception as e:
            print(f"[Warn] LSTM training skipped due to error: {e}")
    else:
        print("[Info] No Date column found; skipping LSTM.")

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics_summary.csv'), index=False)
    print('\n=== Metrics Summary ===')
    print(metrics_df.sort_values('R2', ascending=False))

    with open(os.path.join(OUTPUT_DIR, 'features_used.json'), 'w') as f:
        json.dump(features, f, indent=2)


if __name__ == "__main__":
    main()
