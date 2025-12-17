import importlib.util
import os
import sys
import numpy as np
import pandas as pd

module_path = os.path.join(os.getcwd(), '空氣品質預測.py')
spec = importlib.util.spec_from_file_location('aq_app', module_path)
app = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(app)
except Exception as e:
    print('Error importing app module:', e)
    sys.exit(1)

print('Module loaded:', app)

# Load data
# Manually load CSV and pivot using itemengname to ensure pollutant column names (some CSVs include itemid first)
raw = pd.read_csv(app.CSV_FILE_PATH)
if 'monitordate' not in [c.lower() for c in raw.columns]:
    # normalize column names to lower
    raw.columns = [c.lower() for c in raw.columns]
raw['monitordate'] = pd.to_datetime(raw['monitordate'])
raw['monitordate'] = raw['monitordate'].dt.floor('H')
raw['concentration'] = pd.to_numeric(raw['concentration'], errors='coerce')
if 'itemengname' in raw.columns:
    itemcol = 'itemengname'
elif 'itemname' in raw.columns:
    itemcol = 'itemname'
else:
    itemcol = 'itemid'
csv_df = raw.pivot_table(index='monitordate', columns=itemcol, values='concentration')
csv_df.sort_index(inplace=True)
try:
    full_index = pd.date_range(start=csv_df.index.min(), end=csv_df.index.max(), freq='H')
    csv_df = csv_df.reindex(full_index)
except Exception:
    pass
api_df = pd.DataFrame()  # skip API in demo to avoid network/SSL issues; CSV provides training data
print('CSV rows (pivoted):', csv_df.shape)
print('API rows: skipped')

data = app.preprocess_and_merge(csv_df, api_df)
print('Merged data range:', data.index.min(), 'to', data.index.max())

pollutants = [p for p in app.POLLUTANT_CONFIG.keys() if p in data.columns]
print('Available pollutants for demo:', pollutants)
print('All data columns:', list(data.columns))

results = {}
for pollutant in pollutants:
    print('\n---')
    print('Processing pollutant:', pollutant)
    aqi_t, n_steps = app.POLLUTANT_CONFIG[pollutant]
    series = data[pollutant]
    series_t = app.transform_data(series, aqi_t).dropna()
    print('Series length after transform/dropna:', len(series_t))
    if len(series_t) < n_steps + 10:
        print('SKIP: insufficient data')
        continue
    scaler = app.MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(series_t.values.reshape(-1,1))
    X, y = app.create_dataset(scaled.flatten(), look_back=n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]; y_train = y[:train_size]
    X_val = X[train_size:]; y_val = y[train_size:]
    print('Train samples:', len(y_train), 'Val samples:', len(y_val))
    model, history = app.build_and_train(X_train, y_train, epochs=5, batch_size=16)
    # predictions on train/val
    try:
        y_train_pred = model.predict(X_train.reshape((X_train.shape[0], X_train.shape[1])))
    except Exception:
        y_train_pred = model.predict(X_train)
    if len(X_val) > 0:
        try:
            y_val_pred = model.predict(X_val.reshape((X_val.shape[0], X_val.shape[1])))
        except Exception:
            y_val_pred = model.predict(X_val)
    else:
        y_val_pred = None
    y_train_true = scaler.inverse_transform(y_train.reshape(-1,1)).flatten()
    y_train_pred_inv = scaler.inverse_transform(np.array(y_train_pred).reshape(-1,1)).flatten()
    if y_val_pred is not None:
        y_val_true = scaler.inverse_transform(y_val.reshape(-1,1)).flatten()
        y_val_pred_inv = scaler.inverse_transform(np.array(y_val_pred).reshape(-1,1)).flatten()
    else:
        y_val_true = y_val_pred_inv = None
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    train_mae = mean_absolute_error(y_train_true, y_train_pred_inv)
    train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred_inv))
    train_r2 = r2_score(y_train_true, y_train_pred_inv)
    print(f'Train MAE: {train_mae:.3f}, RMSE: {train_rmse:.3f}, R2: {train_r2:.3f}')
    if y_val_pred is not None:
        val_mae = mean_absolute_error(y_val_true, y_val_pred_inv)
        val_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred_inv))
        val_r2 = r2_score(y_val_true, y_val_pred_inv)
        print(f'Val MAE: {val_mae:.3f}, RMSE: {val_rmse:.3f}, R2: {val_r2:.3f}')
    # predict next hour
    last_window = scaled[-n_steps:]
    X_last = last_window.reshape((1, n_steps, 1))
    try:
        pred_scaled = model.predict(X_last.reshape((1, n_steps)))
    except Exception:
        pred_scaled = model.predict(X_last)
    pred = scaler.inverse_transform(np.array(pred_scaled).reshape(-1,1))[0][0]
    print('Predicted next hour:', pred)
    # evaluate against standards
    std = app.evaluate_against_standards(pollutant, pred)
    print('Standards eval warning:', std.get('warning_text'))
    print('Sensitive groups:', ','.join(std.get('sensitive_groups', [])))

print('\nDemo complete')
