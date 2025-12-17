import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Force sklearn-only mode (no TensorFlow).
# This avoids ModuleNotFoundError in environments without TensorFlow.
TF_AVAILABLE = False
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# ==========================================
# 設定與參數
# ==========================================
STATION_NAME = "二林"
API_URL = "https://data.moenv.gov.tw/api/v2/aqx_p_223"
API_KEY = "d12e32cf-81f9-4555-a211-a614c03de02d"
CSV_FILE_PATH = "空氣品質小時值_彰化縣_二林站.csv"

# Pollutant config: pollutant -> (aqi_time_window, look_back)
POLLUTANT_CONFIG = {
    'PM2.5': (24, 30),
    'PM10': (24, 30),
    'O3': (1, 30),
    'CO': (8, 30),
    'SO2': (1, 30),
    'NO2': (1, 30)
}


@st.cache_data
def load_csv_data(filepath: str) -> pd.DataFrame:
    """讀取本地 CSV，並 pivot 成時間 x 污染物的寬格式表格"""
    df = pd.read_csv(filepath)
    # support different column namings from dataset
    col_names = {c.lower(): c for c in df.columns}
    # try to find expected columns
    lower_cols = [c.lower() for c in df.columns]
    # map likely column names
    if 'sitename' in lower_cols:
        sitename_col = [c for c in df.columns if c.lower() == 'sitename'][0]
        df = df[df[sitename_col] == STATION_NAME].copy()
    # expected columns: itemengname / item, monitordate, concentration
    date_col = next((c for c in df.columns if c.lower().startswith('monitor') or c.lower().startswith('date')), None)
    item_col = next((c for c in df.columns if 'item' in c.lower() or 'pollut' in c.lower()), None)
    value_col = next((c for c in df.columns if 'concent' in c.lower() or 'value' in c.lower()), None)
    if date_col is None or item_col is None or value_col is None:
        st.warning('CSV 欄位名稱找不到預期的 monitordate/itemengname/concentration，請確認檔案格式')
        return pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df_pivot = df.pivot_table(index=date_col, columns=item_col, values=value_col)
    df_pivot.sort_index(inplace=True)
    return df_pivot


@st.cache_data
def fetch_api_data() -> pd.DataFrame:
    """從環保署 API 獲取即時資料並轉為 pivot 格式"""
    params = {
        'api_key': API_KEY,
        'limit': 1000,
        'sort': 'MonitorDate desc',
        'format': 'json',
        'filters': f"SiteName,EQ,{STATION_NAME}"
    }
    try:
        r = requests.get(API_URL, params=params, timeout=10)
        data = r.json()
    except requests.exceptions.SSLError as e:
        # SSL 驗證失敗：嘗試在不驗證憑證的情況下重試（不安全）
        try:
            urllib3.disable_warnings(InsecureRequestWarning)
        except Exception:
            pass
        try:
            r = requests.get(API_URL, params=params, timeout=10, verify=False)
            data = r.json()
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if 'records' not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data['records'])
    # normalize columns
    df.columns = [c.lower() for c in df.columns]
    # expected lower: sitename, itemengname, monitordate, concentration
    if 'monitordate' not in df.columns or 'itemengname' not in df.columns or 'concentration' not in df.columns:
        # try alternative keys
        df = df.rename(columns={k: k.lower() for k in df.columns})
    df['monitordate'] = pd.to_datetime(df['monitordate'])
    df['concentration'] = pd.to_numeric(df['concentration'], errors='coerce')
    df_pivot = df.pivot_table(index='monitordate', columns='itemengname', values='concentration')
    df_pivot.sort_index(inplace=True)
    return df_pivot


def preprocess_and_merge(csv_df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
    """合併 CSV 與 API，插值並刪除長度超過 4 的缺失段落"""
    if not csv_df.empty and not api_df.empty:
        full = pd.concat([csv_df, api_df])
        full = full[~full.index.duplicated(keep='last')]
    elif not csv_df.empty:
        full = csv_df
    else:
        full = api_df
    full = full.sort_index()
    # 線性插值，limit=4
    full_imputed = full.interpolate(method='linear', limit=4)
    # 注意：不要對整個 DataFrame 做 dropna()，因為不同時間點可能只缺少部分污染物。
    # 我們改為在每個污染物訓練前針對該欄位做 dropna()，以保留更多可用時間點。
    return full_imputed


def transform_data(series: pd.Series, aqi_t: int) -> pd.Series:
    if aqi_t > 1:
        return series.rolling(window=aqi_t, min_periods=int(aqi_t * 0.7)).mean()
    return series


def create_dataset(arr: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(len(arr) - look_back):
        X.append(arr[i:(i + look_back)])
        y.append(arr[i + look_back])
    return np.array(X), np.array(y)


def build_and_train(X_train, y_train, epochs=10, batch_size=16):
    """
    Train a GradientBoostingRegressor on flattened time-window features (sklearn-only mode).
    Returns (model, None).
    """
    # Flatten the time-series windows for sklearn
    X_flat = X_train.reshape((X_train.shape[0], X_train.shape[1]))
    y_flat = y_train.ravel()
    # Use a reasonably strong ensemble regressor as fallback
    model = GradientBoostingRegressor(n_estimators=200)
    model.fit(X_flat, y_flat)
    return model, None


def AQI_category(val, pollutant_idx):
    # pollutant_idx: 0=CO,1=PM25,2=PM10,3=NO2,4=SO2,5=O3,6=O3_8h
    # This implements the same thresholds as the Aiot_Project notebook
    v = float(val)
    if pollutant_idx == 0:  # CO
        if v <= 4.4: return 0
        if v <= 9.4: return 1
        if v <= 12.4: return 2
        if v <= 15.4: return 3
        if v <= 30.4: return 4
        if v <= 40.4: return 5
        return 6
    if pollutant_idx == 1:  # PM2.5
        if v < 15.5: return 0
        if v < 35.5: return 1
        if v < 54.5: return 2
        if v < 150.5: return 3
        if v < 250.5: return 4
        if v <= 350.4: return 5
        return 6
    if pollutant_idx == 2:  # PM10
        if v < 51: return 0
        if v < 101: return 1
        if v < 255: return 2
        if v < 355: return 3
        if v < 425: return 4
        if v <= 504: return 5
        return 6
    if pollutant_idx == 3:  # NO2
        if v <= 30: return 0
        if v <= 100: return 1
        if v <= 360: return 2
        if v <= 649: return 3
        if v <= 1249: return 4
        if v <= 1649: return 5
        return 6
    if pollutant_idx == 4:  # SO2
        if v <= 20: return 0
        if v <= 75: return 1
        if v <= 185: return 2
        if v <= 304: return 3
        if v <= 604: return 4
        if v <= 804: return 5
        return 6
    if pollutant_idx == 5:  # O3 (1h)
        if v <= 124: return 0
        if v <= 164: return 2
        if v <= 204: return 3
        if v <= 404: return 4
        if v <= 504: return 5
        return 6
    if pollutant_idx == 6:  # O3 8h
        if v < 55: return 0
        if v < 71: return 1
        if v < 86: return 2
        if v < 106: return 3
        if v <= 200: return 4
        return 6


def plot_series(x, real_vals, pred_vals, pollutant):
    """Plot real and predicted values against the same x (datetime or numeric)."""
    fig, ax = plt.subplots(figsize=(8, 3))
        # real: pandas Series (indexed by timestamp)
        # pred: tuple (pred_time, pred_value) or array of predicted values
        if hasattr(real_vals, 'index'):
            ax.plot(real_vals.index, real_vals.values, label='Real', color='red')
        else:
            ax.plot(real_vals, label='Real', color='red')

        # If pred is a tuple (time, value), plot as a scatter point
        if isinstance(pred_vals, tuple) and len(pred_vals) == 2:
            pred_time, pred_val = pred_vals
            try:
                ax.scatter([pred_time], [pred_val], color='blue', label='Predicted (next)')
            except Exception:
                ax.plot([len(real_vals)], [pred_val], marker='o', color='blue', label='Predicted (next)')
        else:
            # pred is an array of values aligned after real (e.g., appended)
            ax.plot(pred_vals, label='Predicted', color='blue')

        ax.set_title(f"{pollutant} Prediction")
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)


def app():
    st.title('空氣品質訓練與預測 (Streamlit)')

    st.sidebar.header('設定')
    # Load data
    if not os.path.exists(CSV_FILE_PATH):
        st.error(f"找不到 CSV：{CSV_FILE_PATH}，請將 CSV 放在此目錄下")
        return

    csv_df = load_csv_data(CSV_FILE_PATH)
    api_df = fetch_api_data()
    data = preprocess_and_merge(csv_df, api_df)

    if data.empty:
        st.warning('合併後沒有可用資料，請檢查 CSV 與 API 欄位')
        return

    st.sidebar.write(f"資料時間範圍： {data.index.min()} — {data.index.max()}")

    pollutants = [p for p in POLLUTANT_CONFIG.keys() if p in data.columns]
    if not pollutants:
        st.warning('資料中沒有可用污染物欄位')
        return

    selected = st.sidebar.multiselect('選擇要訓練/預測的污染物', pollutants, default=pollutants)
    epochs = st.sidebar.number_input('Epochs', min_value=1, max_value=200, value=10)
    batch_size = st.sidebar.number_input('Batch size', min_value=1, max_value=128, value=16)
    force_train = st.sidebar.checkbox('允許強制訓練（即使資料不足）', value=False)

    # 顯示每個污染物的可用資料筆數與需求（方便 debug 為何會被跳過）
    st.sidebar.markdown('### 資料概況（每個污染物）')
    for p in pollutants:
        aqi_t, n_steps = POLLUTANT_CONFIG[p]
        series = data[p]
        series_t = transform_data(series, aqi_t).dropna()
        st.sidebar.write(f"{p}: 原始 {len(series)} 筆，移動平均後 {len(series_t)} 筆（需求 >= {n_steps + 10}）")

    # 將按鈕放在側欄以確保能看到
    train_btn = st.sidebar.button('訓練選取的模型')
    predict_btn = st.sidebar.button('進行預測並顯示結果')

    models = {}
    predictions = {}

    if train_btn:
        with st.spinner('模型訓練中，請稍候...'):
            for pollutant in selected:
                aqi_t, n_steps = POLLUTANT_CONFIG[pollutant]
                series = data[pollutant]
                series_t = transform_data(series, aqi_t).dropna()
                if len(series_t) < n_steps + 10:
                    st.write(f"跳過 {pollutant}：資料不足")
                    continue
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(series_t.values.reshape(-1, 1))
                X, y = create_dataset(scaled.flatten(), look_back=n_steps)
                X = X.reshape((X.shape[0], X.shape[1], 1))
                model, history = build_and_train(X, y, epochs=epochs, batch_size=batch_size)
                models[pollutant] = (model, scaler, series_t, n_steps)
                # save model to file for reuse (sklearn joblib .pkl)
                joblib.dump(model, f"lstm_model({pollutant}).pkl")
                st.write(f"{pollutant} 模型訓練完成（sklearn），已儲存 lstm_model({pollutant}).pkl")

    if predict_btn:
        # try to load any pre-trained models saved in run
        for pollutant in selected:
            model_file = f"lstm_model({pollutant}).pkl"
            aqi_t, n_steps = POLLUTANT_CONFIG[pollutant]
            series = data[pollutant]
            series_t = transform_data(series, aqi_t).dropna()
            if len(series_t) < n_steps + 1:
                st.write(f"跳過 {pollutant}：資料不足")
                continue
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(series_t.values.reshape(-1, 1))
            # prepare last window
            last_window = scaled[-n_steps:]
            X_last = last_window.reshape((1, n_steps, 1))

            # if trained in this session, use it; otherwise try loading saved model
            if pollutant in models:
                model = models[pollutant][0]
            else:
                try:
                    model = joblib.load(model_file)
                except Exception:
                    st.write(f"{pollutant} 無可用模型 (請先訓練或上傳模型檔案)")
                    continue

            # sklearn model expects flattened input
            X_last_flat = X_last.reshape((1, n_steps))
            pred_scaled = model.predict(X_last_flat)
            # pred_scaled is on scaled space because we trained on scaled values
            pred = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1))[0][0]
            current = series_t.iloc[-1]
            time_last = series_t.index[-1]
            predictions[pollutant] = (time_last, current, pred)

            st.subheader(f"{pollutant} 預測")
            st.write(f"最新時間: {time_last}  當前數值: {current:.2f}  預測下一值: {pred:.2f}")
            # plot: show last 100 points vs predicted last point appended
            real_plot = series_t[-100:]
            pred_plot = np.concatenate([real_plot.values, [pred]])
            plot_series(real_plot.index.union(pd.Index([time_last])), pred_plot, pollutant)

        # AQI aggregation similar to notebook: choose pollutant with max AQI
        if predictions:
            idx_map = {'CO':0,'PM2.5':1,'PM10':2,'NO2':3,'SO2':4,'O3':5}
            aqi_scores = {}
            for i, (p, (t,c,pred)) in enumerate(predictions.items()):
                p_idx = idx_map.get(p, None)
                if p_idx is None:
                    continue
                aqi_real = AQI_category(c, p_idx)
                aqi_pre = AQI_category(pred, p_idx)
                aqi_scores[p] = (aqi_real, aqi_pre)
            # display aggregated
            st.write('---')
            st.header('AQI 分類比較')
            for p, (r, pr) in aqi_scores.items():
                st.write(f"{p}: 目前 AQI 類別 {r}  / 預測 AQI 類別 {pr}")


if __name__ == '__main__':
    app()