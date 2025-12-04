import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))

def find_and_load_data(folder, base_name="final_long"):
    candidates = [f"{base_name}.csv", f"{base_name}.xlsx", f"{base_name}.xls", base_name]
    for c in candidates:
        p = os.path.join(folder, c)
        if os.path.exists(p):
            try:
                if p.lower().endswith('.csv'):
                    return pd.read_csv(p)
                else:
                    return pd.read_excel(p)
            except Exception:
                pass
    for f in os.listdir(folder):
        if f.lower().startswith(base_name.lower()):
            p = os.path.join(folder, f)
            try:
                if f.lower().endswith('.csv'):
                    return pd.read_csv(p)
                elif f.lower().endswith(('.xls', '.xlsx')):
                    return pd.read_excel(p)
                else:
                    try:
                        return pd.read_csv(p)
                    except Exception:
                        return pd.read_excel(p)
            except Exception:
                continue
    raise FileNotFoundError(f"Could not find '{base_name}.csv' or Excel equivalent in {folder}")

def load_model(folder, model_name='ridership_model.json'):
    p = os.path.join(folder, model_name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model file not found: {p}")
    booster = xgb.Booster()
    booster.load_model(p)
    return booster

def detect_station_column(df):
    def find_col(df, keywords):
        keys = [k.lower() for k in keywords]
        for col in df.columns:
            cl = col.lower()
            for k in keys:
                if k in cl:
                    return col
        return None
    station_col = find_col(df, ["station", "station_name", "stop_name", "stationid"]) or None
    if station_col is None:
        obj_cols = df.select_dtypes(include=[object]).columns.tolist()
        if len(obj_cols) > 0:
            station_col = obj_cols[0]
        else:
            raise ValueError("Could not find a station column in the data")
    return station_col

def prepare_data_and_encoder(df):
    station_col = detect_station_column(df)
    df[station_col] = df[station_col].astype(str)
    le = LabelEncoder()
    df['__station_encoded'] = le.fit_transform(df[station_col])
    return df, station_col, le

def compute_mean_weather(df):
    def find_col(df, keywords):
        keys = [k.lower() for k in keywords]
        for col in df.columns:
            cl = col.lower()
            for k in keys:
                if k in cl:
                    return col
        return None
    temp_col = find_col(df, ["temp", "temperature", "avg temp"]) or 'temperature'
    precip_col = find_col(df, ["precip", "precipitation", "total precipitation"]) or 'precipitation'
    rain10_col = find_col(df, ["rain10", "rain 10", "rain_more_than_10", "rain more than 10", "rain_more_than_10in"]) or 'rain10'
    if temp_col not in df.columns:
        df[temp_col] = 72.0
    if precip_col not in df.columns:
        df[precip_col] = 0.0
    if rain10_col not in df.columns:
        df[rain10_col] = 0.0
    return temp_col, precip_col, rain10_col, float(df[temp_col].dropna().mean()), float(df[precip_col].dropna().mean()), float(df[rain10_col].dropna().mean())

def get_model_features(booster):
    try:
        mf = booster.feature_names
    except Exception:
        mf = None
    return mf

def find_model_feature(model_features, substrings):
    if not model_features:
        return None
    subs = [s.lower() for s in (substrings if isinstance(substrings, (list, tuple)) else [substrings])]
    for mf in model_features:
        ml = mf.lower()
        for s in subs:
            if s in ml:
                return mf
    return None

def build_rows_for_station(df, model_features, le, station_label, temp_col, precip_col, rain10_col, mean_temp, mean_precip, mean_rain10, overrides=None):
    base = {}
    for fn in model_features:
        fl = fn.lower()
        if 'station' in fl:
            base[fn] = station_label
        elif any(x in fl for x in ['mercedes', 'stadium', 'mbs']):
            base[fn] = 0.0
        elif any(x in fl for x in ['state farm', 'statefarm', 'arena', 'sfa']):
            base[fn] = 0.0
        elif 'gwcc' in fl:
            base[fn] = 0.0
        elif any(x in fl for x in ['other', 'americasmart', 'parks']):
            base[fn] = 0.0
        elif 'temp' in fl:
            base[fn] = mean_temp
        elif 'precip' in fl:
            base[fn] = mean_precip
        elif 'rain' in fl:
            base[fn] = mean_rain10
        else:
            if fn in df.columns and pd.api.types.is_numeric_dtype(df[fn]):
                base[fn] = float(df[fn].dropna().mean())
            else:
                base[fn] = 0.0
    if overrides:
        for k, v in overrides.items():
            if k in base:
                base[k] = v
                continue
            if isinstance(k, str) and k in df.columns:
                if k in model_features:
                    base[k] = v
                    continue
                mf = find_model_feature(model_features, k)
                if mf:
                    base[mf] = v
                    continue
            canon_map = {
                'MBS': ['mercedes', 'mbs', 'stadium'],
                'SFA': ['state farm', 'statefarm', 'arena', 'sfa'],
                'GWCC': ['gwcc'],
                'OTHER': ['other', 'americasmart', 'parks'],
                'temperature': ['temp', 'temperature', 'avg temp'],
                'precip': ['precip', 'precipitation', 'total precipitation'],
                'rain10': ['rain', 'rain more', 'rain10', 'more than 10', 'rain more than 10in']
            }
            mapped = False
            if isinstance(k, str):
                kl = k.lower()
                for subs in canon_map.values():
                    if any(s in kl for s in subs):
                        mf = find_model_feature(model_features, subs)
                        if mf:
                            base[mf] = v
                            mapped = True
                            break
            if mapped:
                continue
            mf = find_model_feature(model_features, k)
            if mf:
                base[mf] = v
                continue
    rows = pd.DataFrame([base.copy()])
    cols_order = [c for c in model_features if c in rows.columns]
    extra_cols = [c for c in rows.columns if c not in cols_order]
    cols_order.extend(extra_cols)
    return rows[cols_order]

def predict_df(booster, model_features, X_df):
    for fn in model_features:
        if fn not in X_df.columns:
            X_df[fn] = 0.0
    X_df = X_df[model_features]
    dmat = xgb.DMatrix(X_df.values, feature_names=model_features)
    preds = booster.predict(dmat)
    return np.array(preds)

def get_station_list_from_df(df, station_col, le):
    stations_unique = list(zip(le.transform(le.classes_), le.classes_)) if hasattr(le, 'classes_') else [(int(v), str(v)) for v in df[["__station_encoded", station_col]].drop_duplicates().values]
    stations_unique = sorted(stations_unique, key=lambda x: x[0])
    station_labels = [s[0] for s in stations_unique]
    station_names = [s[1] for s in stations_unique]
    return station_labels, station_names
