import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Self-contained visualization script
# Expects ridership_model.json and final_long.csv in the same directory as this script

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "final_long.csv")
MODEL_PATH = os.path.join(HERE, "ridership_model.json")

# Read data: support CSV or Excel (auto-detect common filenames)
def find_and_load_data(folder, base_name="final_long"):
    candidates = [f"{base_name}.csv", f"{base_name}.xlsx", f"{base_name}.xls", base_name]
    for c in candidates:
        p = os.path.join(folder, c)
        if os.path.exists(p):
            try:
                if p.lower().endswith('.csv'):
                    return pd.read_csv(p)
                else:
                    # assume Excel for other matches
                    return pd.read_excel(p)
            except Exception:
                # try next candidate
                pass
    # fallback: look for any file that starts with base_name (case-insensitive)
    for f in os.listdir(folder):
        if f.lower().startswith(base_name.lower()):
            p = os.path.join(folder, f)
            try:
                if f.lower().endswith('.csv'):
                    return pd.read_csv(p)
                elif f.lower().endswith(('.xls', '.xlsx')):
                    return pd.read_excel(p)
                else:
                    # try csv then excel
                    try:
                        return pd.read_csv(p)
                    except Exception:
                        return pd.read_excel(p)
            except Exception:
                continue
    raise FileNotFoundError(f"Could not find '{base_name}.csv' or Excel equivalent in {folder}")

try:
    df = find_and_load_data(HERE, base_name="final_long")
except Exception as e:
    print(f"Error reading data file in '{HERE}': {e}")
    sys.exit(1)

# Utility: find first column matching any of the keywords (case-insensitive substring)
def find_col(df, keywords):
    keys = [k.lower() for k in keywords]
    for col in df.columns:
        cl = col.lower()
        for k in keys:
            if k in cl:
                return col
    return None

# Identify station column (common names)
station_col = find_col(df, ["station", "station_name", "stop_name", "stationid"]) or None
if station_col is None:
    # If not found, pick the first object/string dtype column
    obj_cols = df.select_dtypes(include=[object]).columns.tolist()
    if len(obj_cols) > 0:
        station_col = obj_cols[0]
    else:
        print("Could not find a station column in the CSV and there are no object columns.")
        sys.exit(1)

# Encode station column
le = LabelEncoder()
try:
    df[station_col] = df[station_col].astype(str)
    df["__station_encoded"] = le.fit_transform(df[station_col])
except Exception as e:
    print(f"Error encoding station column '{station_col}': {e}")
    sys.exit(1)

# Detect event columns (MBS, SFA, GWCC, OTHER) - create if missing
event_columns = {}
for code in ["MBS", "SFA", "GWCC", "OTHER"]:
    # try direct match first, then substring
    col = None
    if code in df.columns:
        col = code
    else:
        # look for substring
        for c in df.columns:
            if code.lower() in c.lower():
                col = c
                break
    if col is None:
        # create the column filled with zeros
        df[code] = 0.0
        col = code
    event_columns[code] = col

# Detect weather columns
temp_col = find_col(df, ["temp", "temperature", "avg temp"]) or None
precip_col = find_col(df, ["precip", "precipitation", "total precipitation"]) or None
rain10_col = find_col(df, ["rain10", "rain 10", "rain_more_than_10", "rain more than 10", "rain_more_than_10in"]) or None

# If not found, create them with sensible defaults (0 or mean if possible)
if temp_col is None:
    # try any numeric column that looks like temperature units
    temp_candidates = [c for c in df.columns if "temp" in c.lower()]
    if temp_candidates:
        temp_col = temp_candidates[0]
    else:
        df["temperature"] = 72.0
        temp_col = "temperature"

if precip_col is None:
    precip_candidates = [c for c in df.columns if "precip" in c.lower()]
    if precip_candidates:
        precip_col = precip_candidates[0]
    else:
        df["precipitation"] = 0.0
        precip_col = "precipitation"

if rain10_col is None:
    rain10_candidates = [c for c in df.columns if "rain" in c.lower() and ("10" in c.lower() or "more" in c.lower())]
    if rain10_candidates:
        rain10_col = rain10_candidates[0]
    else:
        # create
        df["rain10"] = 0.0
        rain10_col = "rain10"

# Compute mean weather values
mean_temp = float(df[temp_col].dropna().mean()) if temp_col in df.columns else 72.0
mean_precip = float(df[precip_col].dropna().mean()) if precip_col in df.columns else 0.0
mean_rain10 = float(df[rain10_col].dropna().mean()) if rain10_col in df.columns else 0.0

# Load trained XGBoost model
booster = xgb.Booster()
try:
    booster.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model '{MODEL_PATH}': {e}")
    sys.exit(1)

# Determine model's expected feature names (use exactly when available)
model_features = None
try:
    model_features = booster.feature_names
except Exception:
    model_features = None

if not model_features:
    # fallback: infer from dataframe numeric columns plus encoded station
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "__station_encoded" not in numeric_cols:
        numeric_cols.append("__station_encoded")
    for t in ["ridership", "riders", "count", "passengers"]:
        if t in numeric_cols:
            numeric_cols.remove(t)
    model_features = numeric_cols

# Helper: find a model feature by case-insensitive substring
def find_model_feature(substrings):
    subs = [s.lower() for s in (substrings if isinstance(substrings, (list, tuple)) else [substrings])]
    for mf in model_features:
        ml = mf.lower()
        for s in subs:
            if s in ml:
                return mf
    return None

# Helper to build a DataFrame row (or many rows) with specified overrides
def build_rows_for_station(station_label, n_rows=1, overrides=None):
    # Use a template of mean values per feature
    base = {}
    for fn in model_features:
        # Map model feature names by substring to appropriate default values
        fl = fn.lower()
        if "station" in fl:
            base[fn] = station_label
        elif any(x in fl for x in ["mercedes", "stadium", "mbs"]):
            base[fn] = 0.0
        elif any(x in fl for x in ["state farm", "statefarm", "arena", "sfa"]):
            base[fn] = 0.0
        elif "gwcc" in fl:
            base[fn] = 0.0
        elif "other" in fl or "americasmart" in fl or "parks" in fl:
            base[fn] = 0.0
        elif "temp" in fl:
            base[fn] = mean_temp
        elif "precip" in fl:
            base[fn] = mean_precip
        elif "rain" in fl:
            base[fn] = mean_rain10
        else:
            # try to use column mean if available in df
            if fn in df.columns and pd.api.types.is_numeric_dtype(df[fn]):
                base[fn] = float(df[fn].dropna().mean())
            else:
                base[fn] = 0.0
    # apply overrides (map keys to model feature names when possible)
    if overrides:
        for k, v in overrides.items():
            # direct match to model feature
            if k in base:
                base[k] = v
                continue
            # if key is a dataframe column name that matches a model feature
            if isinstance(k, str) and k in df.columns:
                if k in model_features:
                    base[k] = v
                    continue
                mf = find_model_feature(k)
                if mf:
                    base[mf] = v
                    continue
            # canonical mappings
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
                        mf = find_model_feature(subs)
                        if mf:
                            base[mf] = v
                            mapped = True
                            break
            if mapped:
                continue
            # try to find model feature by substring of the override key
            mf = find_model_feature(k)
            if mf:
                base[mf] = v
                continue
            # unknown override: ignore (model won't use it)
    # create DataFrame
    rows = pd.DataFrame([base.copy() for _ in range(n_rows)])
    # ensure columns order matches feature_names (plus any extras)
    cols_order = [c for c in model_features if c in rows.columns]
    # include any extra cols at end
    extra_cols = [c for c in rows.columns if c not in cols_order]
    cols_order.extend(extra_cols)
    return rows[cols_order]

# Prediction helper
def predict_df(X_df):
    # Ensure columns match feature_names order; fill missing with zeros
    for fn in model_features:
        if fn not in X_df.columns:
            X_df[fn] = 0.0
    X_df = X_df[model_features]
    dmat = xgb.DMatrix(X_df.values, feature_names=model_features)
    preds = booster.predict(dmat)
    return np.array(preds)

# Prepare list of stations (use encoded labels and original names)
stations_unique = list(zip(le.transform(le.classes_), le.classes_)) if hasattr(le, 'classes_') else [(int(v), str(v)) for v in df[["__station_encoded", station_col]].drop_duplicates().values]
# sort by encoded label
stations_unique = sorted(stations_unique, key=lambda x: x[0])
station_labels = [s[0] for s in stations_unique]
station_names = [s[1] for s in stations_unique]

# ----------------------
# (A) Mercedes-Benz Stadium lift curves
# ----------------------
att_vals = np.linspace(0, 90000, num=91)  # step 1000
plt.figure(figsize=(12, 8))
for lab, name in zip(station_labels, station_names):
    preds = []
    for a in att_vals:
        overrides = {event_columns["MBS"]: float(a)}
        # ensure other events zero
        overrides[event_columns["SFA"]] = 0.0
        overrides[event_columns["GWCC"]] = 0.0
        overrides[event_columns["OTHER"]] = 0.0
        rows = build_rows_for_station(lab, overrides=overrides)
        p = predict_df(rows)
        preds.append(float(np.mean(p)))
    plt.plot(att_vals, preds, label=str(name))

plt.title('Mercedes-Benz Stadium lift curves (one line per station)')
plt.xlabel('MBS attendance')
plt.ylabel('Predicted ridership')
plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'plot_a_mbs_lift.png'), dpi=150)

# ----------------------
# (B) Temperature sensitivity curves
# ----------------------
temps = np.linspace(30, 90, num=61)
plt.figure(figsize=(12, 8))
for lab, name in zip(station_labels, station_names):
    preds = []
    for t in temps:
        overrides = {}
        # all events zero
        overrides[event_columns["MBS"]] = 0.0
        overrides[event_columns["SFA"]] = 0.0
        overrides[event_columns["GWCC"]] = 0.0
        overrides[event_columns["OTHER"]] = 0.0
        # set temperature and use mean precip/rain10
        overrides[temp_col] = float(t)
        overrides[precip_col] = float(mean_precip)
        overrides[rain10_col] = float(mean_rain10)
        rows = build_rows_for_station(lab, overrides=overrides)
        p = predict_df(rows)
        preds.append(float(np.mean(p)))
    plt.plot(temps, preds, label=str(name))

plt.title('Temperature sensitivity curves (30-90°F)')
plt.xlabel('Temperature (°F)')
plt.ylabel('Predicted ridership')
plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'plot_b_temp_sensitivity.png'), dpi=150)

# ----------------------
# (C) Baseline ridership bar chart (all events = 0, weather = mean)
# ----------------------
baseline_preds = []
for lab, name in zip(station_labels, station_names):
    overrides = {
        event_columns["MBS"]: 0.0,
        event_columns["SFA"]: 0.0,
        event_columns["GWCC"]: 0.0,
        event_columns["OTHER"]: 0.0,
        temp_col: mean_temp,
        precip_col: mean_precip,
        rain10_col: mean_rain10,
    }
    rows = build_rows_for_station(lab, overrides=overrides)
    p = predict_df(rows)
    baseline_preds.append(float(np.mean(p)))

plt.figure(figsize=(12, 6))
x = np.arange(len(station_names))
plt.bar(x, baseline_preds)
plt.xticks(x, station_names, rotation=90)
plt.ylabel('Predicted ridership')
plt.title('Baseline ridership per station (events=0, weather=mean)')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'plot_c_baseline_bar.png'), dpi=150)

# ----------------------
# (D) "Mega Event" scenario bar chart
# ----------------------
# MBS = 70,000; SFA = 18,000; GWCC = 4,000; OTHER = 2,000; temp=72; precip=0; rain10=0
mega_values = {
    event_columns["MBS"]: 70000.0,
    event_columns["SFA"]: 18000.0,
    event_columns["GWCC"]: 4000.0,
    event_columns["OTHER"]: 2000.0,
    temp_col: 72.0,
    precip_col: 0.0,
    rain10_col: 0.0,
}
mega_preds = []
for lab, name in zip(station_labels, station_names):
    rows = build_rows_for_station(lab, overrides=mega_values)
    p = predict_df(rows)
    mega_preds.append(float(np.mean(p)))

plt.figure(figsize=(12, 6))
x = np.arange(len(station_names))
plt.bar(x, mega_preds, color='orange')
plt.xticks(x, station_names, rotation=90)
plt.ylabel('Predicted ridership')
plt.title('Mega Event scenario ridership per station')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'plot_d_mega_event_bar.png'), dpi=150)

# Show all figures
plt.show()

print("Plots saved:")
print(" - ", os.path.join(HERE, 'plot_a_mbs_lift.png'))
print(" - ", os.path.join(HERE, 'plot_b_temp_sensitivity.png'))
print(" - ", os.path.join(HERE, 'plot_c_baseline_bar.png'))
print(" - ", os.path.join(HERE, 'plot_d_mega_event_bar.png'))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# paths – adjust if your files are somewhere else
CSV_PATH = r"C:\Users\brody\Documents\final_long.csv"
MODEL_PATH = r"C:\Users\brody\Documents\Invest ATL\ridership_model.json"

# load data and model
df = pd.read_csv(CSV_PATH)

le = LabelEncoder()
df["station_id"] = le.fit_transform(df["station"])
stations = df[["station", "station_id"]].drop_duplicates().sort_values("station_id")

model = xgb.Booster()
model.load_model(MODEL_PATH)

# avg weather
weather_mean = df[["Avg temp", "total precipitation (in)", "rain more than 10in"]].mean()

mbs_range = np.linspace(0, 90000, 50)
curves = {}

for _, s in stations.iterrows():
    rows = []
    for mbs in mbs_range:
        rows.append({
            "station_id": s.station_id,
            "Mercedes-Benz Stadium Est. Attendance": mbs,
            "State Farm Arena Est. Attendance": 0,
            "GWCC Est. Attendance": 0,
            "Other Venues (AmericasMart, Parks)": 0,
            "Avg temp": weather_mean["Avg temp"],
            "total precipitation (in)": weather_mean["total precipitation (in)"],
            "rain more than 10in": weather_mean["rain more than 10in"],
        })
    batch = pd.DataFrame(rows)
    preds = model.predict(xgb.DMatrix(batch))
    curves[s.station] = preds

plt.figure(figsize=(12, 7))
for station, pred_curve in curves.items():
    plt.plot(mbs_range, pred_curve, label=station, linewidth=1)
plt.xlabel("MBS Attendance")
plt.ylabel("Predicted Ridership")
plt.title("Station Lift Curves vs. Mercedes-Benz Stadium Attendance")
plt.legend(fontsize=6)
plt.tight_layout()
plt.show()

# build design matrix X matching training
feature_cols = [
    "station_id",
    "Mercedes-Benz Stadium Est. Attendance",
    "State Farm Arena Est. Attendance",
    "GWCC Est. Attendance",
    "Other Venues (AmericasMart, Parks)",
    "Avg temp",
    "total precipitation (in)",
    "rain more than 10in"
]

X = df[feature_cols]
y_true = df["ridership"].values

dall = xgb.DMatrix(X)
y_pred = model.predict(dall)

plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, s=8, alpha=0.5)
plt.xlabel("Actual Ridership")
plt.ylabel("Predicted Ridership")
plt.title("Actual vs Predicted Ridership (All Stations/Weeks)")
max_val = max(y_true.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val])  # y=x line
plt.tight_layout()
plt.show()
