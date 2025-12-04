import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from viz_common import find_and_load_data, load_model, prepare_data_and_encoder, compute_mean_weather, get_model_features

HERE = os.path.dirname(os.path.abspath(__file__))

def find_target_column(df):
    for candidate in ["ridership", "riders", "count", "passengers", "y"]:
        for col in df.columns:
            if candidate in col.lower():
                return col
    return None

def find_week_column(df):
    for col in df.columns:
        if 'week' in col.lower():
            return col
    # try to find a date column
    for col in df.columns:
        if any(x in col.lower() for x in ['date', 'datetime', 'day']):
            return col
    return None

def map_feature_to_series(mf, df, station_col, mean_temp, mean_precip, mean_rain10, temp_col, precip_col, rain10_col):
    ml = mf.lower()
    # station mapping
    if 'station' in ml:
        return df['__station_encoded']
    # direct column name
    if mf in df.columns:
        return df[mf]
    # substring match to df columns
    for c in df.columns:
        if c.lower() == mf.lower():
            return df[c]
    for c in df.columns:
        if mf.lower() in c.lower() or c.lower() in mf.lower():
            return df[c]
    # known weather mappings
    if 'temp' in ml:
        return df[temp_col] if temp_col in df.columns else pd.Series(mean_temp, index=df.index)
    if 'precip' in ml:
        return df[precip_col] if precip_col in df.columns else pd.Series(mean_precip, index=df.index)
    if 'rain' in ml:
        return df[rain10_col] if rain10_col in df.columns else pd.Series(mean_rain10, index=df.index)
    # attendance / event columns: try common names
    for keyword in ['mercedes', 'stadium', 'mbs', 'state farm', 'statefarm', 'arena', 'sfa', 'gwcc', 'other', 'americasmart']:
        if keyword in ml:
            for c in df.columns:
                if keyword in c.lower():
                    return df[c]
    # default to zeros
    return pd.Series(0.0, index=df.index)

def main():
    try:
        df = find_and_load_data(HERE, base_name='final_long')
    except Exception as e:
        print(f"Could not load data file: {e}")
        sys.exit(1)

    # Prepare station encoding
    try:
        df, station_col, le = prepare_data_and_encoder(df)
    except Exception as e:
        print(f"Error preparing station encoder: {e}")
        sys.exit(1)

    # Compute mean weather and canonical column names
    temp_col, precip_col, rain10_col, mean_temp, mean_precip, mean_rain10 = compute_mean_weather(df)

    # Load model
    try:
        booster = load_model(HERE)
    except Exception as e:
        print(f"Could not load model: {e}")
        sys.exit(1)

    model_features = get_model_features(booster)
    if not model_features:
        print("Model feature names not available; cannot proceed.")
        sys.exit(1)

    # Build X dataframe with columns in model_features order
    X_parts = {}
    for mf in model_features:
        X_parts[mf] = map_feature_to_series(mf, df, station_col, mean_temp, mean_precip, mean_rain10, temp_col, precip_col, rain10_col)
    X_df = pd.DataFrame(X_parts, index=df.index)[model_features]

    # Predict
    try:
        dmat = xgb.DMatrix(X_df.values, feature_names=model_features)
        y_pred = booster.predict(dmat)
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)

    # Find target column
    target_col = find_target_column(df)
    if target_col is None:
        print("Could not find target column (ridership). Add a column named 'ridership' or similar.")
        sys.exit(1)

    # Build residuals
    df_res = df.copy()
    df_res['pred'] = y_pred
    df_res['residual'] = df_res[target_col] - df_res['pred']

    # Determine week column; if it's a date column, convert to week numbers
    week_col = find_week_column(df_res)
    if week_col is None:
        print("No week or date column found to group by week. Please add a 'week' column or a date column.")
        sys.exit(1)

    # If the found week_col is a date, convert to ISO week number
    if 'week' not in week_col.lower():
        try:
            df_res['__week'] = pd.to_datetime(df_res[week_col]).dt.isocalendar().week
            week_col_use = '__week'
        except Exception:
            # fallback to string representation
            df_res['__week'] = df_res[week_col].astype(str)
            week_col_use = '__week'
    else:
        week_col_use = week_col

    # Pivot to get residual heatmap
    try:
        heat = df_res.pivot_table(index=station_col, columns=week_col_use, values='residual', aggfunc='mean')
    except Exception as e:
        print(f"Pivot table failed: {e}")
        sys.exit(1)

    # Prepare heatmap values and mask NaNs
    heat_vals = heat.values
    # center colormap at zero
    finite = heat_vals[np.isfinite(heat_vals)]
    if finite.size == 0:
        print("No finite residuals to plot.")
        sys.exit(1)
    vmax = np.nanmax(np.abs(finite))

    plt.figure(figsize=(14, 6))
    im = plt.imshow(heat_vals, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label='Residual (Actual - Predicted)')
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks([])
    plt.title('Residual heatmap by station and week')
    plt.tight_layout()
    outpath = os.path.join(HERE, 'residual_heatmap_by_station_week.png')
    plt.savefig(outpath, dpi=150)
    plt.show()
    print('Saved heatmap to', outpath)


if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# paths – adjust if needed
CSV_PATH = r"C:\Users\brody\Documents\final_long.csv"
MODEL_PATH = r"C:\Users\brody\Documents\Invest ATL\ridership_model.json"

# load data
df = pd.read_csv(CSV_PATH)

le = LabelEncoder()
df["station_id"] = le.fit_transform(df["station"])
stations = df[["station", "station_id"]].drop_duplicates().sort_values("station_id")

model = xgb.Booster()
model.load_model(MODEL_PATH)

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
