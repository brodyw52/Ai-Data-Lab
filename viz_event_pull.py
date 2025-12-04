import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Single-file script to reproduce the Event Demand Pull Map and Baseline vs Mega-Event charts
# It is defensive about file locations and column names.

HERE = os.path.dirname(os.path.abspath(__file__))

# Candidate file locations (common places based on your workspace)
CSV_CANDIDATES = [
    os.path.join(HERE, 'final_long.csv'),
    os.path.join(HERE, '..', 'final_long.csv'),
    os.path.join(os.path.expanduser('~'), 'Documents', 'final_long.csv'),
    os.path.join(HERE, 'final_long.xlsx'),
]
MODEL_CANDIDATES = [
    os.path.join(HERE, 'ridership_model.json'),
    os.path.join(HERE, '..', 'ridership_model.json'),
    os.path.join(os.path.expanduser('~'), 'Documents', 'ridership_model.json'),
]


def find_file(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def load_data():
    p = find_file(CSV_CANDIDATES)
    if p is None:
        print('Could not find final_long.csv in expected locations. Looked in:', CSV_CANDIDATES)
        sys.exit(1)
    if p.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(p)
    return pd.read_csv(p)


def load_model():
    p = find_file(MODEL_CANDIDATES)
    if p is None:
        print('Could not find ridership_model.json in expected locations. Looked in:', MODEL_CANDIDATES)
        sys.exit(1)
    booster = xgb.Booster()
    booster.load_model(p)
    return booster


def find_col(df, keywords):
    keys = [k.lower() for k in keywords]
    for col in df.columns:
        cl = col.lower()
        for k in keys:
            if k in cl:
                return col
    return None


def map_feature_columns(df, desired):
    # Return a list of actual column names in df that map to desired feature names
    mapped = []
    for feat in desired:
        if feat in df.columns:
            mapped.append(feat)
            continue
        # try substring matching
        fmatch = find_col(df, [feat])
        if fmatch:
            mapped.append(fmatch)
            continue
        # try some tokens from the desired name
        tokens = [t.strip() for t in feat.replace('(', ' ').replace(')', ' ').replace(',', ' ').split() if t.strip()]
        fmatch = None
        for t in tokens[:3]:
            fmatch = find_col(df, [t])
            if fmatch:
                break
        if fmatch:
            mapped.append(fmatch)
        else:
            # not found, use a synthetic zero column name (we'll fill with zeros later)
            mapped.append(feat)
    return mapped


def predict_rows(model, feature_cols, rows: pd.DataFrame) -> np.ndarray:
    # Ensure all feature_cols present in rows; fill missing with zeros
    X = pd.DataFrame(columns=feature_cols)
    for c in feature_cols:
        if c in rows.columns:
            X[c] = rows[c]
        else:
            X[c] = 0.0
    # create DMatrix with explicit feature names to match model
    dmat = xgb.DMatrix(X.values, feature_names=feature_cols)
    return model.predict(dmat)


def main():
    df = load_data()
    model = load_model()

    # find station column
    station_col = find_col(df, ['station', 'station_name', 'stop_name'])
    if station_col is None:
        print('Could not find station column (expected "station").')
        sys.exit(1)

    # ensure station strings
    df[station_col] = df[station_col].astype(str)

    le = LabelEncoder()
    df['station_id'] = le.fit_transform(df[station_col])

    stations = df[[station_col, 'station_id']].drop_duplicates().sort_values('station_id')

    # canonical feature names expected by the model
    feature_cols_desired = [
        'station_id',
        'Mercedes-Benz Stadium Est. Attendance',
        'State Farm Arena Est. Attendance',
        'GWCC Est. Attendance',
        'Other Venues (AmericasMart, Parks)',
        'Avg temp',
        'total precipitation (in)',
        'rain more than 10in',
    ]

    # Map to actual df columns (or keep the desired name if missing - we'll fill zeros)
    mapped = map_feature_columns(df, feature_cols_desired)
    desired_to_actual = dict(zip(feature_cols_desired, mapped))

    # Compute weather mean using the actual mapped names if present
    temp_col = desired_to_actual.get('Avg temp', 'Avg temp')
    precip_col = desired_to_actual.get('total precipitation (in)', 'total precipitation (in)')
    rain10_col = desired_to_actual.get('rain more than 10in', 'rain more than 10in')

    def col_mean(col_name, default=0.0):
        if col_name in df.columns:
            return float(df[col_name].dropna().mean())
        return default

    weather_mean = {
        'Avg temp': col_mean(temp_col, 72.0),
        'total precipitation (in)': col_mean(precip_col, 0.0),
        'rain more than 10in': col_mean(rain10_col, 0.0),
    }

    feature_cols_for_model = feature_cols_desired

    # ---------- 1. EVENT DEMAND PULL: HEATMAP + TOP-10 UPLIFT ----------
    mbs_col = 'Mercedes-Benz Stadium Est. Attendance'

    # elasticity: Δ riders per +10k when MBS goes 0 → 70k
    elasticity = []
    for _, s in stations.iterrows():
        base_row = {fc: 0.0 for fc in feature_cols_for_model}
        base_row['station_id'] = s['station_id']
        base_row['Avg temp'] = weather_mean['Avg temp']
        base_row['total precipitation (in)'] = weather_mean['total precipitation (in)']
        base_row['rain more than 10in'] = weather_mean['rain more than 10in']

        high_row = base_row.copy()
        high_row[mbs_col] = 70000.0

        base_pred = predict_rows(model, feature_cols_for_model, pd.DataFrame([base_row]))[0]
        high_pred = predict_rows(model, feature_cols_for_model, pd.DataFrame([high_row]))[0]
        slope_per_10k = (high_pred - base_pred) / (70000.0 / 10000.0)

        elasticity.append((s[station_col], s['station_id'], slope_per_10k))

    elasticity_df = (
        pd.DataFrame(elasticity, columns=['station', 'station_id', 'delta_per_10k'])
        .sort_values('delta_per_10k', ascending=False)
        .reset_index(drop=True)
    )

    # HEATMAP: extra riders vs no event, stations sorted by sensitivity
    att_bins = np.linspace(0, 90000, 19)  # 0, 5k, ..., 90k
    lift_rows = []
    station_order = []

    for _, row in elasticity_df.iterrows():
        sid = row['station_id']
        name = row['station']

        base_row = {fc: 0.0 for fc in feature_cols_for_model}
        base_row['station_id'] = sid
        base_row['Avg temp'] = weather_mean['Avg temp']
        base_row['total precipitation (in)'] = weather_mean['total precipitation (in)']
        base_row['rain more than 10in'] = weather_mean['rain more than 10in']

        base_pred = predict_rows(model, feature_cols_for_model, pd.DataFrame([base_row]))[0]

        lifts = []
        for a in att_bins:
            ev_row = base_row.copy()
            ev_row[mbs_col] = float(a)
            pred = predict_rows(model, feature_cols_for_model, pd.DataFrame([ev_row]))[0]
            lifts.append(pred - base_pred)  # extra riders vs no event
        lift_rows.append(lifts)
        station_order.append(name)

    lift = np.array(lift_rows)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(
        lift,
        aspect='auto',
        origin='lower',
        extent=[att_bins[0], att_bins[-1], 0, len(station_order)]
    )
    plt.colorbar(im, label='Extra riders vs no event')
    plt.yticks(np.arange(len(station_order)) + 0.5, station_order)
    plt.xlabel('Mercedes-Benz Stadium attendance')
    plt.ylabel('Station (sorted by event sensitivity)')
    plt.title('Event Demand Pull Heatmap: extra ridership vs no event')
    plt.tight_layout()
    out1 = os.path.join(HERE, 'event_pull_mbs_heatmap.png')
    plt.savefig(out1, dpi=150)
    plt.show()
    print('Saved', out1)

    # TOP-10 BAR: extra riders at 70k attendance
    topN = 10
    seventy_idx = np.where(att_bins == 70000)[0]
    if len(seventy_idx) > 0:
        idx = int(seventy_idx[0])
    else:
        idx = int(np.argmin(np.abs(att_bins - 70000)))
    seventy_lift = lift[:, idx]

    bar_df = pd.DataFrame({
        'station': station_order,
        'extra_riders_70k': seventy_lift
    }).sort_values('extra_riders_70k', ascending=False).head(topN)

    plt.figure(figsize=(10, 6))
    plt.bar(bar_df['station'], bar_df['extra_riders_70k'])
    plt.xticks(rotation=90)
    plt.ylabel('Extra riders vs no event (MBS = 70k)')
    plt.title('Top stations by World-Cup-scale event pull')
    plt.tight_layout()
    out1b = os.path.join(HERE, 'event_pull_top10_70k.png')
    plt.savefig(out1b, dpi=150)
    plt.show()
    print('Saved', out1b)

    # ---------- BASELINE vs MEGA-EVENT UPLIFT ----------
    baseline_preds = []
    mega_preds = []
    station_names = []

    for _, s in stations.iterrows():
        baseline = {fc: 0.0 for fc in feature_cols_for_model}
        baseline['station_id'] = s['station_id']
        baseline['Avg temp'] = weather_mean['Avg temp']
        baseline['total precipitation (in)'] = weather_mean['total precipitation (in)']
        baseline['rain more than 10in'] = weather_mean['rain more than 10in']

        mega = baseline.copy()
        mega['Mercedes-Benz Stadium Est. Attendance'] = 70000.0
        mega['State Farm Arena Est. Attendance'] = 18000.0
        mega['GWCC Est. Attendance'] = 4000.0
        mega['Other Venues (AmericasMart, Parks)'] = 2000.0

        baseline_preds.append(predict_rows(model, feature_cols_for_model, pd.DataFrame([baseline]))[0])
        mega_preds.append(predict_rows(model, feature_cols_for_model, pd.DataFrame([mega]))[0])
        station_names.append(s[station_col])

    x = np.arange(len(station_names))
    width = 0.4
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, baseline_preds, width, label='Baseline (no events)')
    plt.bar(x + width/2, mega_preds, width, label='Mega-event week')
    plt.xticks(x, station_names, rotation=90)
    plt.ylabel('Predicted Ridership')
    plt.title('Baseline vs Mega-Event Ridership per Station')
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(HERE, 'baseline_vs_mega_event.png')
    plt.savefig(out2, dpi=150)
    plt.show()
    print('Saved', out2)

    # event dependence
    event_dependence = []
    for b, m, name in zip(baseline_preds, mega_preds, station_names):
        if m > 0:
            ratio = (m - b) / m
        else:
            ratio = 0.0
        event_dependence.append((name, ratio))

    dep_df = pd.DataFrame(event_dependence, columns=['station', 'event_dependence']).sort_values('event_dependence', ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(dep_df['station'], dep_df['event_dependence'])
    plt.xticks(rotation=90)
    plt.ylabel('Event dependence ratio')
    plt.title('Share of Ridership Driven by Events (Mega-Event Scenario)')
    plt.tight_layout()
    out3 = os.path.join(HERE, 'event_dependence.png')
    plt.savefig(out3, dpi=150)
    plt.show()
    print('Saved', out3)


if __name__ == '__main__':
    main()
