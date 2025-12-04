import os
import numpy as np
import matplotlib.pyplot as plt
from viz_common import find_and_load_data, load_model, prepare_data_and_encoder, compute_mean_weather, get_model_features, build_rows_for_station, predict_df, get_station_list_from_df

HERE = os.path.dirname(os.path.abspath(__file__))

df = find_and_load_data(HERE, base_name='final_long')
df, station_col, le = prepare_data_and_encoder(df)
temp_col, precip_col, rain10_col, mean_temp, mean_precip, mean_rain10 = compute_mean_weather(df)
booster = load_model(HERE)
model_features = get_model_features(booster)

station_labels, station_names = get_station_list_from_df(df, station_col, le)

temps = np.linspace(30, 90, num=61)
plt.figure(figsize=(12, 8))
for lab, name in zip(station_labels, station_names):
    preds = []
    for t in temps:
        overrides = { 'MBS': 0.0, 'SFA': 0.0, 'GWCC': 0.0, 'OTHER': 0.0 }
        overrides[temp_col] = float(t)
        overrides[precip_col] = float(mean_precip)
        overrides[rain10_col] = float(mean_rain10)
        rows = build_rows_for_station(df, model_features, le, lab, temp_col, precip_col, rain10_col, mean_temp, mean_precip, mean_rain10, overrides=overrides)
        p = predict_df(booster, model_features, rows)
        preds.append(float(np.mean(p)))
    plt.plot(temps, preds, label=str(name))

plt.title('Temperature sensitivity curves (30-90°F)')
plt.xlabel('Temperature (°F)')
plt.ylabel('Predicted ridership')
plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'plot_b_temp_sensitivity.png'), dpi=150)
plt.show()
