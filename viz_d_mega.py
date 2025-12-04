import os
import matplotlib.pyplot as plt
from viz_common import find_and_load_data, load_model, prepare_data_and_encoder, compute_mean_weather, get_model_features, build_rows_for_station, predict_df, get_station_list_from_df

HERE = os.path.dirname(os.path.abspath(__file__))

df = find_and_load_data(HERE, base_name='final_long')
df, station_col, le = prepare_data_and_encoder(df)
temp_col, precip_col, rain10_col, mean_temp, mean_precip, mean_rain10 = compute_mean_weather(df)
booster = load_model(HERE)
model_features = get_model_features(booster)

station_labels, station_names = get_station_list_from_df(df, station_col, le)

mega_values = {
    'MBS': 70000.0,
    'SFA': 18000.0,
    'GWCC': 4000.0,
    'OTHER': 2000.0,
    'temperature': 72.0,
    'precipitation': 0.0,
    'rain10': 0.0,
}

mega_preds = []
for lab, name in zip(station_labels, station_names):
    rows = build_rows_for_station(df, model_features, le, lab, temp_col, precip_col, rain10_col, mean_temp, mean_precip, mean_rain10, overrides=mega_values)
    p = predict_df(booster, model_features, rows)
    mega_preds.append(float(np.mean(p)))

plt.figure(figsize=(12, 6))
x = range(len(station_names))
plt.bar(x, mega_preds, color='orange')
plt.xticks(x, station_names, rotation=90)
plt.ylabel('Predicted ridership')
plt.title('Mega Event scenario ridership per station')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'plot_d_mega_event_bar.png'), dpi=150)
plt.show()
