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

att_vals = np.linspace(0, 90000, num=91)
plt.figure(figsize=(12, 8))
for lab, name in zip(station_labels, station_names):
    preds = []
    for a in att_vals:
        # map attendance to model feature for MBS
        overrides = { 'MBS': float(a), 'SFA': 0.0, 'GWCC': 0.0, 'OTHER': 0.0 }
        rows = build_rows_for_station(df, model_features, le, lab, temp_col, precip_col, rain10_col, mean_temp, mean_precip, mean_rain10, overrides=overrides)
        p = predict_df(booster, model_features, rows)
        preds.append(float(np.mean(p)))
    plt.plot(att_vals, preds, label=str(name))

plt.title('Mercedes-Benz Stadium lift curves (one line per station)')
plt.xlabel('MBS attendance')
plt.ylabel('Predicted ridership')
plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'plot_a_mbs_lift.png'), dpi=150)
plt.show()
