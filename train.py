import numpy as np

# fixed weather
weather_mean = df[["Avg temp", "total precipitation (in)", "rain more than 10in"]].mean()

temps = weather_mean["Avg temp"]
prec = weather_mean["total precipitation (in)"]
rain10 = weather_mean["rain more than 10in"]

# range of event sizes
mbs_range = np.linspace(0, 90000, 50)

curves = {}

for _, s in stations.iterrows():
    preds = []
    for mbs in mbs_range:
        row = pd.DataFrame({
            "station_id": [s.station_id],
            "Mercedes-Benz Stadium Est. Attendance": [mbs],
            "State Farm Arena Est. Attendance": [0],
            "GWCC Est. Attendance": [0],
            "Other Venues (AmericasMart, Parks)": [0],
            "Avg temp": [temps],
            "total precipitation (in)": [prec],
            "rain more than 10in": [rain10]
        })
        preds.append(model.predict(xgb.DMatrix(row))[0])
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
