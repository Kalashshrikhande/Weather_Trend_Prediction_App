import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load weather dataset
data = pd.read_csv("weather.csv")

# Convert date into useful numeric features
data["date"] = pd.to_datetime(data["date"])
data["day"] = data["date"].dt.day
data["month"] = data["date"].dt.month

# Features and target
X = data[["day", "month", "humidity", "rainfall", "wind_speed"]]
y = data["temperature"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "weather_model.pkl")

print("Weather model trained successfully!")