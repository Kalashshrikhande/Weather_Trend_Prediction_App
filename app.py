from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("weather_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        date = pd.to_datetime(request.form["date"])
        humidity = float(request.form["humidity"])
        rainfall = float(request.form["rainfall"])
        wind_speed = float(request.form["wind_speed"])

        data = pd.DataFrame([[
            date.day,
            date.month,
            humidity,
            rainfall,
            wind_speed
        ]], columns=["day", "month", "humidity", "rainfall", "wind_speed"])

        temperature = model.predict(data)[0]
        prediction = round(temperature, 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)