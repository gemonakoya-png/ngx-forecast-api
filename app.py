from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

app = Flask(__name__)


def run_forecast(prices, horizon_days):
    df = pd.DataFrame(prices)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().sort_values("ds").reset_index(drop=True)

    if len(df) < 10:
        return None

    df["t"] = (df["ds"] - df["ds"].min()).dt.days
    X = df[["t"]].values
    y = df["y"].values

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred_train = model.predict(X_poly)
    residuals = y - y_pred_train
    std = np.std(residuals)

    last_t = df["t"].max()
    future_t = np.array([[last_t + horizon_days]])
    future_poly = poly.transform(future_t)
    target = float(model.predict(future_poly)[0])

    high_band = round(target + 1.5 * std, 2)
    low_band  = round(target - 1.5 * std, 2)
    target    = round(target, 2)

    current_price = float(df["y"].iloc[-1])
    pct_change    = round(((target - current_price) / current_price) * 100, 2)

    if pct_change > 2:
        trend = "Up"
    elif pct_change < -2:
        trend = "Down"
    else:
        trend = "Flat"

    return {
        "target":     target,
        "high_band":  high_band,
        "low_band":   low_band,
        "pct_change": pct_change,
        "trend":      trend,
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/forecast", methods=["POST"])
def forecast():
    body = request.get_json()
    if not body:
        return jsonify({"error": "No body provided"}), 400

    ticker        = body.get("ticker")
    prices        = body.get("prices", [])
    current_price = body.get("current_price")

    if not ticker or not prices or not current_price:
        return jsonify({"error": "ticker, prices, and current_price are required"}), 400

    f30 = run_forecast(prices, 30)
    f90 = run_forecast(prices, 90)

    if not f30 or not f90:
        return jsonify({"error": "Not enough price data to forecast"}), 422

    return jsonify({
        "ticker":        ticker,
        "current_price": current_price,
        "trend":         f30["trend"],
        "forecast_30d":  f30,
        "forecast_90d":  f90,
    })


@app.route("/forecast/batch", methods=["POST"])
def forecast_batch():
    body = request.get_json()
    if not body:
        return jsonify({"error": "No body provided"}), 400

    stocks  = body.get("stocks", [])
    results = []

    for stock in stocks:
        ticker        = stock.get("ticker")
        prices        = stock.get("prices", [])
        current_price = stock.get("current_price")

        if not ticker or not prices or not current_price:
            continue

        f30 = run_forecast(prices, 30)
        f90 = run_forecast(prices, 90)

        if not f30 or not f90:
            continue

        results.append({
            "ticker":        ticker,
            "current_price": current_price,
            "trend":         f30["trend"],
            "forecast_30d":  f30,
            "forecast_90d":  f90,
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
