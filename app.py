from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from prophet import Prophet
import os

app = Flask(__name__)

def run_prophet(prices: list[dict], horizon_days: int):
    """
    prices: list of {"ds": "2024-01-01", "y": 110.0}
    Returns forecast dict with target, high_band, low_band
    """
    df = pd.DataFrame(prices)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().sort_values("ds")

    if len(df) < 10:
        return None

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

    last = forecast.iloc[-1]
    current_price = float(df["y"].iloc[-1])

    target = round(float(last["yhat"]), 2)
    high   = round(float(last["yhat_upper"]), 2)
    low    = round(float(last["yhat_lower"]), 2)

    pct_change = round(((target - current_price) / current_price) * 100, 2)

    if pct_change > 2:
        trend = "Up"
    elif pct_change < -2:
        trend = "Down"
    else:
        trend = "Flat"

    return {
        "target":     target,
        "high_band":  high,
        "low_band":   low,
        "pct_change": pct_change,
        "trend":      trend,
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/forecast", methods=["POST"])
def forecast():
    """
    POST /forecast
    Body: {
        "ticker": "ZENITHBANK",
        "prices": [{"ds": "2024-01-01", "y": 110.0}, ...],
        "current_price": 110.0
    }
    Returns: {
        "ticker": "ZENITHBANK",
        "current_price": 110.0,
        "forecast_30d": { target, high_band, low_band, pct_change, trend },
        "forecast_90d": { target, high_band, low_band, pct_change, trend }
    }
    """
    body = request.get_json()
    if not body:
        return jsonify({"error": "No body provided"}), 400

    ticker        = body.get("ticker")
    prices        = body.get("prices", [])
    current_price = body.get("current_price")

    if not ticker or not prices or not current_price:
        return jsonify({"error": "ticker, prices, and current_price are required"}), 400

    f30 = run_prophet(prices, 30)
    f90 = run_prophet(prices, 90)

    if not f30 or not f90:
        return jsonify({"error": "Not enough price data to forecast"}), 422

    # Trend is determined by the 30d forecast
    trend = f30["trend"]

    return jsonify({
        "ticker":        ticker,
        "current_price": current_price,
        "trend":         trend,
        "forecast_30d":  f30,
        "forecast_90d":  f90,
    })


@app.route("/forecast/batch", methods=["POST"])
def forecast_batch():
    """
    POST /forecast/batch
    Body: { "stocks": [ { ticker, prices, current_price }, ... ] }
    Returns: { "results": [ forecast, ... ] }
    """
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

        f30 = run_prophet(prices, 30)
        f90 = run_prophet(prices, 90)

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
