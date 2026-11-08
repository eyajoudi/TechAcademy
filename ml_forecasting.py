"""Revenue forecasting from actual recorded data, with a separate synthetic demo."""
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def forecast_from_records(records, periods=7):
    df = pd.DataFrame(records)
    if df.empty or "created_at" not in df or "payment_status" not in df:
        return None, "Aucune donnée historique disponible."
    df = df[df["payment_status"] == "succeeded"].copy()
    if df.empty:
        return None, "Aucun paiement réussi dans les données historiques."
    df["date"] = pd.to_datetime(df["created_at"], errors="coerce").dt.date
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    daily = df.groupby("date", as_index=False)["amount"].sum().sort_values("date")
    if len(daily) < 3:
        return None, "Il faut au moins 3 journées distinctes avec des paiements réussis pour entraîner une prévision sur les données réelles."
    x = np.arange(len(daily)).reshape(-1, 1)
    y = daily["amount"].values
    model = LinearRegression().fit(x, y)
    future_x = np.arange(len(daily), len(daily) + periods).reshape(-1, 1)
    pred = np.maximum(model.predict(future_x), 0)
    future_dates = pd.date_range(pd.to_datetime(daily["date"].max()) + pd.Timedelta(days=1), periods=periods)
    result = pd.DataFrame({"date": future_dates, "predicted_revenue": pred})
    return {"history": daily, "forecast": result, "model": "Linear Regression", "training_days": len(daily)}, None

def synthetic_demo_forecast(periods=7):
    """Pedagogical synthetic data, deliberately kept separate from real KPIs."""
    x = np.arange(14).reshape(-1, 1)
    y = np.array([250, 280, 270, 310, 305, 340, 360, 355, 390, 410, 405, 440, 465, 480], dtype=float)
    model = LinearRegression().fit(x, y)
    future_x = np.arange(14, 14 + periods).reshape(-1, 1)
    pred = np.maximum(model.predict(future_x), 0)
    dates = pd.date_range("2026-11-08", periods=periods)
    return pd.DataFrame({"date": dates, "predicted_revenue": pred}), pd.DataFrame({"day": np.arange(1,15), "revenue": y})
