import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(page_title="Forecast Control Tower", page_icon="📈", layout="wide")

st.title("📈 Forecast Control Tower")
st.caption("CSV-Upload, Prophet-Forecast und Download")

def prepare_prophet_df(df, date_col, target_col):
    out = df[[date_col, target_col]].copy()
    out.columns = ["ds", "y"]

    out["ds"] = pd.to_datetime(out["ds"], errors="coerce", dayfirst=True)

    out["y"] = (
        out["y"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    out["y"] = pd.to_numeric(out["y"], errors="coerce")

    out = out.dropna(subset=["ds", "y"]).sort_values("ds")
    out = out.groupby("ds", as_index=False)["y"].sum()

    return out

def infer_datetime_column(df):
    for col in df.columns:
        test = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        if test.notna().mean() > 0.8:
            return col
    return None

def infer_target_column(df, exclude=None):
    for col in df.columns:
        if col == exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    for col in df.columns:
        if col == exclude:
            continue
        test = pd.to_numeric(
            df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce"
        )
        if test.notna().mean() > 0.8:
            return col
    return None

uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

if uploaded_file is not None:
    raw = pd.read_csv(uploaded_file, sep=";")

    st.subheader("Vorschau der Rohdaten")
    st.dataframe(raw.head(20), use_container_width=True)

    date_guess = infer_datetime_column(raw)
    target_guess = infer_target_column(raw, exclude=date_guess)

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox(
            "Datums-Spalte",
            raw.columns,
            index=list(raw.columns).index(date_guess) if date_guess in raw.columns else 0
        )
    with col2:
        target_col = st.selectbox(
            "Ziel-Spalte",
            raw.columns,
            index=list(raw.columns).index(target_guess) if target_guess in raw.columns else min(1, len(raw.columns) - 1)
        )

    df = prepare_prophet_df(raw, date_col, target_col)

    st.write("Gültige Zeilen nach Bereinigung:", len(df))

    if len(df) < 10:
        st.error("Zu wenige gültige Datenpunkte. Bitte mindestens 10 Zeilen mit Datum und Wert bereitstellen.")
        st.stop()

    periods = st.slider("Forecast-Horizont", min_value=7, max_value=104, value=13, step=1)
    freq = st.selectbox("Frequenz", ["D", "W", "MS", "QS", "YS"], index=1)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True if freq in ["D", "W"] else False,
        daily_seasonality=False,
        interval_width=0.8
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    st.subheader("Forecast")
    fig = plot_plotly(model, forecast)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prognose herunterladen")
    export_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Forecast als CSV herunterladen",
        data=csv_bytes,
        file_name="forecast.csv",
        mime="text/csv"
    )

    st.subheader("Forecast-Tabelle")
    st.dataframe(export_df.tail(30), use_container_width=True)

else:
    st.info("Bitte eine CSV-Datei hochladen. Erwartet werden mindestens eine Datums-Spalte und eine numerische Wertespalte.")
    st.markdown(
        """
**Beispiel-Format:**

| date | sales |
|---|---:|
| 01.01.2024 | 10281 |
| 08.01.2024 | 8443 |
"""
    )
