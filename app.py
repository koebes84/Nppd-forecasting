import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(page_title='Forecast Studio', page_icon='📈', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.metric-card {padding: 1rem; border: 1px solid rgba(120,120,120,.2); border-radius: 16px; background: rgba(250,250,250,.03);}
.small-note {color: #6b7280; font-size: 0.9rem;}
</style>
''', unsafe_allow_html=True)

st.title('📈 Forecast Studio mit Prophet')
st.caption('CSV-Upload, automatische Zeitreihen-Erkennung, Prophet-Modell, Komponenten-Charts und Export der Prognose.')

@dataclass
class ForecastConfig:
    periods: int
    freq: str
    changepoint_prior_scale: float
    seasonality_prior_scale: float
    yearly_seasonality: bool
    weekly_seasonality: bool
    daily_seasonality: bool


def infer_datetime_column(df: pd.DataFrame):
    for col in df.columns:
        converted = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
        if converted.notna().mean() > 0.8:
            return col
    return None


def infer_target_column(df: pd.DataFrame, exclude: str | None = None):
    candidates = [c for c in df.columns if c != exclude]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    if numeric:
        return numeric[0]
    for col in candidates:
        cleaned = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        if cleaned.notna().mean() > 0.8:
            return col
    return None


def prepare_prophet_df(df: pd.DataFrame, date_col: str, target_col: str):
    out = df[[date_col, target_col]].copy()
    out.columns = ['ds', 'y']
    out['ds'] = pd.to_datetime(out['ds'], errors='coerce')
    if not pd.api.types.is_numeric_dtype(out['y']):
        out['y'] = pd.to_numeric(out['y'].astype(str).str.replace(',', '.'), errors='coerce')
    out = out.dropna().sort_values('ds')
    out = out.groupby('ds', as_index=False)['y'].sum()
    return out


def build_model(cfg: ForecastConfig):
    return Prophet(
        changepoint_prior_scale=cfg.changepoint_prior_scale,
        seasonality_prior_scale=cfg.seasonality_prior_scale,
        yearly_seasonality=cfg.yearly_seasonality,
        weekly_seasonality=cfg.weekly_seasonality,
        daily_seasonality=cfg.daily_seasonality,
        interval_width=0.8,
    )


def detect_freq(ds: pd.Series) -> str:
    diffs = ds.sort_values().diff().dropna()
    if diffs.empty:
        return 'D'
    median_days = diffs.dt.total_seconds().median() / 86400
    if median_days <= 1.5:
        return 'D'
    if median_days <= 8:
        return 'W'
    if median_days <= 31:
        return 'MS'
    if median_days <= 92:
        return 'QS'
    return 'YS'


def components_chart(fcst: pd.DataFrame, freq: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['trend'], mode='lines', name='Trend'))
    if 'yearly' in fcst.columns and fcst['yearly'].abs().sum() > 0:
        fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yearly'], mode='lines', name='Yearly'))
    if 'weekly' in fcst.columns and fcst['weekly'].abs().sum() > 0:
        fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['weekly'], mode='lines', name='Weekly'))
    if 'daily' in fcst.columns and fcst['daily'].abs().sum() > 0:
        fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['daily'], mode='lines', name='Daily'))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), title='Trend & Saisonalität')
    return fig


with st.sidebar:
    st.header('Konfiguration')
    uploaded = st.file_uploader('CSV-Datei hochladen', type=['csv'])
    periods = st.slider('Forecast-Horizont', min_value=7, max_value=365, value=90, step=1)
    freq = st.selectbox('Frequenz', ['Auto', 'D', 'W', 'MS', 'QS', 'YS'], index=0)
    cps = st.slider('Trend-Flexibilität', 0.001, 1.0, 0.05, 0.001)
    sps = st.slider('Seasonality Prior', 0.01, 20.0, 10.0, 0.01)
    yearly = st.toggle('Jährliche Saisonalität', value=True)
    weekly = st.toggle('Wöchentliche Saisonalität', value=True)
    daily = st.toggle('Tägliche Saisonalität', value=False)

st.info('Erwartetes Format: mindestens eine Datums-Spalte und eine numerische Zielspalte. Beispiel: date,sales')

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    date_guess = infer_datetime_column(raw)
    target_guess = infer_target_column(raw, exclude=date_guess)

    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox('Datums-Spalte', raw.columns, index=list(raw.columns).index(date_guess) if date_guess in raw.columns else 0)
    with c2:
        target_col = st.selectbox('Ziel-Spalte', raw.columns, index=list(raw.columns).index(target_guess) if target_guess in raw.columns else min(1, len(raw.columns)-1))

    df = prepare_prophet_df(raw, date_col, target_col)

    if len(df) < 10:
        st.error('Zu wenige gültige Datenpunkte. Bitte mindestens 10 Zeilen mit Datum und Wert bereitstellen.')
    else:
        detected_freq = detect_freq(df['ds'])
        effective_freq = detected_freq if freq == 'Auto' else freq
        cfg = ForecastConfig(periods, effective_freq, cps, sps, yearly, weekly, daily)
        model = build_model(cfg)
        model.fit(df)
        future = model.make_future_dataframe(periods=cfg.periods, freq=cfg.freq)
        forecast = model.predict(future)

        history_last = float(df['y'].iloc[-1])
        forecast_last = float(forecast['yhat'].iloc[-1])
        delta_pct = ((forecast_last / history_last) - 1) * 100 if history_last != 0 else np.nan

        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Historische Punkte', f'{len(df):,}'.replace(',', '.'))
        k2.metric('Erkannte Frequenz', effective_freq)
        k3.metric('Letzter Ist-Wert', f'{history_last:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.'))
        k4.metric('Ende Forecast', f'{forecast_last:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.'), None if np.isnan(delta_pct) else f'{delta_pct:+.1f}%')

        st.subheader('Datenvorschau')
        st.dataframe(df.tail(20), use_container_width=True)

        st.subheader('Forecast')
        fig_forecast = plot_plotly(model, forecast)
        fig_forecast.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader('Komponenten')
        st.plotly_chart(components_chart(forecast, effective_freq), use_container_width=True)

        export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].copy()
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
        st.download_button('Forecast als CSV herunterladen', data=csv_bytes, file_name='prophet_forecast.csv', mime='text/csv')

        with st.expander('Deployment öffentlich bereitstellen'):
            st.markdown('''
1. Repository mit `app.py` und `requirements.txt` nach GitHub pushen.  
2. Auf [share.streamlit.io](https://share.streamlit.io) mit GitHub verbinden.  
3. Repository und Branch auswählen, dann `app.py` als Entry Point setzen.  
4. Nach dem Deploy erhältst du eine öffentliche URL.  

Alternative: Deployment auf Hugging Face Spaces oder Render mit identischem Projektsetup.
''')

        with st.expander('Technische Hinweise'):
            st.code(json.dumps({
                'detected_frequency': effective_freq,
                'date_column': date_col,
                'target_column': target_col,
                'rows_used': int(len(df)),
                'forecast_horizon': int(cfg.periods)
            }, indent=2), language='json')
else:
    st.markdown('''
### Schnellstart
- Lade eine CSV mit Datum und Kennzahl hoch.
- Wähle die Spalten für `ds` und `y`.
- Passe Forecast-Horizont und Seasonality an.
- Lade die Prognose als CSV herunter.
''')

    sample = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=24, freq='MS'),
        'sales': [120, 128, 133, 142, 150, 158, 166, 172, 181, 177, 169, 160,
                  164, 171, 178, 186, 195, 204, 213, 220, 228, 223, 214, 206]
    })
    st.dataframe(sample, use_container_width=True)
