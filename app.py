import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.make_holidays import make_holidays_df

st.set_page_config(page_title="Forecast Control Tower", page_icon="📈", layout="wide")

st.title("📈 Forecast Control Tower")
st.caption("CSV-Upload, Prophet, Backtesting, ABC/XYZ, FVA und Forecast-Export.")


def read_uploaded_csv(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode("utf-8", errors="ignore")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    sep = ";" if first_line.count(";") >= first_line.count(",") else ","
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, sep=sep)


def infer_datetime_column(df):
    for col in df.columns:
        s = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        if s.notna().mean() >= 0.8:
            return col
    return None


def infer_target_column(df, exclude=None):
    candidates = [c for c in df.columns if c != exclude]
    for col in candidates:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    for col in candidates:
        s = pd.to_numeric(
            df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if s.notna().mean() >= 0.8:
            return col
    return None


def candidate_dims(df, exclude):
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        if 1 < df[c].nunique(dropna=True) <= 500:
            out.append(c)
    return out


def prepare_prophet_df(df, date_col, target_col):
    out = df[[date_col, target_col]].copy()
    out.columns = ["ds", "y"]
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce", dayfirst=True)
    out["y"] = pd.to_numeric(
        out["y"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    out = out.dropna(subset=["ds", "y"]).sort_values("ds")
    out = out.groupby("ds", as_index=False)["y"].sum()
    return out


def detect_freq(ds):
    diffs = ds.sort_values().diff().dropna()
    if diffs.empty:
        return "D"
    median_days = diffs.dt.total_seconds().median() / 86400
    if median_days <= 1.5:
        return "D"
    if median_days <= 8:
        return "W"
    if median_days <= 31:
        return "MS"
    if median_days <= 92:
        return "QS"
    return "YS"


def fmt(v):
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def make_model(cfg, years):
    holidays = make_holidays_df(year_list=years, country=cfg["holiday_country"]) if cfg["holiday_country"] else None
    return Prophet(
        holidays=holidays,
        changepoint_prior_scale=cfg["cps"],
        seasonality_prior_scale=cfg["sps"],
        yearly_seasonality=cfg["yearly"],
        weekly_seasonality=cfg["weekly"],
        daily_seasonality=cfg["daily"],
        interval_width=0.8,
    )


def calc_metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - actual
    abs_err = np.abs(err)
    mae = abs_err.mean()
    rmse = np.sqrt((err ** 2).mean())
    mape = np.nanmean(np.where(actual != 0, abs_err / np.abs(actual), np.nan)) * 100
    wmape = abs_err.sum() / np.abs(actual).sum() * 100 if np.abs(actual).sum() != 0 else np.nan
    bias = err.sum()
    bias_pct = bias / actual.sum() * 100 if actual.sum() != 0 else np.nan
    return mae, rmse, mape, wmape, bias, bias_pct


def naive_forecast(train, horizon):
    return np.repeat(float(train["y"].iloc[-1]), horizon)


def classify_traffic_light(wmape, fva):
    if pd.isna(wmape):
        return "Red"
    if wmape <= 15 and fva > 0:
        return "Green"
    if wmape <= 25:
        return "Yellow"
    return "Red"


def abc_xyz_table(raw, date_col, target_col, sku_col):
    tmp = raw[[date_col, target_col, sku_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce", dayfirst=True)
    tmp[target_col] = pd.to_numeric(
        tmp[target_col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    tmp = tmp.dropna()

    vol = tmp.groupby(sku_col)[target_col].sum().sort_values(ascending=False).reset_index()
    vol.columns = [sku_col, "volume"]
    vol["cum_share"] = vol["volume"].cumsum() / vol["volume"].sum()
    vol["ABC"] = np.select([vol["cum_share"] <= 0.8, vol["cum_share"] <= 0.95], ["A", "B"], default="C")

    stat = tmp.groupby(sku_col)[target_col].agg(["mean", "std"]).reset_index()
    stat["cv"] = np.where(stat["mean"] != 0, stat["std"] / stat["mean"], np.nan)
    stat["XYZ"] = np.select([stat["cv"] <= 0.3, stat["cv"] <= 0.6], ["X", "Y"], default="Z")

    merged = vol.merge(stat[[sku_col, "mean", "std", "cv", "XYZ"]], on=sku_col, how="left")
    merged["segment"] = merged["ABC"] + merged["XYZ"]
    return merged


def evaluate_series(raw, date_col, target_col, sku_col, selected_skus, cfg, max_series=30):
    rows = []
    for sku in selected_skus[:max_series]:
        sdf = raw[raw[sku_col].astype(str) == str(sku)].copy()
        ts = prepare_prophet_df(sdf, date_col, target_col)
        if len(ts) <= cfg["holdout"] + 8:
            continue

        freq = detect_freq(ts["ds"]) if cfg["freq"] == "Auto" else cfg["freq"]
        local = dict(cfg)
        local["freq"] = freq

        train = ts.iloc[:-local["holdout"]].copy()
        test = ts.iloc[-local["holdout"]:].copy()
        years = sorted(set(ts["ds"].dt.year.tolist() + [ts["ds"].max().year + 1]))

        model = make_model(local, years)
        model.fit(train)
        fcst = model.predict(model.make_future_dataframe(periods=local["holdout"], freq=local["freq"]))
        pred = fcst[["ds", "yhat"]].merge(test, on="ds", how="inner")
        naive = naive_forecast(train, len(pred))

        _, rmse_p, mape_p, wmape_p, _, biasp = calc_metrics(pred["y"], pred["yhat"])
        _, rmse_n, mape_n, wmape_n, _, biasn = calc_metrics(pred["y"], naive)
        fva = wmape_n - wmape_p
        light = classify_traffic_light(wmape_p, fva)

        rows.append({
            sku_col: sku,
            "hist_points": len(ts),
            "holdout_sum": pred["y"].sum(),
            "prophet_wmape": wmape_p,
            "naive_wmape": wmape_n,
            "fva_points": fva,
            "prophet_mape": mape_p,
            "naive_mape": mape_n,
            "prophet_rmse": rmse_p,
            "naive_rmse": rmse_n,
            "prophet_bias_pct": biasp,
            "naive_bias_pct": biasn,
            "winner": "Prophet" if wmape_p < wmape_n else "Naive",
            "traffic_light": light,
            "exception_flag": "Review" if (light == "Red" or fva < 0) else "OK",
        })
    return pd.DataFrame(rows)


def topdown_bottomup_gap(raw, date_col, target_col, sku_col, selected_skus, cfg):
    subset = raw[raw[sku_col].astype(str).isin([str(x) for x in selected_skus])].copy()
    total_ts = prepare_prophet_df(subset, date_col, target_col)
    if len(total_ts) <= cfg["holdout"] + 8:
        return None

    freq = detect_freq(total_ts["ds"]) if cfg["freq"] == "Auto" else cfg["freq"]
    years = sorted(set(total_ts["ds"].dt.year.tolist() + [total_ts["ds"].max().year + 1]))
    top_cfg = dict(cfg)
    top_cfg["freq"] = freq

    model = make_model(top_cfg, years)
    model.fit(total_ts)
    top_fcst = model.predict(model.make_future_dataframe(periods=cfg["periods"], freq=freq))
    top_sum = top_fcst[top_fcst["ds"] > total_ts["ds"].max()]["yhat"].sum()

    bottom_sum = 0
    for sku in selected_skus:
        sdf = raw[raw[sku_col].astype(str) == str(sku)].copy()
        ts = prepare_prophet_df(sdf, date_col, target_col)
        if len(ts) <= 8:
            continue
        years_s = sorted(set(ts["ds"].dt.year.tolist() + [ts["ds"].max().year + 1]))
        s_cfg = dict(cfg)
        s_cfg["freq"] = freq
        model_s = make_model(s_cfg, years_s)
        model_s.fit(ts)
        fcst_s = model_s.predict(model_s.make_future_dataframe(periods=cfg["periods"], freq=freq))
        bottom_sum += fcst_s[fcst_s["ds"] > ts["ds"].max()]["yhat"].sum()

    gap = bottom_sum - top_sum
    gap_pct = gap / top_sum * 100 if top_sum != 0 else np.nan
    return pd.DataFrame({"view": ["Top-Down", "Bottom-Up"], "forecast_sum": [top_sum, bottom_sum]}), gap, gap_pct


uploaded = st.file_uploader("CSV hochladen", type=["csv"])

with st.sidebar:
    st.header("Konfiguration")
    periods = st.slider("Forecast-Horizont", 4, 104, 13)
    freq_choice = st.selectbox("Frequenz", ["Auto", "D", "W", "MS", "QS", "YS"])
    cps = st.slider("Trend-Flexibilität", 0.001, 1.0, 0.05, 0.001)
    sps = st.slider("Seasonality Prior", 0.01, 20.0, 10.0, 0.01)
    yearly = st.toggle("Jährliche Saisonalität", True)
    weekly = st.toggle("Wöchentliche Saisonalität", True)
    daily = st.toggle("Tägliche Saisonalität", False)
    holiday_country = st.selectbox("Feiertage", ["Kein", "DE", "US", "FR", "GB"], index=1)
    holdout = st.slider("Holdout-Perioden", 4, 26, 8)

if uploaded is not None:
    raw = read_uploaded_csv(uploaded)

    date_guess = infer_datetime_column(raw)
    target_guess = infer_target_column(raw, exclude=date_guess)
    dims = candidate_dims(raw, [date_guess, target_guess])

    c1, c2, c3 = st.columns(3)
    with c1:
        date_col = st.selectbox(
            "Datums-Spalte",
            raw.columns,
            index=list(raw.columns).index(date_guess) if date_guess in raw.columns else 0,
        )
    with c2:
        target_col = st.selectbox(
            "Ziel-Spalte",
            raw.columns,
            index=list(raw.columns).index(target_guess) if target_guess in raw.columns else min(1, len(raw.columns) - 1),
        )
    with c3:
        sku_col = st.selectbox("SKU-/Segment-Spalte", dims if dims else ["Keine"])

    if sku_col == "Keine":
        st.error("Für das Planner-Dashboard wird eine SKU- oder Segmentspalte benötigt.")
        st.stop()

    cfg = {
        "periods": periods,
        "freq": freq_choice,
        "cps": cps,
        "sps": sps,
        "yearly": yearly,
        "weekly": weekly,
        "daily": daily,
        "holiday_country": None if holiday_country == "Kein" else holiday_country,
        "holdout": holdout,
    }

    portfolio = abc_xyz_table(raw, date_col, target_col, sku_col)
    skus = portfolio[sku_col].astype(str).tolist()
    selected_skus = st.multiselect("SKUs für Analyse", skus, default=skus[:5])

    st.subheader("Portfolio-Übersicht")
    p1, p2 = st.columns([1.2, 1])
    with p1:
        st.dataframe(portfolio.sort_values("volume", ascending=False), use_container_width=True)
    with p2:
        seg = portfolio.groupby("segment").size().reset_index(name="count")
        fig_seg = go.Figure(data=[go.Bar(x=seg["segment"], y=seg["count"])])
        fig_seg.update_layout(title="Portfolio nach ABC/XYZ", height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_seg, use_container_width=True)

    eval_df = evaluate_series(raw, date_col, target_col, sku_col, selected_skus, cfg)
    st.subheader("Exception Management")
    if eval_df.empty:
        st.warning("Nicht genug Historie für die ausgewählten SKUs.")
    else:
        red_cnt = int((eval_df["traffic_light"] == "Red").sum())
        yellow_cnt = int((eval_df["traffic_light"] == "Yellow").sum())
        green_cnt = int((eval_df["traffic_light"] == "Green").sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Green", green_cnt)
        k2.metric("Yellow", yellow_cnt)
        k3.metric("Red", red_cnt)
        k4.metric("Ø FVA", f"{eval_df['fva_points'].mean():.2f} pp")

        st.dataframe(eval_df.sort_values(["traffic_light", "prophet_wmape"]), use_container_width=True)

        color_map = {"Green": "#16a34a", "Yellow": "#f59e0b", "Red": "#dc2626"}
        fig_scatter = go.Figure()
        for light, color in color_map.items():
            d = eval_df[eval_df["traffic_light"] == light]
            fig_scatter.add_trace(
                go.Scatter(
                    x=d["prophet_wmape"],
                    y=d["fva_points"],
                    mode="markers",
                    name=light,
                    marker=dict(color=color, size=10),
                    text=d[sku_col],
                )
            )
        fig_scatter.update_layout(
            title="Exception Map: Accuracy vs. FVA",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Prophet WMAPE",
            yaxis_title="FVA (pp)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Top-Down vs. Bottom-Up")
    td_bu = topdown_bottomup_gap(raw, date_col, target_col, sku_col, selected_skus, cfg)
    if td_bu is not None:
        compare_df, gap, gap_pct = td_bu
        t1, t2, t3 = st.columns(3)
        t1.metric("Top-Down Summe", fmt(compare_df.iloc[0, 1]))
        t2.metric("Bottom-Up Summe", fmt(compare_df.iloc[1, 1]))
        t3.metric("Gap", fmt(gap), f"{gap_pct:+.2f}%")

        fig_gap = go.Figure(data=[go.Bar(x=compare_df["view"], y=compare_df["forecast_sum"])])
        fig_gap.update_layout(title="Forecast-Summe im Vergleich", height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gap, use_container_width=True)

    st.subheader("Einzel-SKU-Ansicht")
    if selected_skus:
        sku_pick = st.selectbox("SKU auswählen", selected_skus)
        sdf = raw[raw[sku_col].astype(str) == str(sku_pick)].copy()
        ts = prepare_prophet_df(sdf, date_col, target_col)

        if len(ts) > holdout + 8:
            freq = detect_freq(ts["ds"]) if freq_choice == "Auto" else freq_choice
            years = sorted(set(ts["ds"].dt.year.tolist() + [ts["ds"].max().year + 1]))
            local_cfg = dict(cfg)
            local_cfg["freq"] = freq

            model = make_model(local_cfg, years)
            model.fit(ts)
            fcst = model.predict(model.make_future_dataframe(periods=periods, freq=freq))

            fig = plot_plotly(model, fcst)
            fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            out = fcst[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
            out[sku_col] = sku_pick
            st.download_button(
                "Forecast CSV herunterladen",
                out.to_csv(index=False).encode("utf-8"),
                file_name=f"control_tower_{sku_pick}.csv",
                mime="text/csv",
            )
        else:
            st.warning("Für diese SKU ist die Historie zu kurz.")

else:
    st.markdown("""
### Schnellstart
- CSV mit Datum, Umsatz/Menge und SKU hochladen.
- Datums- und Zielspalte auswählen.
- SKU-Spalte auswählen.
- Portfolio, FVA und Exceptions prüfen.
""")
