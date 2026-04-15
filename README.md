# Forecast Studio Planner

## Neu in Version 3
- SKU-Portfolioübersicht
- ABC/XYZ-Segmentierung
- Accuracy-Tabelle je SKU
- Vergleich Prophet vs. Naive Baseline
- Detail-Forecast je SKU mit CSV-Export

## Lokal starten
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Erwartete CSV
Pflichtspalten:
- Datums-Spalte
- Numerische Zielspalte
- SKU- oder Segmentspalte

## Use Cases
- Demand-Planning Review je SKU
- Priorisierung nach AX/BX/CZ
- Modellvergleich gegen naive Baseline
- Identifikation, wo Prophet echten Mehrwert liefert

## Nächste sinnvolle Erweiterungen
- Hierarchisches Forecasting
- Separate Forecasts für Kunde x SKU
- Promotion-/Preis-Regressoren
- Forecast Value Added Analyse
