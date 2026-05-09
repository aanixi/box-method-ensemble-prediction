# Box Method Day Trading -strategian onnistumisen ennustaminen

Ensemble-menetelmiä hyödyntävä projektityö, jossa tutkitaan Box Method -day trading -strategian kannattavuutta kryptomarkkinassa.

## Tutkimuskysymys

Tuottaako YouTube-trading-yhteisössä suosittu Box Method -strategia tilastollisesti merkittävää edge-etua, ja voivatko ensemble-mallit erottaa onnistuneet setupit epäonnistuneista paremmin kuin pelkkä mekaaninen sääntö?

## Data

Binance API, ETH/USDT ja SOL/USDT, 15 minuutin aikakehys, ajanjakso 1.1.2022 – 8.5.2026.

## Menetelmät

- Random Forest (bagging)
- XGBoost (boosting)
- Stacking Classifier (meta-ensemble)
- Baseline: logistinen regressio + mekaaninen sääntö

## Status

🚧 Työn alla — projekti aloitettu toukokuussa 2026