"""
Datan hakumoduuli Binance Spot API:sta.

Hakee historiallista OHLCV-kynttilädataa (Open, High, Low, Close, Volume)
ja tallentaa sen Parquet-muodossa nopeaa uudelleenkäyttöä varten.

Binance API:
- Päätepiste: https://api.binance.com/api/v3/klines
- Maksimi 1000 kynttilää per pyyntö → pitkät jaksot vaativat sivutusta
- Ei vaadi API-avainta julkiseen markkinadataan
- Rate limit: 1200 painoyksikköä per minuutti, klines = 2 yksikköä per kutsu
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # Binancen maksimi yhdellä pyynnöllä

# Aikakehysten kestot millisekunteina (helpottaa sivutusta)
INTERVAL_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def _to_ms(dt: datetime) -> int:
    """Muuntaa datetimen Unix-aikaleimaksi millisekunneissa (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    sleep_sec: float = 0.25,
) -> pd.DataFrame:
    """
    Hakee kynttilädataa Binance API:sta annetulla aikavälillä.

    Sivuttaa automaattisesti 1000 kynttilän erissä, koska se on Binancen
    maksimi yhdessä vastauksessa.

    Args:
        symbol: Esim. "ETHUSDT"
        interval: Esim. "15m", "1h", "4h"
        start: Aloituspäivä (UTC)
        end: Lopetuspäivä (UTC)
        sleep_sec: Tauko pyyntöjen välillä (kohteliaisuus API:lle)

    Returns:
        DataFrame, jossa sarakkeet:
        open, high, low, close, volume, close_time, quote_volume,
        trades, taker_base_vol, taker_quote_vol
        Indeksinä open_time (UTC).
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Tuntematon aikakehys: {interval}")

    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    interval_ms = INTERVAL_MS[interval]

    all_rows = []
    current_start = start_ms
    request_count = 0

    print(f"  Haetaan {symbol} {interval} aikaväliltä {start.date()} – {end.date()}...")

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }

        response = requests.get(BINANCE_API_URL, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        request_count += 1

        if not batch:
            # Ei enempää dataa
            break

        all_rows.extend(batch)

        # Seuraava erä alkaa viimeisen kynttilän jälkeen
        last_open_time = batch[-1][0]
        next_start = last_open_time + interval_ms

        if next_start <= current_start:
            # Suoja äärettömälle silmukalle
            break

        current_start = next_start

        # Tauko pyyntöjen välillä
        time.sleep(sleep_sec)

    print(f"    {request_count} API-kutsua, {len(all_rows)} kynttilää saatu")

    if not all_rows:
        return pd.DataFrame()

    # Muunna DataFrameksi ja siisti tietotyypit
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base_vol", "taker_quote_vol", "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=columns)

    # Tietotyyppien muunnokset
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_base_vol", "taker_quote_vol"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    df["trades"] = df["trades"].astype(int)

    # Pudota turha sarake ja aseta indeksi
    df = df.drop(columns=["ignore"])
    df = df.set_index("open_time").sort_index()

    # Poista mahdolliset duplikaatit (jos sivutus aiheutti niitä)
    df = df[~df.index.duplicated(keep="first")]

    return df


def fetch_and_save(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    output_dir: Path = Path("data/raw"),
) -> Path:
    """
    Hakee kynttilädataa ja tallentaa sen Parquet-tiedostoon.

    Returns:
        Tallennetun tiedoston polku
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}_{interval}.parquet"

    df = fetch_klines(symbol, interval, start, end)

    if df.empty:
        print(f"  VAROITUS: Ei dataa {symbol} {interval}")
        return output_path

    df.to_parquet(output_path, engine="pyarrow", compression="snappy")
    print(f"  Tallennettu: {output_path} "
          f"({len(df)} riviä, {output_path.stat().st_size / 1024:.1f} KB)")

    return output_path