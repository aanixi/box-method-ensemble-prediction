"""
Vaihe 3: Piirteiden rakentaminen Box Method -setupeille.

Tämä moduuli lisää setups.parquet:in dataan ennustepiirteitä,
jotka muodostavat ML-mallien syöteavaruuden Vaiheessa 4.

Piirteet on jaoteltu ryhmiin (A–E), joista jokainen lisätään
omalla funktiollaan add_group_a_features() jne. Tämä helpottaa
inkrementaalista kehitystä ja sanity-tarkistuksia.
"""

import numpy as np
import pandas as pd


def add_group_a_features(setups: pd.DataFrame) -> pd.DataFrame:
    """
    Lisää Ryhmä A:n piirteet (setup-piirteet) setups-DataFrameen.

    Lisättävät piirteet:
        - box_size_pct: boxin koko prosentteina entry_price:stä
        - entry_position_in_box: entry_price:n etäisyys box_mid:stä,
          normalisoitu boxin koolla. Arvoalue noin -1..1.
        - hour_sin, hour_cos: entry_time:n tunti syklisesti koodattuna.
          Sin/cos-koodaus säilyttää syklisen luonteen (klo 23 ja klo 0
          ovat lähellä toisiaan).

    HUOM: setup_index_in_day, direction ja risk_reward_ratio ovat jo
    valmiina sarakkeina, joten niitä ei lisätä — käytetään suoraan
    olemassa olevia arvoja.

    Parametrit:
        setups: Vaiheen 2 tuottama DataFrame (setups.parquet).

    Palauttaa:
        DataFrame, jossa uudet sarakkeet lisättynä. Alkuperäinen ei muutu.
    """
    # Tee kopio jotta alkuperäinen DataFrame ei muutu kutsuvan puolelle
    df = setups.copy()

    # --- 1. box_size_pct (POISTETTU) ---
    # Aiemmassa versiossa laskimme box_size_pct:n, mutta korrelaatioanalyysi
    # paljasti, että se korreloi 0,81 piirteen box_size_vs_atr14d kanssa
    # (Ryhmä B). Molemmat olivat vahvoja monotonisia ennustepiirteitä, mutta
    # box_size_vs_atr14d on teoreettisesti perustellumpi (huomioi 14 päivän
    # historiallisen volatiliteetin). Poistettu päällekkäisyyden takia.
    
    # --- 2. entry_position_in_box ---
    # Kuinka kaukana entry_price on boxin keskiviivasta, normalisoituna
    # boxin puolikkaaseen. Lähellä 0 = keskellä, ±1 = boxin reunalla,
    # > 1 tai < -1 = sisäänmeno tapahtui boxin ulkopuolella (boxi "rikki").
    box_size = df["box_high"] - df["box_low"]
    raw_position = (df["entry_price"] - df["box_mid"]) / (box_size / 2)

    # Lippu-piirre: oliko boxi jo rikki sisäänmenohetkellä?
    # Tämä on usein vahva varoitussignaali — boxi ei enää kuvaa
    # nykyistä hintatasoa (esim. uutispohjainen breakout).
    df["entry_outside_box"] = (raw_position.abs() > 1).astype(int)

    # Clip ääriarvot välille [-3, 3] jotta puut ja muut mallit oppivat
    # paremmin. Kaikki yli ±3 tarkoittaa joka tapauksessa "boxi on
    # selvästi rikki", joten ero 3:n ja 13:n välillä ei tuo lisätietoa.
    df["entry_position_in_box"] = raw_position.clip(-3, 3)

    # --- 3. hour_of_day -> sin/cos -koodaus ---
    # Tunti on syklinen muuttuja: 23 ja 0 ovat "vieretysten", mutta
    # numerot 23 ja 0 ovat kaukana toisistaan. sin/cos-koodaus korjaa tämän.
    hour = df["entry_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return df

def add_group_b_features(setups: pd.DataFrame, daily_data: dict) -> pd.DataFrame:
    """
    Lisää Ryhmä B:n piirteet (boxin konteksti) setups-DataFrameen.

    Lisättävät piirteet:
        - prev_day_range_pct: edellisen päivän kynttilän koko prosentteina
          edellisen päivän close-hinnasta. Volatiliteettiproxy.
        - box_size_vs_atr14d: tämän päivän boxin koko jaettuna 14 päivän
          ATR:llä. Normalisoi boxin koon yleiseen volatiliteettiin.

    Look-ahead-bias: kaikki päivätason laskut käyttävät vain niitä
    päivätason kynttilöitä, jotka ovat päättyneet ennen setupin entry_time:ä.
    Käytännössä tämä tarkoittaa: setupin "date"-päivää (boxin määräytymispäivää)
    edeltäviä päiväkynttilöitä.

    Parametrit:
        setups: DataFrame setupeista (vähintään symbol, date, box_high, box_low).
        daily_data: dict, jossa avaimena symboli (esim. "ETHUSDT") ja
          arvona päivätason DataFrame (open, high, low, close jne., indeksinä open_time).

    Palauttaa:
        DataFrame, jossa uudet sarakkeet lisättynä.
    """
    df = setups.copy()

    # Esiprosessoi päivädatat: lasketaan tarvittavat sarakkeet kerran per symboli
    prepared = {}
    for symbol, daily in daily_data.items():
        d = daily.copy()
        # Varmistetaan että data on aikajärjestyksessä
        d = d.sort_index()

        # True Range (TR) = max kolmesta:
        #   1) high - low
        #   2) |high - prev_close|
        #   3) |low - prev_close|
        prev_close = d["close"].shift(1)
        tr1 = d["high"] - d["low"]
        tr2 = (d["high"] - prev_close).abs()
        tr3 = (d["low"] - prev_close).abs()
        d["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR(14) = TR:n liukuva keskiarvo 14 päivän yli.
        # Käytetään yksinkertaista keskiarvoa (SMA). Wilder käytti alun perin
        # tasoittavaa keskiarvoa, mutta SMA on tässä yhtä hyvä.
        d["atr14"] = d["tr"].rolling(window=14, min_periods=14).mean()

        # Päivän range prosentteina close:sta
        d["range_pct"] = (d["high"] - d["low"]) / d["close"] * 100

        prepared[symbol] = d

    # Apufunktio: hae edellisen päivän tietoa setupin "date":n perusteella
    def get_prev_day_value(row, column):
        symbol = row["symbol"]
        # Setupin "date" on boxin muodostumispäivä (eli ETM-päivä). Boxi tehdään
        # edellisen päivän perusteella. Eli "edellinen päivä" on (date - 1 päivä).
        # Mutta tässä haetaan boxin muodostumispäivän data — se on jo päättynyt
        # ennen setupin entry_time:ä, joten se on oikein.
        d = prepared[symbol]
        # Tehdään aika UTC:nä, tunnit nollattu, jotta osuu päivätason indeksiin
        target_date = row["date"].normalize()
        if target_date in d.index:
            return d.loc[target_date, column]
        return np.nan

    # Lasketaan piirteet riveittäin
    df["prev_day_range_pct"] = df.apply(lambda r: get_prev_day_value(r, "range_pct"), axis=1)

    box_size = df["box_high"] - df["box_low"]
    atr14_values = df.apply(lambda r: get_prev_day_value(r, "atr14"), axis=1)
    df["box_size_vs_atr14d"] = box_size / atr14_values

    return df

def add_group_c_features(setups_df, df_4h):
    """
    Lisää Ryhmä C:n piirteet: multi-timeframe trendi 4h-aikakehyksestä.
    
    Parametrit:
        setups_df: DataFrame setupeista, jossa pakollinen sarake 'entry_time' (UTC, tz-aware)
        df_4h: DataFrame 4h-kynttilöistä, indeksinä open_time (UTC, tz-aware),
               vähintään sarakkeet 'close' ja 'close_time'
    
    Lisättävät piirteet:
        ema50_4h: 4h EMA(50) close-hinnasta (aputieto, ei välttämättä mukaan ML:ään)
        ema200_4h: 4h EMA(200) close-hinnasta (aputieto)
        trend_4h: (EMA50 - EMA200) / EMA200, jatkuva trendin voimakkuusmitta
                  Positiivinen = nouseva trendi, negatiivinen = laskeva
        trend_aligned: 1 jos setupin direction on trend_4h:n kanssa samansuuntainen,
                       0 muuten (long+nouseva tai short+laskeva = 1)
    
    Look-ahead-suojaus: jokaiselle setupille käytetään viimeisintä 4h-kynttilää,
    jonka close_time <= entry_time. Eli kynttilä on jo sulkeutunut entry-hetkellä.
    """
    setups = setups_df.copy()
    
    # 1) Laske EMA:t koko 4h-datasta. EMA on rekursiivinen, mutta pandas
    #    hoitaa sen oikein: ewm(span=N, adjust=False) vastaa klassista
    #    EMA-kaavaa alpha = 2/(N+1).
    df = df_4h.sort_index().copy()
    df['ema50_4h'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200_4h'] = df['close'].ewm(span=200, adjust=False).mean()
    df['trend_4h'] = (df['ema50_4h'] - df['ema200_4h']) / df['ema200_4h']
    
    # 2) Asetetaan aikaleima jolla mergeataan: kynttilän close_time.
    #    Käytetään merge_asof:ia: jokaiselle setupille etsitään suurin
    #    close_time joka on <= entry_time. Tämä takaa look-ahead-suojan.
    trend_data = df[['close_time', 'ema50_4h', 'ema200_4h', 'trend_4h']].copy()
    trend_data = trend_data.sort_values('close_time')
    
    setups_sorted = setups.sort_values('entry_time').copy()
    
    merged = pd.merge_asof(
        setups_sorted,
        trend_data,
        left_on='entry_time',
        right_on='close_time',
        direction='backward'  # viimeisin close_time joka <= entry_time
    )
    
    # 3) Trend_aligned -lippu: long+positiivinen trendi tai short+negatiivinen
    merged['trend_aligned'] = (
        ((merged['direction'] == 'long') & (merged['trend_4h'] > 0)) |
        ((merged['direction'] == 'short') & (merged['trend_4h'] < 0))
    ).astype(int)
    
    # 4) Palautetaan alkuperäiseen järjestykseen (merge_asof vaati lajittelun)
    merged = merged.sort_index() if merged.index.is_monotonic_increasing else merged
    
    return merged

def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Laske Average True Range (ATR) Wilderin smoothed moving average -menetelmällä.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = Wilderin tasoitus TR:stä periodilla N (= ewm alpha=1/N, adjust=False).

    Parametrit
    ----------
    df : pd.DataFrame
        OHLC-data sarakkeilla 'high', 'low', 'close'. Indeksin tulee olla
        aikajärjestyksessä.
    period : int
        ATR-periodi (oletus 14).

    Palauttaa
    ---------
    pd.Series
        ATR-arvot samalla indeksillä kuin df.
    """
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()

    return atr

def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Laske Relative Strength Index (RSI) Wilderin tasoituksella.

    Tämä on TradingView-yhteensopiva laskenta (sama kuin pandas-ta:n oletus).

    Parametrit
    ----------
    close : pd.Series
        Sulkemishintojen sarja aikajärjestyksessä.
    period : int
        RSI-periodi (oletus 14).

    Palauttaa
    ---------
    pd.Series
        RSI-arvot välillä 0–100, samalla indeksillä kuin close.
    """
    delta = close.diff()

    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def add_group_d_features(
    setups_df: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
) -> pd.DataFrame:
    """
    Lisää Ryhmä D:n piirteet (volatiliteetti ja momentum) setupeihin.

    Lisättävät piirteet (yhteensä 7):
        ATR:
        - atr_15m_pct: ATR(14, 15m) / close, prosenttiosuus hinnasta
        - atr_15m_vs_atr_daily: ATR(14, 15m) / ATR(14, 1d)
        - atr_1h_pct: ATR(14, 1h) / close
        - atr_1h_vs_atr_daily: ATR(14, 1h) / ATR(14, 1d)

        RSI:
        - rsi_15m: RSI(14, 15m), Wilder, 0–100
        - rsi_1h: RSI(14, 1h), Wilder, 0–100

        Volyymi:
        - volume_vs_ma20_15m: volume / SMA(volume, 20) 15m-aikakehyksessä

    Look-ahead-suojaus: kaikki piirteet poimitaan viimeisimmästä valmiista
    kynttilästä ennen setupin entry_time:ä, käyttäen merge_asof
    direction='backward' close_time:n perusteella.

    Parametrit
    ----------
    setups_df : pd.DataFrame
        Setupit, joissa on vähintään sarakkeet 'entry_time' ja 'symbol'.
    df_15m, df_1h, df_1d : pd.DataFrame
        OHLCV-data 15m, 1h ja 1d aikakehyksissä. Indeksinä open_time,
        sarakkeissa myös 'close_time'. Sarakkeet: open, high, low, close,
        volume, close_time.

    Palauttaa
    ---------
    pd.DataFrame
        setups_df kopiona, jossa 7 uutta saraketta.
    """
    result = setups_df.copy()
    result = result.drop(
        columns=['close_time', 'close_time_x', 'close_time_y'],
        errors='ignore',
    )
    # --- 1. Laske indikaattorit raakadatoille (symbolikohtaisesti) ---

    def _prepare_15m(df: pd.DataFrame) -> pd.DataFrame:
        """Laske ATR, RSI ja volyymisuhde 15m-datalle, symbolikohtaisesti."""
        out = []
        for symbol, group in df.groupby('symbol', sort=False):
            g = group.sort_values('open_time').copy()
            g['atr_15m'] = _calculate_atr(g, period=14)
            g['rsi_15m'] = _calculate_rsi(g['close'], period=14)
            g['volume_ma20'] = g['volume'].rolling(window=20, min_periods=20).mean()
            g['volume_vs_ma20_15m'] = g['volume'] / g['volume_ma20']
            out.append(g)
        return pd.concat(out, ignore_index=True)

    def _prepare_1h(df: pd.DataFrame) -> pd.DataFrame:
        """Laske ATR ja RSI 1h-datalle, symbolikohtaisesti."""
        out = []
        for symbol, group in df.groupby('symbol', sort=False):
            g = group.sort_values('open_time').copy()
            g['atr_1h'] = _calculate_atr(g, period=14)
            g['rsi_1h'] = _calculate_rsi(g['close'], period=14)
            out.append(g)
        return pd.concat(out, ignore_index=True)

    def _prepare_1d(df: pd.DataFrame) -> pd.DataFrame:
        """Laske ATR 1d-datalle, symbolikohtaisesti."""
        out = []
        for symbol, group in df.groupby('symbol', sort=False):
            g = group.sort_values('open_time').copy()
            g['atr_1d'] = _calculate_atr(g, period=14)
            out.append(g)
        return pd.concat(out, ignore_index=True)

    df_15m_prep = _prepare_15m(df_15m)
    df_1h_prep = _prepare_1h(df_1h)
    df_1d_prep = _prepare_1d(df_1d)

    # --- 2. Valmistele merge_asof:ia varten ---

    # Setupit järjestettävä entry_time:n mukaan merge_asof:ia varten
    result = result.sort_values('entry_time').reset_index(drop=True)

    # Raakadatat järjestettävä close_time:n mukaan
    df_15m_prep = df_15m_prep.sort_values('close_time').reset_index(drop=True)
    df_1h_prep = df_1h_prep.sort_values('close_time').reset_index(drop=True)
    df_1d_prep = df_1d_prep.sort_values('close_time').reset_index(drop=True)

    # --- 3. Yhdistä 15m-piirteet ---

    result = pd.merge_asof(
        result,
        df_15m_prep[['close_time', 'symbol', 'atr_15m', 'rsi_15m',
                     'volume_vs_ma20_15m']],
        left_on='entry_time',
        right_on='close_time',
        by='symbol',
        direction='backward',
    )
    result = result.drop(
    columns=['close_time', 'close_time_x', 'close_time_y'],
    errors='ignore',
    )

    # --- 4. Yhdistä 1h-piirteet ---

    result = pd.merge_asof(
        result,
        df_1h_prep[['close_time', 'symbol', 'atr_1h', 'rsi_1h']],
        left_on='entry_time',
        right_on='close_time',
        by='symbol',
        direction='backward',
    )
    result = result.drop(
    columns=['close_time', 'close_time_x', 'close_time_y'],
    errors='ignore',
    )
    # --- 5. Yhdistä 1d-ATR ---

    result = pd.merge_asof(
        result,
        df_1d_prep[['close_time', 'symbol', 'atr_1d']],
        left_on='entry_time',
        right_on='close_time',
        by='symbol',
        direction='backward',
    )
    result = result.drop(
    columns=['close_time', 'close_time_x', 'close_time_y'],
    errors='ignore',
    )

    # --- 6. Johda lopulliset normalisoidut piirteet ---
    # HUOM: alkuperäisestä 7 piirteestä karsittiin 3 korkean korrelaation
    # vuoksi (ks. Ryhmä D löytödokumentti).
    # Karsittu: atr_15m_pct (r=0.89 atr_1h_pct), atr_1h_vs_atr_daily (r=0.86
    # atr_15m_vs_atr_daily), rsi_15m (r=0.89 rsi_1h).

    result['atr_15m_vs_atr_daily'] = result['atr_15m'] / result['atr_1d']
    result['atr_1h_pct'] = result['atr_1h'] / result['entry_price']

    # --- 7. Pudota rsi_15m (karsittu) ja välilaskennan sarakkeet ---

    result = result.drop(
        columns=['atr_15m', 'atr_1h', 'atr_1d', 'rsi_15m'],
        errors='ignore',
    )

    return result

# =============================================================================
# RYHMÄ E: KÄÄNNEKYNTTILÄN RAKENNE
# =============================================================================

# =============================================================================
# RYHMÄ E: KÄÄNNEKYNTTILÄN RAKENNE
# =============================================================================

def add_group_e_features(
    setups_df: pd.DataFrame,
    df_5m: pd.DataFrame,
) -> pd.DataFrame:
    """
    Lisää Ryhmä E:n piirteet setup-DataFrame:en.

    Piirteet:
    - reversal_candle_size_pct: käännekynttilän koko (high-low) / entry_price
    - minutes_in_opposite_half: aika (minuuteissa) jonka hinta vietti
      "vastakkaisessa" puoliskossa päivän alusta käännekynttilään asti
      (long: close < box_mid; short: close > box_mid)

    Look-ahead-suojaus:
    - Käännekynttilän high/low haetaan tarkasti reversal_candle_time:n perusteella
    - Vastakkaisen puoliskon laskenta tehdään vain käännekynttilään asti

    Args:
        setups_df: setup-DataFrame, jossa pitää olla sarake 'reversal_candle_time'
        df_5m: 5min-kynttilät, indeksinä open_time (UTC), sarakkeena 'symbol'

    Returns:
        DataFrame jossa Ryhmä E:n piirteet lisättyinä
    """
    setups = setups_df.copy()

    # ----- 1. Käännekynttilän koko (high-low) -----
    # Mergeä 5m-kynttilän high ja low käännekynttilän aikaleimaan
    df_5m_subset = df_5m[['symbol', 'high', 'low']].copy()
    df_5m_subset = df_5m_subset.reset_index().rename(
        columns={'open_time': 'reversal_candle_time',
                 'high': 'reversal_high',
                 'low': 'reversal_low'}
    )

    setups = setups.merge(
        df_5m_subset,
        on=['symbol', 'reversal_candle_time'],
        how='left',
    )

    # Sanity: kaikilla pitäisi nyt olla high ja low
    missing = setups['reversal_high'].isna().sum()
    if missing > 0:
        print(f"VAROITUS: {missing} setupia joilta puuttuu käännekynttilän high/low")

    # Lasketaan piirre 1
    setups['reversal_candle_size_pct'] = (
        (setups['reversal_high'] - setups['reversal_low']) / setups['entry_price']
    )

    # ----- 2. Aika vastakkaisessa puoliskossa -----
    # Jokaiselle setupille katsotaan saman päivän 5m-kynttilät päivän alusta
    # käännekynttilään asti, ja lasketaan kuinka monta niistä oli
    # "vastakkaisessa" puoliskossa.

    def _minutes_in_opposite_half(row, df_5m):
        symbol = row['symbol']
        direction = row['direction']
        box_mid = row['box_mid']
        reversal_time = row['reversal_candle_time']

        # Päivän alku UTC
        day_start = reversal_time.normalize()

        # Saman päivän 5m-kynttilät symbolille, päivän alusta käännekynttilään asti
        mask = (
            (df_5m['symbol'] == symbol)
            & (df_5m.index >= day_start)
            & (df_5m.index <= reversal_time)
        )
        day_candles = df_5m.loc[mask]

        if len(day_candles) == 0:
            return 0

        # Vastakkainen puolisko:
        # - long: close < box_mid (hinta oli alapuoliskossa)
        # - short: close > box_mid (hinta oli yläpuoliskossa)
        if direction == 'long':
            opposite_count = (day_candles['close'] < box_mid).sum()
        else:  # short
            opposite_count = (day_candles['close'] > box_mid).sum()

        return int(opposite_count) * 5  # minuuteiksi

    print("Lasketaan minutes_in_opposite_half-piirrettä (kestää muutaman sekunnin)...")
    setups['minutes_in_opposite_half'] = setups.apply(
        lambda row: _minutes_in_opposite_half(row, df_5m),
        axis=1,
    )

    # ----- Siivous -----
    setups = setups.drop(
        columns=['reversal_high', 'reversal_low'],
        errors='ignore',
    )

    return setups