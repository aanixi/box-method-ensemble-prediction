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