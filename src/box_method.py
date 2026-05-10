"""
Box Method -setupien tunnistaminen ja labelointi.

Algoritmi (long-puoli, short on peilikuva):
1. Käytetään edellisen UTC-päivän high/low boxin määrittelyyn
2. Boxin keskiviiva = (high + low) / 2
3. Käydään 5min-kynttilät läpi päivän aikana:
   - Odotetaan että hinta sulkeutuu yli/alle keskiviivan → "watching"-tila
   - Etsitään käännekynttilä: ensimmäinen vihreä punaisten ketjun jälkeen
   - Sisäänmeno kun seuraava kynttilä sulkee käännekynttilän huipun yli (long)
4. Take profit = boxin yläraja (long), stop loss = sisäänmenokynttilän low
5. Tulos: TP-osuma, SL-osuma, tai expired (päivä loppui)
6. Useita setupeja per päivä sallittu, mutta ei päällekkäisiä
"""

from dataclasses import dataclass, asdict
from typing import Literal, Optional

import pandas as pd


@dataclass
class Setup:
    """Yksittäinen Box-setup ja sen tulos."""
    setup_id: str
    symbol: str
    date: pd.Timestamp           # UTC-päivä jolloin setup tapahtui
    direction: Literal["long", "short"]
    setup_index_in_day: int      # 1, 2, 3, ... järjestyksessä päivän sisällä

    # Boxin parametrit
    box_high: float
    box_low: float
    box_mid: float

    # Sisäänmeno
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float     # (TP - entry) / (entry - SL)

    # Tulos
    outcome_time: Optional[pd.Timestamp]
    outcome: Literal["tp", "sl", "expired"]
    label: Optional[int]         # 1 = voitto, 0 = tappio, None = expired


def _is_green(candle: pd.Series) -> bool:
    """Vihreä kynttilä = close > open."""
    return candle["close"] > candle["open"]


def _is_red(candle: pd.Series) -> bool:
    """Punainen kynttilä = close < open."""
    return candle["close"] < candle["open"]


def find_setups_for_day(
    day_5m: pd.DataFrame,
    box_high: float,
    box_low: float,
    symbol: str,
    date: pd.Timestamp,
) -> list[Setup]:
    """
    Etsii kaikki Box-setupit yhdelle päivälle.

    Args:
        day_5m: 5min-kynttilät yhdeltä UTC-päivältä (indeksinä open_time)
        box_high, box_low: edellisen päivän high ja low
        symbol: esim. "ETHUSDT"
        date: päivä (UTC)

    Returns:
        Lista löydettyjä Setup-objekteja (sisältää sekä avoimet että suljetut)
    """
    box_mid = (box_high + box_low) / 2
    setups: list[Setup] = []

    # Tila-automaatti
    state = "waiting"  # waiting / watching_lower / watching_upper / in_trade
    pending_pivot = None  # käännekynttilä (Series), kun sellainen löytyy
    prev_candle = None
    active_setup: Optional[Setup] = None

    setup_counter = 0  # juokseva indeksi päivän sisällä

    for ts, candle in day_5m.iterrows():

        # ---- Jos kauppa on auki, seuraa tulosta ----
        if state == "in_trade" and active_setup is not None:
            tp = active_setup.take_profit
            sl = active_setup.stop_loss

            hit_tp = (candle["high"] >= tp) if active_setup.direction == "long" \
                     else (candle["low"] <= tp)
            hit_sl = (candle["low"] <= sl) if active_setup.direction == "long" \
                     else (candle["high"] >= sl)

            if hit_tp and hit_sl:
                # Konservatiivinen: SL ensin
                active_setup.outcome = "sl"
                active_setup.outcome_time = ts
                active_setup.label = 0
                setups.append(active_setup)
                active_setup = None
                state = "waiting"
                pending_pivot = None
            elif hit_sl:
                active_setup.outcome = "sl"
                active_setup.outcome_time = ts
                active_setup.label = 0
                setups.append(active_setup)
                active_setup = None
                state = "waiting"
                pending_pivot = None
            elif hit_tp:
                active_setup.outcome = "tp"
                active_setup.outcome_time = ts
                active_setup.label = 1
                setups.append(active_setup)
                active_setup = None
                state = "waiting"
                pending_pivot = None

            prev_candle = candle
            continue

        # ---- Jos odotamme suuntaa, tarkista tuleeko close yli tai ali keskiviivan ----
        if state == "waiting":
            if candle["close"] < box_mid:
                state = "watching_lower"
                pending_pivot = None
            elif candle["close"] > box_mid:
                state = "watching_upper"
                pending_pivot = None

        # ---- Etsi käännekynttilä ja sisäänmeno alapuoliskossa (long) ----
        if state == "watching_lower":
            # Päivitä käännekynttilä jos edellinen oli punainen ja tämä vihreä
            if prev_candle is not None and _is_red(prev_candle) and _is_green(candle):
                pending_pivot = candle

            # Sisäänmenoehto: tämä kynttilä sulkee käännekynttilän huipun yläpuolelle
            # HUOM: käännekynttilä ei voi olla sama kuin sisäänmenokynttilä,
            # koska kynttilä ei voi sulkeutua oman huippunsa yläpuolelle.
            if pending_pivot is not None and pending_pivot.name != ts:
                if candle["close"] > pending_pivot["high"]:
                    setup_counter += 1
                    entry_price = candle["close"]
                    sl = pending_pivot["low"] - entry_price * 0.0015  # pieni bufferi SL:ään
                    tp = box_high
                    rr = (tp - entry_price) / (entry_price - sl) if entry_price > sl else float("nan")

                    active_setup = Setup(
                        setup_id=f"{symbol}_{date.date()}_{setup_counter}_long",
                        symbol=symbol,
                        date=date,
                        direction="long",
                        setup_index_in_day=setup_counter,
                        box_high=box_high,
                        box_low=box_low,
                        box_mid=box_mid,
                        entry_time=ts,
                        entry_price=entry_price,
                        stop_loss=sl,
                        take_profit=tp,
                        risk_reward_ratio=rr,
                        outcome_time=None,
                        outcome="expired",  # päivitetään jos osuu
                        label=None,
                    )
                    state = "in_trade"
                    pending_pivot = None

        # ---- Etsi käännekynttilä ja sisäänmeno yläpuoliskossa (short) ----
        elif state == "watching_upper":
            if prev_candle is not None and _is_green(prev_candle) and _is_red(candle):
                pending_pivot = candle

            if pending_pivot is not None and pending_pivot.name != ts:
                if candle["close"] < pending_pivot["low"]:
                    setup_counter += 1
                    entry_price = candle["close"]
                    sl = pending_pivot["high"] + entry_price * 0.0015  # pieni bufferi SL:ään
                    tp = box_low
                    rr = (entry_price - tp) / (sl - entry_price) if sl > entry_price else float("nan")

                    active_setup = Setup(
                        setup_id=f"{symbol}_{date.date()}_{setup_counter}_short",
                        symbol=symbol,
                        date=date,
                        direction="short",
                        setup_index_in_day=setup_counter,
                        box_high=box_high,
                        box_low=box_low,
                        box_mid=box_mid,
                        entry_time=ts,
                        entry_price=entry_price,
                        stop_loss=sl,
                        take_profit=tp,
                        risk_reward_ratio=rr,
                        outcome_time=None,
                        outcome="expired",
                        label=None,
                    )
                    state = "in_trade"
                    pending_pivot = None

        # ---- Tilan vaihto: jos hinta palaa keskiviivan toiselle puolelle ----
        if state == "watching_lower" and candle["close"] > box_mid:
            state = "watching_upper"
            pending_pivot = None
        elif state == "watching_upper" and candle["close"] < box_mid:
            state = "watching_lower"
            pending_pivot = None

        prev_candle = candle

    # ---- Päivä loppui, kauppa vielä auki → expired ----
    if active_setup is not None:
        active_setup.outcome = "expired"
        active_setup.outcome_time = None
        active_setup.label = None
        setups.append(active_setup)

    return setups


def find_all_setups(
    df_5m: pd.DataFrame,
    df_1d: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Etsii kaikki Box-setupit koko datasetistä.

    Args:
        df_5m: 5min-kynttilät, indeksinä open_time (UTC)
        df_1d: päivä-kynttilät, indeksinä open_time (UTC)
        symbol: esim. "ETHUSDT"

    Returns:
        DataFrame, jossa jokainen rivi on yksi setup
    """
    all_setups: list[Setup] = []

    # Käydään päivät läpi, alkaen toisesta päivästä (tarvitsemme edellisen)
    for i in range(1, len(df_1d)):
        prev_day = df_1d.iloc[i - 1]
        current_day = df_1d.iloc[i]

        date = current_day.name  # UTC-päivä
        box_high = float(prev_day["high"])
        box_low = float(prev_day["low"])

        # Poimi vain tämän päivän 5min-kynttilät
        day_start = date
        day_end = date + pd.Timedelta(days=1)
        day_5m = df_5m.loc[(df_5m.index >= day_start) & (df_5m.index < day_end)]

        if len(day_5m) == 0:
            continue

        day_setups = find_setups_for_day(
            day_5m=day_5m,
            box_high=box_high,
            box_low=box_low,
            symbol=symbol,
            date=date,
        )
        all_setups.extend(day_setups)

    if not all_setups:
        return pd.DataFrame()

    return pd.DataFrame([asdict(s) for s in all_setups])