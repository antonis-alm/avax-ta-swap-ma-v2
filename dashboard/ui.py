from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from almanak.framework.dashboard.templates import (
    TADashboardConfig,
    get_macd_config,
    prepare_ta_session_state,
    render_ta_dashboard,
)


def _ema(values: list[Decimal], period: int) -> list[Decimal]:
    if period <= 0 or not values:
        return []
    multiplier = Decimal("2") / Decimal(period + 1)
    ema_values = [values[0]]
    for value in values[1:]:
        ema_values.append((value * multiplier) + (ema_values[-1] * (Decimal("1") - multiplier)))
    return ema_values


def _display_base_token(base_token: str) -> str:
    if base_token.upper() == "WAVAX":
        return "AVAX"
    return base_token


def _build_dashboard_config(strategy_config: dict[str, Any]) -> TADashboardConfig:
    fast = int(strategy_config.get("ema_fast_period", 5))
    slow = int(strategy_config.get("ema_slow_period", 10))

    config = get_macd_config(fast=fast, slow=slow, signal=0)
    base_token = str(strategy_config.get("base_token", "AVAX"))
    quote_token = str(strategy_config.get("quote_token", "USDC"))
    display_base = _display_base_token(base_token)

    config.indicator_name = f"EMA Crossover Momentum ({display_base}/{quote_token})"
    config.indicator_period = fast
    config.secondary_periods = [slow]
    config.value_format = "{:+.4f}"
    config.value_suffix = " spread"
    config.custom_signal_fn = _ema_signal_text
    config.base_token = base_token
    config.quote_token = quote_token
    config.chain = str(strategy_config.get("chain", "avalanche"))
    config.protocol = str(strategy_config.get("protocol", "traderjoe_v2"))
    return config


def _ema_signal_text(session_state: dict[str, Any]) -> str:
    spread_key = next(
        (k for k in session_state.keys() if k.startswith("ema crossover momentum") and k.endswith("_value")),
        "",
    )
    spread = Decimal(str(session_state.get(spread_key, "0")))
    if spread > 0:
        return "BUY SIGNAL: Fast EMA is above slow EMA (bullish momentum)."
    if spread < 0:
        return "SELL SIGNAL: Fast EMA is below slow EMA (bearish momentum)."
    return "NEUTRAL: Fast and slow EMAs are equal (no crossover edge)."


def _enrich_ema_session_state(
    session_state: dict[str, Any],
    strategy_config: dict[str, Any],
    config: TADashboardConfig,
) -> dict[str, Any]:
    indicator_key = config.indicator_name.lower()
    data_key = f"{indicator_key}_data"
    value_key = f"{indicator_key}_value"
    signal_key = f"{indicator_key}_signal_{config.secondary_periods[0]}"

    if data_key in session_state and value_key in session_state:
        return session_state

    price_history = session_state.get("price_history")
    if not isinstance(price_history, pd.DataFrame) or price_history.empty:
        return session_state

    if "time" not in price_history.columns or "price" not in price_history.columns:
        return session_state

    prices = [Decimal(str(p)) for p in price_history["price"].tolist()]
    if len(prices) < max(config.indicator_period, config.secondary_periods[0]):
        return session_state

    fast_ema = _ema(prices, config.indicator_period)
    slow_ema = _ema(prices, config.secondary_periods[0])
    spread = [fast - slow for fast, slow in zip(fast_ema, slow_ema, strict=False)]

    series_data = list(zip(price_history["time"].tolist(), [float(value) for value in spread], strict=False))
    session_state[data_key] = series_data
    session_state[value_key] = float(spread[-1])
    session_state[signal_key] = float(slow_ema[-1])
    return session_state


def render_custom_dashboard(
    strategy_id: str,
    strategy_config: dict[str, Any],
    api_client: Any,
    session_state: dict[str, Any],
) -> None:
    config = _build_dashboard_config(strategy_config)

    session_state = prepare_ta_session_state(
        api_client,
        session_state=session_state,
        config=config,
    )
    session_state = _enrich_ema_session_state(session_state, strategy_config, config)
    session_state.setdefault("base_token", config.base_token)
    session_state.setdefault("quote_token", config.quote_token)

    display_strategy_config = dict(strategy_config)
    display_strategy_config["base_token"] = _display_base_token(config.base_token)
    display_strategy_config["quote_token"] = config.quote_token
    display_strategy_config["chain"] = "Avalanche"
    display_strategy_config["protocol"] = "TraderJoe V2"

    render_ta_dashboard(strategy_id, display_strategy_config, session_state, config)
