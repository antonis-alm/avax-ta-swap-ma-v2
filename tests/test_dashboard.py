from datetime import datetime, timedelta

import pandas as pd

from dashboard.ui import _build_dashboard_config, _enrich_ema_session_state, render_custom_dashboard


def _price_history(rows: int = 30) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    return pd.DataFrame(
        {
            "time": [start + timedelta(hours=i) for i in range(rows)],
            "price": [20 + (i * 0.5) for i in range(rows)],
        }
    )


def test_build_dashboard_config_reflects_ema_crossover_strategy() -> None:
    strategy_config = {
        "chain": "avalanche",
        "protocol": "traderjoe_v2",
        "base_token": "WAVAX",
        "quote_token": "USDC",
        "ema_fast_period": 5,
        "ema_slow_period": 10,
    }

    config = _build_dashboard_config(strategy_config)

    assert config.indicator_name == "EMA Crossover Momentum (AVAX/USDC)"
    assert config.signal_type == "momentum"
    assert config.indicator_period == 5
    assert config.secondary_periods == [10]
    assert config.chain == "avalanche"
    assert config.protocol == "traderjoe_v2"


def test_enrich_ema_session_state_adds_spread_series() -> None:
    strategy_config = {"ema_fast_period": 5, "ema_slow_period": 10, "base_token": "AVAX", "quote_token": "USDC"}
    config = _build_dashboard_config(strategy_config)
    session_state = {"price_history": _price_history(40)}

    enriched = _enrich_ema_session_state(session_state, strategy_config, config)

    indicator_key = config.indicator_name.lower()
    data_key = f"{indicator_key}_data"
    value_key = f"{indicator_key}_value"

    assert data_key in enriched
    assert value_key in enriched
    assert len(enriched[data_key]) == 40
    assert isinstance(enriched[value_key], float)


def test_render_custom_dashboard_wires_template(monkeypatch) -> None:
    captured = {}

    def fake_prepare(_api_client, session_state, config):
        captured["prepared_config"] = config
        return {**session_state, "price_history": _price_history(20)}

    def fake_render(strategy_id, strategy_config, session_state, config):
        captured["strategy_id"] = strategy_id
        captured["strategy_config"] = strategy_config
        captured["session_state"] = session_state
        captured["render_config"] = config

    monkeypatch.setattr("dashboard.ui.prepare_ta_session_state", fake_prepare)
    monkeypatch.setattr("dashboard.ui.render_ta_dashboard", fake_render)

    strategy_config = {
        "chain": "avalanche",
        "protocol": "traderjoe_v2",
        "base_token": "WAVAX",
        "quote_token": "USDC",
        "ema_fast_period": 5,
        "ema_slow_period": 10,
    }

    render_custom_dashboard("a_v_a_x_t_a_swap_m_a", strategy_config, api_client=object(), session_state={})

    assert captured["strategy_id"] == "a_v_a_x_t_a_swap_m_a"
    assert captured["render_config"].indicator_name == "EMA Crossover Momentum (AVAX/USDC)"
    assert captured["strategy_config"]["base_token"] == "AVAX"
    assert captured["strategy_config"]["chain"] == "Avalanche"
    assert captured["strategy_config"]["protocol"] == "TraderJoe V2"
    assert "ema crossover momentum (avax/usdc)_data" in captured["session_state"]
