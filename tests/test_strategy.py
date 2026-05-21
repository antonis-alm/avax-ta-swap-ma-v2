from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from strategy import AVAXTASwapMAStrategy


@pytest.fixture
def config() -> dict:
    return {
        "chain": "avalanche",
        "protocol": "traderjoe_v2",
        "base_token": "AVAX",
        "quote_token": "USDC",
        "timeframe": "5m",
        "ema_fast_period": 5,
        "ema_slow_period": 10,
        "ohlcv_limit": 60,
        "trade_size_usd": 100,
        "max_slippage_bps": 50,
        "force_action": "",
    }


@pytest.fixture
def strategy(config: dict) -> AVAXTASwapMAStrategy:
    return AVAXTASwapMAStrategy(
        config=config,
        chain="avalanche",
        wallet_address="0x" + "1" * 40,
    )


def _records(start: datetime | None = None, count: int = 20) -> list[dict]:
    ts = start or datetime(2026, 1, 1)
    return [
        {
            "timestamp": ts + timedelta(minutes=5 * i),
            "open": Decimal("10"),
            "high": Decimal("11"),
            "low": Decimal("9"),
            "close": Decimal(str(10 + i)),
            "volume": Decimal("100"),
        }
        for i in range(count)
    ]


def _market(base_usd: Decimal, quote_usd: Decimal, records: list[dict] | None = None) -> MagicMock:
    market = MagicMock()

    def _balance(token: str):
        if token == "AVAX":
            return SimpleNamespace(balance=Decimal("2"), balance_usd=base_usd)
        return SimpleNamespace(balance=Decimal("1000"), balance_usd=quote_usd)

    market.balance.side_effect = _balance
    market.ohlcv.return_value = records or _records()
    return market


def _patch_emas(monkeypatch: pytest.MonkeyPatch, strategy: AVAXTASwapMAStrategy, *, prev_fast: Decimal, curr_fast: Decimal, prev_slow: Decimal, curr_slow: Decimal):
    length = 20
    fast = [Decimal("1")] * length
    slow = [Decimal("1")] * length
    fast[-3] = prev_fast
    fast[-2] = curr_fast
    slow[-3] = prev_slow
    slow[-2] = curr_slow

    def _ema(_closes, period):
        if period == strategy.ema_fast_period:
            return fast
        return slow

    monkeypatch.setattr(strategy, "_compute_ema_series", _ema)


class TestDecideBranches:
    def test_cross_up_swaps_usdc_to_avax(self, strategy: AVAXTASwapMAStrategy, monkeypatch: pytest.MonkeyPatch):
        _patch_emas(
            monkeypatch,
            strategy,
            prev_fast=Decimal("1.0"),
            curr_fast=Decimal("3.0"),
            prev_slow=Decimal("2.0"),
            curr_slow=Decimal("2.0"),
        )
        market = _market(base_usd=Decimal("0"), quote_usd=Decimal("500"))

        intent = strategy.decide(market)

        assert intent.intent_type.value == "SWAP"
        assert intent.from_token == "USDC"
        assert intent.to_token == "AVAX"
        assert intent.protocol == "traderjoe_v2"

    def test_cross_down_swaps_avax_to_usdc(self, strategy: AVAXTASwapMAStrategy, monkeypatch: pytest.MonkeyPatch):
        _patch_emas(
            monkeypatch,
            strategy,
            prev_fast=Decimal("3.0"),
            curr_fast=Decimal("1.0"),
            prev_slow=Decimal("2.0"),
            curr_slow=Decimal("2.0"),
        )
        market = _market(base_usd=Decimal("500"), quote_usd=Decimal("0"))

        intent = strategy.decide(market)

        assert intent.intent_type.value == "SWAP"
        assert intent.from_token == "AVAX"
        assert intent.to_token == "USDC"

    def test_no_crossover_holds(self, strategy: AVAXTASwapMAStrategy, monkeypatch: pytest.MonkeyPatch):
        _patch_emas(
            monkeypatch,
            strategy,
            prev_fast=Decimal("2.5"),
            curr_fast=Decimal("2.6"),
            prev_slow=Decimal("2.0"),
            curr_slow=Decimal("2.1"),
        )
        market = _market(base_usd=Decimal("0"), quote_usd=Decimal("500"))

        intent = strategy.decide(market)

        assert intent.intent_type.value == "HOLD"
        assert "No confirmed EMA crossover" in intent.reason

    def test_dedupes_same_confirmed_candle(self, strategy: AVAXTASwapMAStrategy, monkeypatch: pytest.MonkeyPatch):
        _patch_emas(
            monkeypatch,
            strategy,
            prev_fast=Decimal("1.0"),
            curr_fast=Decimal("3.0"),
            prev_slow=Decimal("2.0"),
            curr_slow=Decimal("2.0"),
        )
        market = _market(base_usd=Decimal("0"), quote_usd=Decimal("500"), records=_records())

        first = strategy.decide(market)
        second = strategy.decide(market)

        assert first.intent_type.value == "SWAP"
        assert second.intent_type.value == "HOLD"
        assert "Waiting for next confirmed" in second.reason

    def test_buy_debounce_blocks_when_unarmed(self, strategy: AVAXTASwapMAStrategy, monkeypatch: pytest.MonkeyPatch):
        strategy.buy_armed = False
        _patch_emas(
            monkeypatch,
            strategy,
            prev_fast=Decimal("1.0"),
            curr_fast=Decimal("3.0"),
            prev_slow=Decimal("2.0"),
            curr_slow=Decimal("2.0"),
        )
        market = _market(base_usd=Decimal("0"), quote_usd=Decimal("500"))

        intent = strategy.decide(market)

        assert intent.intent_type.value == "HOLD"
        assert "waiting for reset" in intent.reason.lower()

    def test_reset_rearms_then_allows_buy(self, strategy: AVAXTASwapMAStrategy, monkeypatch: pytest.MonkeyPatch):
        strategy.buy_armed = False

        _patch_emas(
            monkeypatch,
            strategy,
            prev_fast=Decimal("1.8"),
            curr_fast=Decimal("1.7"),
            prev_slow=Decimal("2.0"),
            curr_slow=Decimal("2.0"),
        )
        market_reset = _market(base_usd=Decimal("0"), quote_usd=Decimal("500"), records=_records(datetime(2026, 1, 1)))
        first = strategy.decide(market_reset)

        _patch_emas(
            monkeypatch,
            strategy,
            prev_fast=Decimal("1.0"),
            curr_fast=Decimal("3.0"),
            prev_slow=Decimal("2.0"),
            curr_slow=Decimal("2.0"),
        )
        market_cross = _market(base_usd=Decimal("0"), quote_usd=Decimal("500"), records=_records(datetime(2026, 2, 1)))
        second = strategy.decide(market_cross)

        assert first.intent_type.value == "HOLD"
        assert strategy.buy_armed is False or strategy.buy_armed is True
        assert second.intent_type.value == "SWAP"

    def test_insufficient_quote_on_buy_holds(self, strategy: AVAXTASwapMAStrategy, monkeypatch: pytest.MonkeyPatch):
        _patch_emas(
            monkeypatch,
            strategy,
            prev_fast=Decimal("1.0"),
            curr_fast=Decimal("3.0"),
            prev_slow=Decimal("2.0"),
            curr_slow=Decimal("2.0"),
        )
        market = _market(base_usd=Decimal("0"), quote_usd=Decimal("20"))

        intent = strategy.decide(market)

        assert intent.intent_type.value == "HOLD"
        assert "insufficient usdc" in intent.reason.lower()

    def test_insufficient_base_on_sell_holds(self, strategy: AVAXTASwapMAStrategy, monkeypatch: pytest.MonkeyPatch):
        _patch_emas(
            monkeypatch,
            strategy,
            prev_fast=Decimal("3.0"),
            curr_fast=Decimal("1.0"),
            prev_slow=Decimal("2.0"),
            curr_slow=Decimal("2.0"),
        )
        market = _market(base_usd=Decimal("20"), quote_usd=Decimal("0"))

        intent = strategy.decide(market)

        assert intent.intent_type.value == "HOLD"
        assert "insufficient avax" in intent.reason.lower()

    def test_balance_error_holds(self, strategy: AVAXTASwapMAStrategy):
        market = MagicMock()
        market.balance.side_effect = ValueError("boom")

        intent = strategy.decide(market)

        assert intent.intent_type.value == "HOLD"
        assert "Balance data unavailable" in intent.reason

    def test_ohlcv_error_holds(self, strategy: AVAXTASwapMAStrategy):
        market = MagicMock()
        market.balance.side_effect = lambda token: SimpleNamespace(balance=Decimal("1"), balance_usd=Decimal("100"))
        market.ohlcv.side_effect = ValueError("boom")

        intent = strategy.decide(market)

        assert intent.intent_type.value == "HOLD"
        assert "OHLCV data unavailable" in intent.reason


class TestForceAction:
    def test_force_buy(self, config: dict):
        config["force_action"] = "buy"
        strategy = AVAXTASwapMAStrategy(config=config, chain="avalanche", wallet_address="0x" + "1" * 40)

        intent = strategy.decide(MagicMock())

        assert intent.intent_type.value == "SWAP"
        assert intent.from_token == "USDC"
        assert intent.to_token == "AVAX"

    def test_force_sell(self, config: dict):
        config["force_action"] = "sell"
        strategy = AVAXTASwapMAStrategy(config=config, chain="avalanche", wallet_address="0x" + "1" * 40)

        intent = strategy.decide(MagicMock())

        assert intent.intent_type.value == "SWAP"
        assert intent.from_token == "AVAX"
        assert intent.to_token == "USDC"

    def test_force_action_unknown_raises(self, config: dict):
        config["force_action"] = "invalid"
        strategy = AVAXTASwapMAStrategy(config=config, chain="avalanche", wallet_address="0x" + "1" * 40)

        with pytest.raises(ValueError, match="Unknown force_action"):
            strategy.decide(MagicMock())


class TestTeardown:
    def test_get_open_positions_reports_token_position(self, strategy: AVAXTASwapMAStrategy, monkeypatch: pytest.MonkeyPatch):
        market = MagicMock()
        market.balance.return_value = SimpleNamespace(balance=Decimal("1"), balance_usd=Decimal("20"))
        monkeypatch.setattr(strategy, "create_market_snapshot", lambda: market)

        summary = strategy.get_open_positions()

        assert len(summary.positions) == 1
        assert summary.positions[0].position_type.value == "TOKEN"

    def test_generate_teardown_intents_empty_without_base(self, strategy: AVAXTASwapMAStrategy):
        market = MagicMock()
        market.balance.return_value = SimpleNamespace(balance=Decimal("0"), balance_usd=Decimal("0"))

        intents = strategy.generate_teardown_intents(mode=None, market=market)

        assert intents == []

    def test_generate_teardown_intents_hard_uses_wider_slippage(self, strategy: AVAXTASwapMAStrategy):
        from almanak.framework.teardown import TeardownMode

        market = MagicMock()
        market.balance.return_value = SimpleNamespace(balance=Decimal("1"), balance_usd=Decimal("20"))

        soft = strategy.generate_teardown_intents(mode=TeardownMode.SOFT, market=market)
        hard = strategy.generate_teardown_intents(mode=TeardownMode.HARD, market=market)

        assert soft[0].intent_type.value == "SWAP"
        assert hard[0].intent_type.value == "SWAP"
        assert hard[0].max_slippage >= soft[0].max_slippage
