import logging
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from almanak.framework.intents import Intent
from almanak.framework.market import MarketSnapshot
from almanak.framework.strategies import IntentStrategy, almanak_strategy

logger = logging.getLogger(__name__)


def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, datetime | date):
        return v.isoformat()
    if isinstance(v, Enum):
        return getattr(v, "value", str(v))
    return v


@almanak_strategy(
    name="a_v_a_x_t_a_swap_m_a",
    description="EMA crossover swap strategy on TraderJoe V2 (AVAX/USDC, 5m candles)",
    version="1.0.0",
    author="Generated",
    tags=["generated", "ema", "momentum", "traderjoe_v2"],
    supported_chains=["avalanche"],
    supported_protocols=["traderjoe_v2"],
    intent_types=["SWAP", "HOLD"],
    default_chain="avalanche",
)
class AVAXTASwapMAStrategy(IntentStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.protocol = self.get_config("protocol", "traderjoe_v2")
        self.base_token = self.get_config("base_token", "AVAX")
        self.quote_token = self.get_config("quote_token", "USDC")

        self.timeframe = self.get_config("timeframe", "5m")
        self.ema_fast_period = int(self.get_config("ema_fast_period", 5))
        self.ema_slow_period = int(self.get_config("ema_slow_period", 10))
        self.ohlcv_limit = int(self.get_config("ohlcv_limit", 60))

        self.trade_size_usd = Decimal(str(self.get_config("trade_size_usd", "100")))
        self.max_slippage_bps = int(self.get_config("max_slippage_bps", 50))
        self.force_action = str(self.get_config("force_action", "") or "").lower()

        self.buy_armed = True
        self.sell_armed = True
        self.last_signal_direction: str | None = None
        self.last_processed_candle_ts: str | None = None

    def _timestamp_to_str(self, ts: Any) -> str:
        if isinstance(ts, datetime):
            return ts.isoformat()
        if hasattr(ts, "isoformat"):
            return ts.isoformat()
        return str(ts)

    def _extract_closes_and_timestamps(self, candles: Any) -> tuple[list[Decimal], list[Any]]:
        records: list[dict[str, Any]]
        if isinstance(candles, list):
            records = candles
        elif hasattr(candles, "to_dict"):
            records = candles.to_dict("records")
        else:
            raise ValueError("Unsupported OHLCV shape")

        closes: list[Decimal] = []
        timestamps: list[Any] = []
        for row in records:
            if "close" not in row:
                raise ValueError("OHLCV missing close")
            if "timestamp" not in row:
                raise ValueError("OHLCV missing timestamp")
            closes.append(Decimal(str(row["close"])))
            timestamps.append(row["timestamp"])

        return closes, timestamps

    def _compute_ema_series(self, closes: list[Decimal], period: int) -> list[Decimal]:
        if period <= 0:
            raise ValueError("EMA period must be positive")
        if not closes:
            raise ValueError("No closes to compute EMA")

        k = Decimal("2") / Decimal(period + 1)
        ema_values = [closes[0]]
        for price in closes[1:]:
            ema_values.append((price * k) + (ema_values[-1] * (Decimal("1") - k)))
        return ema_values

    def _forced_intent(self) -> Intent:
        max_slippage = Decimal(str(self.max_slippage_bps)) / Decimal("10000")
        if self.force_action == "buy":
            return Intent.swap(
                from_token=self.quote_token,
                to_token=self.base_token,
                amount_usd=self.trade_size_usd,
                max_slippage=max_slippage,
                protocol=self.protocol,
                chain=self.chain,
            )
        if self.force_action == "sell":
            return Intent.swap(
                from_token=self.base_token,
                to_token=self.quote_token,
                amount_usd=self.trade_size_usd,
                max_slippage=max_slippage,
                protocol=self.protocol,
                chain=self.chain,
            )
        raise ValueError(f"Unknown force_action: {self.force_action!r}")

    def decide(self, market: MarketSnapshot) -> Optional[Intent]:
        if self.force_action:
            return self._forced_intent()

        try:
            quote_balance = market.balance(self.quote_token)
            base_balance = market.balance(self.base_token)
        except ValueError:
            return Intent.hold(reason="Balance data unavailable")

        try:
            candles = market.ohlcv(self.base_token, timeframe=self.timeframe, limit=self.ohlcv_limit)
            closes, timestamps = self._extract_closes_and_timestamps(candles)
        except ValueError:
            return Intent.hold(reason="OHLCV data unavailable")

        min_required = max(self.ema_slow_period + 3, 4)
        if len(closes) < min_required:
            return Intent.hold(reason="Insufficient candles for EMA crossover")

        ema_fast = self._compute_ema_series(closes, self.ema_fast_period)
        ema_slow = self._compute_ema_series(closes, self.ema_slow_period)

        prev_fast = ema_fast[-3]
        prev_slow = ema_slow[-3]
        curr_fast = ema_fast[-2]
        curr_slow = ema_slow[-2]

        confirmed_ts = self._timestamp_to_str(timestamps[-2])
        if self.last_processed_candle_ts == confirmed_ts:
            return Intent.hold(reason="Waiting for next confirmed 5m candle close")

        if curr_fast <= curr_slow:
            self.buy_armed = True
        if curr_fast >= curr_slow:
            self.sell_armed = True

        is_cross_up = prev_fast <= prev_slow and curr_fast > curr_slow
        is_cross_down = prev_fast >= prev_slow and curr_fast < curr_slow

        max_slippage = Decimal(str(self.max_slippage_bps)) / Decimal("10000")

        if is_cross_up:
            self.last_processed_candle_ts = confirmed_ts
            if not self.buy_armed:
                return Intent.hold(reason="Buy crossover ignored: waiting for reset")
            if quote_balance.balance_usd < self.trade_size_usd:
                return Intent.hold(reason="Buy crossover but insufficient USDC")
            if base_balance.balance_usd >= self.trade_size_usd:
                return Intent.hold(reason="Buy crossover but already holding AVAX")

            self.buy_armed = False
            self.last_signal_direction = "buy"
            return Intent.swap(
                from_token=self.quote_token,
                to_token=self.base_token,
                amount_usd=self.trade_size_usd,
                max_slippage=max_slippage,
                protocol=self.protocol,
                chain=self.chain,
            )

        if is_cross_down:
            self.last_processed_candle_ts = confirmed_ts
            if not self.sell_armed:
                return Intent.hold(reason="Sell crossover ignored: waiting for reset")
            if base_balance.balance_usd < self.trade_size_usd:
                return Intent.hold(reason="Sell crossover but insufficient AVAX")

            self.sell_armed = False
            self.last_signal_direction = "sell"
            return Intent.swap(
                from_token=self.base_token,
                to_token=self.quote_token,
                amount_usd=self.trade_size_usd,
                max_slippage=max_slippage,
                protocol=self.protocol,
                chain=self.chain,
            )

        self.last_processed_candle_ts = confirmed_ts
        return Intent.hold(reason="No confirmed EMA crossover")

    def get_status(self) -> dict[str, Any]:
        return {
            "strategy": "a_v_a_x_t_a_swap_m_a",
            "chain": self.chain,
            "wallet": self.wallet_address[:10] + "..." if self.wallet_address else None,
            "base_token": self.base_token,
            "quote_token": self.quote_token,
            "timeframe": self.timeframe,
            "buy_armed": self.buy_armed,
            "sell_armed": self.sell_armed,
            "last_signal_direction": _safe(self.last_signal_direction),
            "last_processed_candle_ts": _safe(self.last_processed_candle_ts),
        }

    def get_persistent_state(self):
        return {
            "buy_armed": self.buy_armed,
            "sell_armed": self.sell_armed,
            "last_signal_direction": self.last_signal_direction,
            "last_processed_candle_ts": self.last_processed_candle_ts,
        }

    def load_persistent_state(self, state):
        if not state:
            return
        self.buy_armed = bool(state.get("buy_armed", True))
        self.sell_armed = bool(state.get("sell_armed", True))
        self.last_signal_direction = state.get("last_signal_direction")
        self.last_processed_candle_ts = state.get("last_processed_candle_ts")

    def get_open_positions(self):
        from almanak.framework.teardown import PositionInfo, PositionType, TeardownPositionSummary

        positions = []
        try:
            market = self.create_market_snapshot()
            base_balance = market.balance(self.base_token)
        except ValueError:
            base_balance = None

        if base_balance and base_balance.balance > 0:
            positions.append(
                PositionInfo(
                    position_type=PositionType.TOKEN,
                    position_id="a_v_a_x_t_a_swap_m_a_base_token",
                    chain=self.chain,
                    protocol=self.protocol,
                    value_usd=base_balance.balance_usd,
                    details={"asset": self.base_token, "quote": self.quote_token, "balance": str(base_balance.balance)},
                )
            )

        return TeardownPositionSummary(
            strategy_id=getattr(self, "strategy_id", "a_v_a_x_t_a_swap_m_a"),
            timestamp=datetime.now(UTC),
            positions=positions,
        )

    def generate_teardown_intents(self, mode=None, market=None) -> list[Intent]:
        from almanak.framework.teardown import TeardownMode

        snapshot = market
        if snapshot is None:
            try:
                snapshot = self.create_market_snapshot()
            except ValueError:
                snapshot = None

        if snapshot is not None:
            try:
                base_balance = snapshot.balance(self.base_token)
                if base_balance.balance <= 0:
                    return []
            except ValueError:
                return []

        max_slippage = Decimal("0.03") if mode == TeardownMode.HARD else Decimal(str(self.max_slippage_bps)) / Decimal("10000")
        return [
            Intent.swap(
                from_token=self.base_token,
                to_token=self.quote_token,
                amount="all",
                max_slippage=max_slippage,
                protocol=self.protocol,
                chain=self.chain,
            )
        ]
