BEGIN;

ALTER TABLE binance_stocks_paper.operator_limits
  ADD COLUMN IF NOT EXISTS daily_loss_limit NUMERIC NOT NULL DEFAULT 200;

UPDATE binance_stocks_paper.operator_limits
SET max_order_notional = 500,
    max_symbol_exposure = 1000,
    max_managed_exposure = 2000,
    daily_loss_limit = 200,
    updated_at = now()
WHERE singleton = TRUE;

UPDATE binance_stocks_paper.symbol_whitelist
SET max_position_notional = 1000,
    updated_at = now()
WHERE symbol IN ('AAPL', 'TSLA', 'SPY', 'QQQ');

ALTER TABLE binance_stocks_paper.runtime_state
  ADD COLUMN IF NOT EXISTS market_phase TEXT,
  ADD COLUMN IF NOT EXISTS market_phase_source TEXT,
  ADD COLUMN IF NOT EXISTS market_event_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS market_valid_until TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS market_trading_date TEXT,
  ADD COLUMN IF NOT EXISTS market_state_conflict BOOLEAN NOT NULL DEFAULT FALSE;

COMMIT;
