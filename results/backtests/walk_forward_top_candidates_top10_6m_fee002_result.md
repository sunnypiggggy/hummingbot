# Walk-forward Portfolio Grid Result

## Overall
- Period: 2026-02-07 03:15:00 to 2026-07-09 03:15:00
- Start equity: 10000.00 USDT
- Final equity: 10052.67 USDT
- Net PnL: 52.67 USDT (0.53%)
- Max weekly-window drawdown: -2.23%
- Total grid moves: 346
- Total liquidations: 0

## Best Realized Parameter Combination
- grid_range: 0.08
- grid_levels: 8
- order_quote_pct: 0.005
- take_profit: 0.005
- move_threshold: 0.015
- stop_loss: 0.04
- weeks_used: 1
- realized_net_pnl: 2.14 USDT (0.02%)
- mean_weekly_pnl: 0.02%
- max_drawdown: -0.11%
- win_rate: 100.00%
- realized_score: -0.001409

## Best Candidate Combo From Training Windows
- rank: 1
- grid_range: 0.1
- grid_levels: 8
- order_quote_pct: 0.005
- take_profit: 0.005
- move_threshold: 0.005
- stop_loss: 0.04
- windows_evaluated: 22
- avg_selection_score: -0.003063
- compound_training_pnl: -0.27%
- avg_training_pnl: -0.01%
- worst_training_drawdown: -2.44%
- win_rate: 86.36%
- liquidation_count: 0

## Robust Recommendation
This combination was selected from parameter groups used in at least two trading weeks, then ranked by realized return with drawdown as the tie breaker.
- grid_range: 0.08
- grid_levels: 8
- order_quote_pct: 0.005
- take_profit: 0.005
- move_threshold: 0.005
- stop_loss: 0.04
- weeks_used: 3
- realized_net_pnl: 34.74 USDT (0.34%)
- mean_weekly_pnl: 0.11%
- max_drawdown: -0.45%
- win_rate: 100.00%

## Best and Worst Weeks
- Best week: #21, 2026-06-27 03:15:00 to 2026-07-04 03:15:00, PnL 0.78%, moves 11
- Worst week: #17, 2026-05-30 03:15:00 to 2026-06-06 03:15:00, PnL -1.66%, moves 48

## Grid Move Leaders
- ADA-USDT: total 57, up 26, down 31
- AVAX-USDT: total 47, up 21, down 26
- SOL-USDT: total 45, up 22, down 23
- XRP-USDT: total 43, up 20, down 23
- DOGE-USDT: total 40, up 18, down 22
- LINK-USDT: total 37, up 18, down 19
- ETH-USDT: total 36, up 17, down 19
- BNB-USDT: total 19, up 10, down 9
- BTC-USDT: total 18, up 9, down 9
- TRX-USDT: total 4, up 2, down 2
