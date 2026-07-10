# Walk-forward Portfolio Grid Result

## Overall
- Period: 2026-02-07 03:15:00 to 2026-07-09 03:15:00
- Start equity: 10000.00 USDT
- Final equity: 10070.54 USDT
- Net PnL: 70.54 USDT (0.71%)
- Max weekly-window drawdown: -4.18%
- Total grid moves: 767
- Total liquidations: 0

## Best Realized Parameter Combination
- grid_range: 0.06
- grid_levels: 12
- order_quote_pct: 0.005
- take_profit: 0.005
- move_threshold: 0.005
- stop_loss: 0.04
- weeks_used: 8
- realized_net_pnl: 72.26 USDT (0.70%)
- mean_weekly_pnl: 0.09%
- max_drawdown: -2.88%
- win_rate: 75.00%
- realized_score: -0.042306

## Best Candidate Combo From Training Windows
- rank: 1
- grid_range: 0.06
- grid_levels: 8
- order_quote_pct: 0.005
- take_profit: 0.005
- move_threshold: 0.015
- stop_loss: 0.04
- windows_evaluated: 22
- avg_selection_score: -0.005203
- compound_training_pnl: -0.72%
- avg_training_pnl: -0.03%
- worst_training_drawdown: -3.94%
- win_rate: 81.82%
- liquidation_count: 0

## Robust Recommendation
This combination was selected from parameter groups used in at least two trading weeks, then ranked by realized return with drawdown as the tie breaker.
- grid_range: 0.06
- grid_levels: 8
- order_quote_pct: 0.005
- take_profit: 0.005
- move_threshold: 0.015
- stop_loss: 0.04
- weeks_used: 5
- realized_net_pnl: 152.26 USDT (1.54%)
- mean_weekly_pnl: 0.31%
- max_drawdown: -4.11%
- win_rate: 80.00%

## Best and Worst Weeks
- Best week: #21, 2026-06-27 03:15:00 to 2026-07-04 03:15:00, PnL 1.48%, moves 24
- Worst week: #17, 2026-05-30 03:15:00 to 2026-06-06 03:15:00, PnL -3.01%, moves 75

## Grid Move Leaders
- ADA-USDT: total 114, up 53, down 61
- SOL-USDT: total 103, up 50, down 53
- DOGE-USDT: total 101, up 46, down 55
- AVAX-USDT: total 100, up 47, down 53
- LINK-USDT: total 89, up 43, down 46
- XRP-USDT: total 77, up 35, down 42
- ETH-USDT: total 76, up 37, down 39
- BNB-USDT: total 49, up 22, down 27
- BTC-USDT: total 48, up 22, down 26
- TRX-USDT: total 10, up 6, down 4
