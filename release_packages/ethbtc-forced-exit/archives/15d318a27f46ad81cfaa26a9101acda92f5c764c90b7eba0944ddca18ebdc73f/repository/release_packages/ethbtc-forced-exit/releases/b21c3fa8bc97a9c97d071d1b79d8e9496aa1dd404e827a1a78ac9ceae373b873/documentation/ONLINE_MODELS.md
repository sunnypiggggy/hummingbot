# OCI 线上模型现状

> 权威范围：本文记录 2026-08-09 完成 v22 切换后的生产结构。动态信号、订单和
> 盈亏必须以 OCI 运行合同、Guard 状态和 Binance 事实为准，不能只看本文快照。

## 结论

- Grid 与 DCA 当前唯一技术风控模型是 `ethbtc-forced-exit` v22；v21 producer 已关闭，
  ROC/SQZMOM 不再作为独立线上 gate，也不是故障回退。
- `grid-live-guard` 是唯一模型 producer，模式为 `live`；Grid 直接消费 FDUSD 信号，
  `dca-live-guard` 只读消费并映射到 USDT，不加载第二份模型。
- v22 只决定普通 BUY 是否允许。健康 Risk-Off 由 `forced-exit-v2` 覆盖层执行撤单和
  归属库存退出；完整性故障 Fail-Closed，退出后进入 `LATCHED`。
- 七类风控和自动重入已开启。最终普通 BUY 是所有已启用门的逻辑 AND。

## 当前生产身份

| 项目 | 当前值 |
|---|---|
| package | `ethbtc-forced-exit` |
| 模型版本 | `xgboost-grid-long-risk-gate-v22-weekly-250d` |
| 执行策略 | `v22-risk-off-forced-exit-v2` |
| 合同 schema | `ethbtc-forced-exit-live-contract-v1` |
| release SHA-256 | `73f59befa431946889a8d5885d04a05adb43c8e81eeab604f1aa89e31f0e9d60` |
| model SHA-256 | `fe487b3b0c8556154d7148583709762e576205059ffc7f7878718730af7fd1a6` |
| feature SHA-256 | `1fdc99293e83bd00b68b18174f3dc4f854f5b972c294f8c64e13ad26e8498da1` |
| strategy SHA-256 | `5de7c72a35911bbaad43fe342a068cb80a276c2b145ded8bfce79bd5eb1f29cb` |
| training data SHA-256 | `106273d229f279bb74ad135bc8296e49696316b812920e6a5e12ab12ef91beb7` |
| 当前签名周 | fold 37 |
| 当前周结束 | `2026-08-09 15:00 UTC` / `2026-08-09 23:00 北京时间` |
| 合同刷新/失效 | 约30秒刷新；150秒失效 |

冻结候选里的 `source_offline_verdict=NO-GO`、`deployment_allowed=false` 保持不可变。
生产执行权来自额外的哈希绑定审批回执和运行时完整性计算，不是修改冻结 release
获得。此次授权记录约 19小时58分观察，24小时的“时长条件”由操作人显式豁免；
其余观察检查和账户预检未豁免。

## 模型输入与更新周期

- BTC、ETH 使用各自的 `BTC-FDUSD`、`ETH-FDUSD` 模型输入和 fold-local 阈值。
- 模型和阈值每7天生成一个连续签名周；不能继承前周阈值。
- 风险概率在完整1小时K线后更新，4小时结构用于进入/恢复确认。
- 周切换保持状态连续，不重置持仓、累计盈亏、权益峰值或恢复阶段。
- 缺周、重叠周、哈希错误、合同过期、交易对缺失或授权错误均 Fail-Closed；禁止
  回退上一周、v21、ROC 或 SQZMOM。

当前快照中的概率和阈值仅用于说明字段，不应写成长期固定参数：

| 来源交易对 | 概率 | 当周 entry threshold | 当前状态 |
|---|---:|---:|---|
| `BTC-FDUSD` | `0.0397472233` | `0.0389975905` | Risk-On |
| `ETH-FDUSD` | `0.0431283377` | `0.1048762053` | Risk-On |

概率超过阈值不等于立即退出；进入与恢复还受 v22 冻结状态机的连续确认、4小时结构
和跨周连续状态约束。详见 [V22_WEEKLY_MODEL.md](V22_WEEKLY_MODEL.md)。

## Grid 实盘

- 机器人：`grid-live-fdusd-400`。
- 交易对：`BTC-FDUSD`、`ETH-FDUSD`，每对风险预算200 FDUSD。
- 当前网格：6%区间、10层、0.6%止盈、挂单最长2小时。
- 普通 Maker 费用模型为0%；强制退出/重入按 Taker 0.1%及2bp不利滑点审计。
- Grid 直接读取 `data/xgboost_risk_gate.json`；合同最大年龄150秒。
- 单对亏损6 FDUSD、单对回撤3%；组合亏损24 FDUSD、组合回撤6%。
- 自动重入开启。持仓保护仍可能在 v22 Risk-On 时限制 BUY，例如额外基础币已经
  达到10 FDUSD上限时；所以“技术门放行”不等于“一定同时存在 BUY 挂单”。

配置中的 `active_parameter_version` 可能保留历史参数搜索的 ROC/SQZ 字样。它只是
网格参数血缘标签，不表示 ROC/SQZMOM 仍是线上技术 gate。

## DCA 实盘

| 机器人 | 成交市场 | v22 信号来源 |
|---|---|---|
| `dca-live-btcusdt-200` | `BTC-USDT` | `BTC-FDUSD` |
| `dca-live-ethusdt-200` | `ETH-USDT` | `ETH-FDUSD` |

`dca-live-guard` 是 DCA controller gate 唯一写入者，聚合 v22、FOMC、策略/组合
亏损与回撤、持仓保护及恢复阶段。DCA 使用 USDT 行情和成交记账，但不改变 v22 的
FDUSD 特征输入。普通策略保留2%止盈、5%止损和5小时 executor 周期。

## 风控与恢复

当前 Grid/DCA 的七类机制均开启：

1. `v22_weekly_buy_gate`
2. `fomc_gate`
3. `strategy_loss_breaker`
4. `strategy_drawdown_breaker`
5. `portfolio_loss_breaker`
6. `portfolio_drawdown_breaker`
7. `position_protection`

恢复状态为 `ACTIVE → EXITING → COOLDOWN → REENTRY → ACTIVE`。模型缺失、合同过期、
哈希错误、授权错误和监控完整性故障在退出后进入 `LATCHED`，必须人工复核解锁。
任一机制恢复都不能覆盖其他仍关闭的门。Grid 和 DCA 自动重入均已开启。

## 每次排查必须读取的真相源

1. Binance 实际余额、活动订单和成交。
2. Grid `live_grid_runtime_state.json` 的逐对 `halted`、恢复阶段和组合状态。
3. DCA `guard_state.json`、controller 实际 BUY/SELL 值和 executor。
4. `xgboost_risk_gate.json` 的 schema、哈希、年龄、授权、逐对事件ID和 Risk-Off。
5. Grid/DCA Guard 的健康、完整性错误和紧急通道状态。

文档、Telegram、Plotly 和 observer 状态都属于审计层，不能单独证明当前允许交易。

