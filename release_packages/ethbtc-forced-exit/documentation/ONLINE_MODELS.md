# OCI 线上模型现状

> 审计快照：2026-09-02。本文只记录核查时点的生产身份和解释口径；动态概率、订单、权益及门控必须读取 OCI 当前 generation、运行合同、Guard 状态和 Binance 事实，不得把本文快照当作实时状态。

## 结论

- Grid 与 DCA 唯一线上技术 BUY 风控模型是 `ethbtc-forced-exit` v22；v21 producer 已关闭，ROC/SQZMOM 不参与线上权限计算，也不是故障回退。
- `grid-live-guard` 是唯一 v22 producer。Grid 消费 `BTC-FDUSD`、`ETH-FDUSD`；DCA 只读消费同一合同并映射为 `BTC-USDT`、`ETH-USDT`，不重复加载模型。
- v22 模型信号、模型不可用和执行恢复阶段是三种不同状态：健康 `RISK_OFF` 是模型判断；`UNAVAILABLE` 是完整性失败；`EXITING/COOLDOWN/REENTRY/LATCHED` 是执行覆盖层。
- v22 只决定普通 BUY 门。Risk-Off 后的撤单、归属库存退出和恢复由 `forced-exit-v2` 执行覆盖层完成；完整性故障完成退出后进入 `LATCHED`。

## 2026-09-02 生产身份快照

| 项目 | 核查值 |
|---|---|
| package | `ethbtc-forced-exit` |
| 模型版本 | `xgboost-grid-long-risk-gate-v22-weekly-250d` |
| 执行策略 | `v22-risk-off-forced-exit-v2` |
| 合同 schema | `ethbtc-forced-exit-live-contract-v1` |
| runtime pointer schema | `ethbtc-forced-exit-runtime-pointer-v1` |
| release SHA-256 | `bc3ef0d97bad6fbfaa6e24db1d695defd69ffffaf514a800f280e166bf7e017c` |
| runtime generation | `292e3801…`，完整值以 `runtime/current.json` 为准 |
| model SHA-256 | `6185e768…`，完整值以当前合同为准 |
| feature SHA-256 | `1fdc99293e83bd00b68b18174f3dc4f854f5b972c294f8c64e13ad26e8498da1` |
| strategy SHA-256 | `6c2afa89b1fa6d2838062871fd7222da95c1bef2ce4599a525dbee98d6b7fd5e` |
| training data SHA-256 | `b8ff6f5a…`，完整值以当前合同为准 |
| 当前签名周 | fold 41 |
| 当前周范围 | `2026-08-30 15:00 → 2026-09-06 15:00 UTC` |
| 北京时间周范围 | `2026-08-30 23:00 → 2026-09-06 23:00` |
| cutover phase | `ACTIVE` |
| 合同刷新/失效 | 约30秒刷新；150秒失效 |

冻结离线候选的 `source_offline_verdict=NO-GO`、`offline_only=true` 和 `deployment_allowed=false` 不得改写。生产权限来自额外的哈希绑定审批、当前周覆盖和运行时完整性验证。

## 模型输入与周度更新

- BTC、ETH 分别使用 `BTC-FDUSD`、`ETH-FDUSD` 特征与 fold-local 阈值；DCA 的 USDT 成交市场不改变模型输入分布。
- 每7天生成一个连续签名周；当前周和下一周在边界前完成候选、证据、审批及 generation 预热。
- 风险概率在完整1小时 K 线后更新，4小时结构用于进入/恢复确认；周边界不重置既有 Risk-Off、累计盈亏、权益峰值或恢复阶段。
- 健康切换使用同一个已提交 runtime generation。临时文件、展示别名或 Scheduler 边界登记不能制造 `UNAVAILABLE`。
- 缺周、重叠周、哈希错误、合同过期、交易对缺失或授权错误才是真正的 Fail-Closed；不得回退上一周、v21、ROC 或 SQZMOM。

核查时概率仅用于证明合同字段有效，不是永久参数：

| 来源交易对 | 概率 | fold-local entry threshold | 模型状态 |
|---|---:|---:|---|
| `BTC-FDUSD` | `0.0381008089` | `0.0392840914` | `RISK_ON` |
| `ETH-FDUSD` | `0.0383108929` | `0.0391260386` | `RISK_OFF` |

概率与阈值不能脱离冻结状态机直接解释为交易动作；连续确认、4小时结构、当前状态和执行覆盖层仍参与最终结果。详见 [V22_WEEKLY_MODEL.md](V22_WEEKLY_MODEL.md) 与 [V22_ZERO_DOWNTIME_CUTOVER.md](V22_ZERO_DOWNTIME_CUTOVER.md)。

## Grid 实盘参数快照

机器人为 `grid-live-fdusd-400`，每对资金边界200 FDUSD；参数合同版本为 `binance-ai-btc-medium-sideways-eth-long-volatility-v1`。

| 交易对 | profile | 总范围 | 理论总格数 | 止盈 | 最小订单 |
|---|---|---:|---:|---:|---:|
| `BTC-FDUSD` | `medium_sideways` | `12.6984%` | 18 | `0.4000%` | `10 FDUSD` |
| `ETH-FDUSD` | `long_volatility` | `52.4651%` | 18 | `1.4180%` | `10 FDUSD` |

- 理论18格只是候选价格拓扑，不承诺9张 BUY 加9张 SELL。
- BUY 受可用资金、每侧预算、组合余量、额外库存额度、动态精度和最低金额共同裁剪；合法预算不足时可以是0张。
- SELL 受策略归属库存、启动库存上限、交易所余额和移动平均成本利润底线约束。多个逻辑 SELL 层落到同一执行价时合并为一张订单，数量相加但不增加总库存风险。
- 2026-09-02 时点审计为 BTC `0 BUY / 4 SELL`、ETH `0 / 0`：BTC 的额外库存额度只剩约0.225 FDUSD，低于10 FDUSD最低单额；ETH 为 v22 Risk-Off，属于 `EXPECTED_EMPTY`。这不是固定订单配置。
- 普通 Grid 始终 Maker-only；风险退出/重入按 Taker 费用和不利滑点审计，禁止使用 BNB 抵扣手续费。

订单形成细节见 [GRID_PAIR_PARAMETER_CUTOVER.md](GRID_PAIR_PARAMETER_CUTOVER.md)。

## DCA 实盘映射

| 机器人 | 成交市场 | v22 信号来源 |
|---|---|---|
| `dca-live-btcusdt-200` | `BTC-USDT` | `BTC-FDUSD` |
| `dca-live-ethusdt-200` | `ETH-USDT` | `ETH-FDUSD` |

`dca-live-guard` 是 DCA controller gate 唯一写入者。DCA 使用 USDT 行情和成交记账；普通策略的止盈、止损和 executor 周期不改变 v22 模型。资金观察状态为 `alert_only`、`enforced=false`，只告警，不参与普通交易权限 AND。

## 当前机制边界

七类风险机制、基础设施/完整性、恢复覆盖层、库存归属、资金观察、订单执行和费用规则的唯一权威定义在 [RISK_MECHANISMS.md](RISK_MECHANISMS.md)。任一机制恢复都不能覆盖其他仍关闭的门。

## 排查真相源优先级

1. Binance 实际余额、活动订单与成交：经济事实。
2. 统一库存合同与 SQLite：资金归属、订单生命周期和恢复证据。
3. 当前 `runtime/current.json` generation 内的 v22 合同和 GateState：模型、哈希、授权、状态血缘。
4. Grid Runtime State schema 13、DCA `guard_state.json` 与 controller 实际值：当前现金、策略归属库存、重入意图、预期订单和落地权限。
5. Guard/Report 健康及运行错误事件：解释系统状态。

Telegram、PNG、Plotly 和本文都是审计展示，不能单独证明当前允许交易。
