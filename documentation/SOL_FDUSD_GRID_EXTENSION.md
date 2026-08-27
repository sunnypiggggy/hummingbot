# SOL-FDUSD Grid 扩展与首次上线门禁

## 当前结论

SOL 支持代码和离线证据已经与 BTC/ETH 运行路径隔离，但首次实盘仍保持关闭。
只有完成以下事务后才允许生成 SOL 订单：选择 `short_sideways` 或
`medium_sideways`、Telegram 三张 PNG 全部送达、写入与候选和参数哈希绑定的
首次人工审批、账户新增至少 200 FDUSD、建立并复核约 100 FDUSD 的 SOL 启动
库存、最后原子启用三交易对参数合同。

BTC 继续使用 `medium_sideways`，ETH 继续使用 `long_volatility`；它们的 v22
合同和 DCA 的 FDUSD→USDT 映射均不包含 SOL。

## 容器和信号链路

```text
grid-live-fdusd-scheduler（同一现有服务）
  ├─ BTC/ETH 参数合同与 v22 周更新（原逻辑）
  ├─ SOL 参数候选与三张回测 PNG 证据
  └─ 首次人工批准后发布 schema-v3 active_selection

grid-live-guard（不新增容器）
  ├─ BTC/ETH：消费 ethbtc-forced-exit v22 合同
  ├─ SOL兼容路径：只消费 sol-grid-weekly-risk-v1 独立合同（当前关闭）
  └─ 将合同写入现有 Grid 实例数据目录

grid-live-fdusd-400（同一交易容器）
  ├─ BTC-FDUSD：中短期横盘
  ├─ ETH-FDUSD：长期波动
  └─ SOL-FDUSD：首次批准后选择短期或中短期横盘

dca-live-guard
  └─ 仍只消费 BTC/ETH v22；不加载、不映射、不交易 SOL

dca-live-report
  └─ 首次候选发送 SOL 360天、1–2月、5–6月三张手机 PNG；激活后增加第五张 SOL 四小时卡片
```

## 参数

| 参数 | SOL 短期横盘 | SOL 中短期横盘 |
|---|---:|---:|
| 总范围 | 7.6956699692% | 12.6983794754% |
| 中心上下范围 | ±3.8478349846% | ±6.3491897377% |
| 总格数 | 18 | 18 |
| BUY / SELL 理论层数 | 9 / 9 | 9 / 9 |
| 止盈 | 0.4% | 0.4% |
| 每侧预算 | 100 FDUSD | 100 FDUSD |
| 最低订单 | 10 FDUSD | 10 FDUSD |
| 移动阈值 / 冷却 | 1.5% / 30分钟 | 1.5% / 30分钟 |
| 刷新 | 2小时 | 2小时 |

运行时继续读取 Binance `PRICE_FILTER`、`LOT_SIZE`、`MARKET_LOT_SIZE` 和
`MIN_NOTIONAL/NOTIONAL`。BUY 采用向上数量量化并删最远层控制预算；SELL 向下
量化并按 Grid 归属库存裁剪。普通订单只能是 Maker。

## 资金与风险边界

- 两交易对兼容模式仍是 420 FDUSD，不部署、不设置环境变量就不会改变现网。
- 三交易对模式是 620 FDUSD：BTC/ETH/SOL 各 200，另有 20 储备。
- SOL 的 200 FDUSD 是外部注资；迁移时组合峰值和周期基准同步加 200，累计盈亏不变。
- 单对亏损阈值 6 FDUSD、单对回撤 3%、组合亏损 36 FDUSD、组合回撤 6%。
- SOL Risk-Off 或 SOL 合同失败只限制和退出 SOL；不能污染 BTC、ETH 或 DCA。
- SOL 合同不存在、过期、哈希错误或周覆盖缺失时是 `UNAVAILABLE`，不是普通
  `RISK_OFF`，并且禁止 SOL 新增 BUY。

## 统一库存归属

统一 SQLite 合同升级到 v4，增加 SOL 资产。SOL 只能由
`grid:grid-live-fdusd-400` 归属，DCA 永远不会生成 SOL 归属。兼容代码部署但 SOL
未激活时，账户中的既有 SOL 标记为 `QUARANTINED_PRE_ACTIVATION`，DCA Guard 不得
把它识别为无归属库存或自动卖成 USDT。首次激活必须先写入 SOL 启动库存归属，
再打开执行开关。

## 离线证据口径

- 窗口：`2025-08-31 00:00 → 2026-08-26 00:00 UTC`。
- 每对 103,680 根连续 5 分钟 K 线；无重复、缺口或 OHLCV 异常。
- 最新研究模型为 `xgboost-sol-grid-long-risk-gate-v22-weekly-360d`：53个自然周
  各自重新拟合 XGBoost 权重和早停树数，使用最近14天成熟校准样本生成该周
  98.5% 分位阈值，状态跨周连续。
- 标签是72小时持续下跌事件，成熟延迟96小时；所有周均满足最晚成熟标签时间
  不晚于训练截止时间。模型使用15项 SOL-only 特征，不读取 BTC/ETH 行情、模型、
  阈值或状态。
- 参数纯效果关闭技术门和熔断；`v22_only` 只加入 SOL-v22 Risk-Off 强制退出；
  `protected` 再叠加亏损/回撤、强制退出和自动恢复。FOMC 因无可信历史事件不参与。
- Maker 0%、风险退出 Taker 0.1% 加 2bp 不利滑点；不使用 BNB 付费。
- 图和统计用于人工判断，不自动产生首次实盘授权。
- 当前本地私有费率预检没有包含 SOL 且已过期，因此回测费率是对照假设，
  `activation_eligible=false`。上线前必须用实际账户重新完成包含 SOL-FDUSD 的
  手续费预检；PNG 可以先发送，但人工批准也不能绕过该硬门槛。

证据目录：`results/backtests/sol_fdusd_binance_ai_profiles_360d/`。

## 本次 360 天结果（独立SOL-v22）

| 口径 | SOL 参数 | SOL 净收益 | SOL 最大回撤 | 强制退出 | 受限小时 |
|---|---|---:|---:|---:|---:|
| 参数纯效果 | 短期横盘 | -91.0526 FDUSD | -61.8475% | 0 | 0 |
| 仅 SOL-v22 | 短期横盘 | -32.3629 FDUSD | -45.8123% | 13 | 1,640 |
| SOL-v22 + 现行熔断 | 短期横盘 | -72.8751 FDUSD | -52.6819% | 97 | 2,292.42 |
| 参数纯效果 | 中短期横盘 | -78.8947 FDUSD | -59.8201% | 0 | 0 |
| 仅 SOL-v22 | 中短期横盘 | +0.0359 FDUSD | -32.3887% | 13 | 1,640 |
| SOL-v22 + 现行熔断 | 中短期横盘 | -8.1398 FDUSD | -34.0635% | 72 | 2,070.17 |

53个周模型具有53个不同模型哈希，共生成8,640个连续小时样本外信号，Risk-Off
1,640小时、进入/恢复各13次。整体 OOS ROC-AUC 为0.5486，Average Precision
为0.1213（阳性率0.1021），模型区分能力偏弱；即使中短期横盘的v22-only收益
接近保本，最大回撤仍达32.39%，因此结论保持 `NO-GO`，不得据此授权实盘。

当前运行兼容合同 `sol-grid-weekly-risk-v1` 与这份离线v22模型包是不同身份；在
另行完成生产封包、合同适配、费率核验和首次人工审批前，不能将离线模型哈希写入
实盘合同，也不能打开 `GRID_SOL_FDUSD_LIVE_ENABLED`。

## 首次激活顺序

1. 部署兼容代码，保持 `GRID_SOL_FDUSD_LIVE_ENABLED=false`。
2. 影子刷新 SOL 合同至少 24 小时，确认只影响 SOL 且合同连续。
3. 频道送达三张哈希绑定 PNG，报告服务生成交付回执。
4. 人工选择短期或中短期横盘，写入一次性 `sol-grid-initial-approval-v1` 回执。
5. 核验账户新增可用 FDUSD 不少于 200；禁止出售其他策略库存筹资。
6. 市价建立约 100 FDUSD SOL，复核成交、费用、余额和归属 SQLite。
7. 原子启用 schema-v3 参数与 SOL 周门；只有全部门放行才挂普通订单。
8. 检查 10 分钟、2 小时和 24 小时。失败只限制 SOL，不回退或清仓 BTC/ETH。

## 回滚

撤销 SOL 执行授权并取消 SOL 普通订单，复核 SOL Grid 归属库存；保持 BTC/ETH 的
参数、v22、账本、累计盈亏和恢复阶段不变。不能通过关闭完整性门绕过无有效 SOL
周模型的 Fail-Closed。
