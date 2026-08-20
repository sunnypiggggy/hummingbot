# Grid 逐交易对参数合同与切换

## 当前生产选择

参数版本固定为 `binance-ai-btc-medium-sideways-eth-long-volatility-v1`，版本名不含时间戳；
审计与幂等以整个 `active_selection.json` 的规范化 SHA-256 为准。

| 交易对 | Profile | 总范围 | 格数 | 相邻格距 | 普通止盈 | 普通订单下限 |
|---|---|---:|---:|---:|---:|---:|
| BTC-FDUSD | `medium_sideways` | 12.698379% | 18（9/9） | 约0.746964% | 0.400000% | 10 FDUSD |
| ETH-FDUSD | `long_volatility` | 52.465116% | 18（9/9） | 约3.086183% | 1.417976% | 10 FDUSD |

两对仍各使用200 FDUSD，BUY侧和启动库存侧各以100 FDUSD为边界；组合另保留20 FDUSD。
Maker手续费为0。强制退出仍使用生产Taker费用与滑点模型，不使用BNB支付手续费。

移动语义保持现网实现：只有价格先越过当前Grid边界，再向外偏移1.5%，且30分钟冷却已结束，
才移动中心。订单每2小时刷新一次。

## schema v2 合同

`active_selection.json` schema v2 必须同时、且仅包含 `BTC-FDUSD` 与 `ETH-FDUSD`。每对必须包含
`profile`、`grid_range`、`grid_levels`、`take_profit`、`minimum_order_quote`、
`move_threshold`、`min_grid_move_seconds` 和 `order_refresh_seconds`。消费者逐字段核对代码内的
不可变批准Profile；缺对、多对、错误Profile、奇数格、低于10 FDUSD或任意值漂移都会拒绝。
schema v1继续兼容，解释为BTC/ETH共享旧参数。

Runtime State schema v9持久化 `active_pair_parameters`、参数版本、合同哈希、每对Grid中心和
参数限制原因。v8迁移只把已有全局参数复制到两对，不重置资金账本、累计盈亏、权益峰值、
恢复阶段、v22状态或持仓计时器。

## 订单裁剪

普通BUY和SELL的最低金额均为 `max(10 FDUSD, 交易所动态MIN_NOTIONAL/NOTIONAL)`，数量先按
`LOT_SIZE`量化，再复核金额。若库存或资金不足，从离中心最远的层开始删除，重新均分后再次
量化，直到所有层都合格。SELL总量不得超过该策略账本、启动库存上限和交易所可用余额三者的
最小值。

切换前快照显示BTC归属基础币只能支持约5个有效SELL层，ETH约可支持9层；最终数量必须以切换
时余额、成本底线、动态过滤器和量化结果为准。

## 原子切换与门控

Scheduler用一次原子文件替换发布BTC/ETH合同。策略在一个控制周期读取一次完整合同并立即更新
参数状态，然后撤销两对旧普通Grid订单、复核残留，再按新参数重建。v22 Risk-Off、FOMC、
熔断、EXITING、COOLDOWN或REENTRY只延迟普通订单创建，不延迟参数合同生效。

参数构建错误只限制对应交易对并产生 `grid_parameter_pair_restricted` 审计事件，不伪装成v22
Risk-Off，也不因普通参数错误清仓；修复前不回退旧6%参数。真实行情、API、合同完整性或交易
风险故障仍按原有Fail-Closed和退出策略处理。

## 360天混合组合证据

产物目录：`results/backtests/grid_btc_medium_eth_long_v22_360d/`。窗口为
2025-08-24 00:00至2026-08-19 00:00 UTC，使用5分钟连续行情、共享420 FDUSD组合熔断、
v22、亏损/回撤、持仓保护、强制退出和自动恢复；FOMC不参与历史执行。

| 方案 | BTC收益 | ETH收益 | 组合收益 | 组合最大回撤 |
|---|---:|---:|---:|---:|
| 现网固定参数，5.25下限 | -4.7304 | -3.8744 | -8.6048 | -13.3572% |
| 现网固定参数，10下限 | -11.2311 | -15.7711 | -27.0023 | -15.2351% |
| BTC中短期／ETH长期 | +7.5117 | +36.2113 | +43.7230 | -9.0170% |

这些数值是离线反事实，不是收益承诺，也不是自动上线门槛。硬门槛仅为：无负仓、无超预算、
无跨机器人库存出售、无低金额订单、v22阻止期间普通BUY为零，并保持退出与恢复幂等。本次三组
回放上述异常计数均为0。

## 部署顺序

1. 备份配置、active selection、Runtime State、SQLite、库存账本、Guard状态、Telegram outbox和release。
2. 先部署支持schema v2的Grid消费者，但保留schema v1；确认旧参数和门控恢复。
3. 再部署Scheduler与报告组件，由Scheduler原子发布schema v2。
4. 核对合同哈希、旧单撤销、新单金额、实际BUY/SELL层、库存归属和全部门控。
5. 观察10分钟、完整2小时刷新周期和24小时；不新增Compose service。
