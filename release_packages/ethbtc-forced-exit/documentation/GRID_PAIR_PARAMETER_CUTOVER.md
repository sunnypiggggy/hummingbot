# Grid 逐交易对参数合同与切换

## 当前生产选择

参数版本固定为 `binance-ai-btc-medium-sideways-eth-long-volatility-v1`，版本名不含时间戳；
审计与幂等以整个 `active_selection.json` 的规范化 SHA-256 为准。

| 交易对 | Profile | 总范围 | 格数 | 相邻格距 | 普通止盈 | 普通订单下限 |
|---|---|---:|---:|---:|---:|---:|
| BTC-FDUSD | `medium_sideways` | 12.698379% | 18（理论9/9） | 约0.746964% | 0.400000% | 10 FDUSD |
| ETH-FDUSD | `long_volatility` | 52.465116% | 18（理论9/9） | 约3.086183% | 1.417976% | 10 FDUSD |

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

当前 Runtime State 为 schema 13，持久化 `active_pair_parameters`、参数版本、合同哈希、
每对Grid中心、`order_build_status`、Maker延迟层、刷新generation和参数限制原因。旧schema
迁移只把已有全局参数复制到两对，并补齐新增状态字段；不得重置资金账本、累计盈亏、权益
峰值、恢复阶段、v22状态或持仓计时器。

## 理论层数与实际订单数

18格只定义候选价格拓扑，不保证任意时刻都有9张BUY和9张SELL。每次刷新时，实际订单
还要依次经过价格相对mid的位置、逐对门控、资金/库存边界、成本利润底线、交易所动态
过滤器和Maker盘口安全检查。

BUY预算为以下值的最小值：

- 该对策略账本中的报价币现金；
- 100 FDUSD BUY侧预算；
- 组合尚未分配的实时可用FDUSD；
- 持仓保护允许的额外库存额度，即“基准缺口价值＋10 FDUSD上限－当前额外库存价值”。

预算低于 `max(10 FDUSD, MIN_NOTIONAL/NOTIONAL)` 时，0张BUY是预期结果，不应触发
零订单自恢复。预算足够时从离现价最近的BUY层开始保留；数量先按交易所步长安全上取整
以跨过最低金额，若超预算则删除最远层后重新分配。

SELL预算为策略账本基础币、启动库存上限和交易所可用基础币三者的最小值。每个逻辑层的
执行价格为：

```text
max(原Grid层, 当前mid × (1 + 普通止盈), 移动平均成本 × (1 + 利润底线率))
```

成本利润底线或价格tick可能把多个逻辑SELL层压到同一执行价格。实现会先合并这些同价层，
将数量相加后只提交一张SELL；这不会增加总出售数量，也避免同一批次出现重复价格拒单。
SELL数量始终向下量化，不得套用BUY的上取整逻辑。

## 动态过滤与Maker保护

普通BUY和SELL的最低金额均为 `max(10 FDUSD, 交易所动态MIN_NOTIONAL/NOTIONAL)`，数量先按
`LOT_SIZE`量化，再复核金额。若库存或资金不足，从离中心最远的层开始删除并重新均分，直到
所有层都合格。

提交前还必须读取同一时点的best bid/ask：BUY严格低于best ask，SELL严格高于best bid。
不安全层进入Maker延迟队列，按2/5/15秒仅重试该层；不得改价追单、退化为普通LIMIT/Taker，
也不得因为一层穿价撤销已经成功挂出的其他层。门控、参数generation或Grid拓扑变化会使旧的
延迟意图失效。

`order_build_status`同时记录预计/实际BUY、SELL层数、构建原因和延迟层。门控或恢复阶段决定
不应挂单时使用`EXPECTED_EMPTY`；门控放行且理论上应有订单但实际为0时才进入逐对零订单
自恢复。一个交易对的重建不得撤销另一个交易对的订单。

## 2026-09-02订单审计示例

以下是北京时间2026-09-02的只读时点示例，不是固定配置，也不能代替下一次实时核查：

- BTC为v22 Risk-On且恢复阶段`ACTIVE`，但基础币相对策略基准多约`0.0001259 BTC`，按当时
  价格约`9.77 FDUSD`，已接近10 FDUSD额外库存上限；剩余BUY额度不足10 FDUSD，因此
  实际为0张BUY。
- BTC成本利润底线把多个较低SELL逻辑层合并，交易所实际为4张`LIMIT_MAKER` SELL，合计
  `0.00155 BTC`；Binance、SQLite和Runtime State三方一致。
- ETH为v22 Risk-Off并停留在`REENTRY`等待模型恢复，故0张BUY、0张SELL属于
  `EXPECTED_EMPTY`，不是订单构建故障。

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

频道证据位于 `parameter_evidence/grid_btc_medium_eth_long_v1/`，包含 BTC/ETH 各自的360天、2026年1—2月、
2026年5—6月三张手机PNG。`grid_parameter_evidence_manifest.json` 将六张图片、连续权益源文件和
参数合同SHA-256绑定。新参数只能在六张图片全部被Telegram接收并生成送达回执后激活；
`REPORT_EVIDENCE_MISSING`不再是允许先激活、事后告警的降级路径。

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
