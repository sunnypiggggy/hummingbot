# Grid/DCA 风控与执行机制全集

本文是 `ethbtc-forced-exit` 发布族的机制权威定义。其他文档只说明合同、容器或运维
细节，不应重新定义门控语义。动态概率、订单和盈亏必须以 Binance、机器人 Runtime
State 和 Guard 合同为准。

## 统一决策原则

- Grid 与 DCA 分别计算策略和组合风险，不跨策略混算。
- 所有已启用的普通 BUY 门按逻辑 AND 聚合；任一门关闭，普通 BUY 就关闭。
- v22 是 BUY 风险模型，不把普通 SELL 变成模型策略；但进入风险退出状态机后，执行
  覆盖层会暂时关闭 BUY/SELL，避免退出和新订单竞态。
- 止损、撤单、归属库存退出和紧急动作优先于普通交易门。
- 一个机制恢复只清除自己的限制，不能覆盖仍生效的其他门。
- 模型授权、合同完整性、账户指纹和库存归属是硬互锁，不能通过关闭普通机制开关绕过。

## 机制总表

| 机制 | Grid / DCA 范围 | 普通 BUY / SELL | 撤单或清仓 | 恢复与冷却 | 锁存条件 | 开关或真相源 |
|---|---|---|---|---|---|---|
| `v22_weekly_buy_gate` | BTC、ETH逐对；DCA固定映射FDUSD信号到USDT | Risk-On仅放行本门；Risk-Off进入执行覆盖层并关闭双侧 | 健康Risk-Off取消该对订单并退出归属库存 | 模型恢复后零固定冷却，仍需3个健康周期及其他门放行 | 模型不可用属于完整性故障，退出后`LATCHED` | `GRID/DCA_RISK_V22_WEEKLY_GATE_ENABLED`；v22 live合同 |
| `fomc_gate` | Grid、DCA；按租约方向 | 可限制BUY、SELL或双侧 | 由批准租约决定；不推测历史区间 | 租约结束自动恢复 | 合同持续失效按完整性策略处理 | `GRID/DCA_RISK_FOMC_GATE_ENABLED`；宏观合同 |
| `strategy_loss_breaker` | 单交易对/单机器人 | 触发后执行覆盖层关闭双侧 | 退出对应策略归属库存 | 退出完成后6小时 | 可恢复交易风险，不因开关变化自动绕过当前事件 | 对应`STRATEGY_LOSS_BREAKER`开关；风险周期权益 |
| `strategy_drawdown_breaker` | 单交易对/单机器人 | 同上 | 同上 | 退出完成后6小时 | 峰值和阶段持久化 | 对应`STRATEGY_DRAWDOWN_BREAKER`开关；持久化峰值 |
| `portfolio_loss_breaker` | Grid内部BTC+ETH；DCA内部BTC+ETH | 触发后组合双侧关闭 | 组合成员全部退出 | 退出完成后12小时，组合原子重入 | 任一成员退出未完成时不得恢复 | 对应`PORTFOLIO_LOSS_BREAKER`开关；组合权益 |
| `portfolio_drawdown_breaker` | 同上 | 同上 | 同上 | 退出完成后12小时 | 组合峰值和阶段持久化 | 对应`PORTFOLIO_DRAWDOWN_BREAKER`开关；组合峰值 |
| `position_protection` | Grid逐对；DCA逐executor/机器人 | Grid可限制新增BUY；DCA止损流程关闭相关普通交易 | 超时库存或止损库存按归属退出 | 单仓/持仓保护30分钟；Grid库存期限见下文 | 仅完整性失败才进入`LATCHED` | 对应`POSITION_PROTECTION`开关；策略账本/executor |
| `infrastructure_integrity_breaker` | 合同、模型、行情、API、数据库、账户指纹 | 达到持续失败门槛后Fail-Closed | 取消订单并退出归属库存 | 不自动恢复交易权限 | 确定性哈希/授权错误或持续不可用退出后`LATCHED` | 已提交generation、Guard健康及完整性合同 |
| 恢复执行覆盖层 | `EXITING/COOLDOWN/REENTRY/LATCHED` | 非`ACTIVE`阶段通常关闭双侧 | `EXITING`负责撤单和退出 | 技术0、持仓30分钟、策略6小时、组合12小时 | `LATCHED`必须人工复核 | Grid Runtime、DCA Guard恢复状态 |
| 统一库存归属 | 共享账户BTC/ETH | 归属缺口时禁止增加或转移风险 | 只允许出售当前机器人归属 | 证据恢复后重新对账；不覆盖其他门 | `ownership_deficit`或账户指纹不一致Fail-Closed | `account_inventory_status.json`和共享SQLite |
| DCA资金观察 | DCA共享USDT | **只告警，不参与普通BUY/SELL聚合** | 不清仓 | 余额恢复后清除告警 | 不单独锁存 | `gate_aggregate.capital.mode=alert_only` |
| Controller/订单落地 | DCA controller；Grid实际订单 | 状态合同与执行面不一致时不能称为正常交易 | 不自行扩大清仓范围 | 落地成功后恢复状态展示 | 持续执行故障按对应错误策略处理 | controller值、Binance订单、Runtime State |
| Grid订单构建与Maker保护 | Grid逐对逐层 | 只生成预算、精度、最低金额和盘口均合法的普通订单 | 单层穿价只延迟该层；不整组极端清仓 | 2/5/15秒单层重试，持续后拓扑刷新 | 普通构建失败只限制该对，不进入模型锁存 | `order_build_status`、动态过滤器、best bid/ask |
| 手续费与禁止BNB抵扣 | 共享Binance账户 | 不直接决定普通门 | 风险市价单仍按真实费用执行 | 账户设置恢复并通过预检 | BNB抵扣开启或费用策略不可验证时不允许武装Guard | `spotBNBBurn=false`及成交回报 |
| 通知与审计 | 全部机器人和机制 | 不参与交易权限计算 | 不发送交易指令 | outbox重试并在恢复后补发当前事件 | 通知失败不锁存交易 | 标准事件、Telegram outbox、四小时报告 |

## 七类风险机制阈值

### 1. v22周度技术门

- Grid直接消费 `BTC-FDUSD`、`ETH-FDUSD`；DCA映射
  `BTC-USDT ← BTC-FDUSD`、`ETH-USDT ← ETH-FDUSD`。
- Risk-On只表示v22本门放行，不等于一定存在BUY挂单。
- Risk-Off清理该机器人、该交易对的归属库存；模型恢复后仍须其他门、资金和恢复状态
  共同允许重入。
- 模型缺失、过期、哈希错误、周缺口、交易对缺失或授权错误输出`UNAVAILABLE`，这是
  完整性故障，不是正常Risk-Off，也不能回退v21、ROC、SQZMOM或上一周模型。

### 2. FOMC宏观门

- 只有已批准且处于有效时间窗的租约生效，可限制BUY、SELL或双侧。
- 租约结束只恢复FOMC本门；v22、熔断或恢复状态仍可能继续阻止交易。
- 无可信历史记录时报告“无数据”，不得推测阴影或事后生成限制区间。

### 3—6. 亏损与回撤熔断

| 阈值 | FDUSD Grid | DCA |
|---|---:|---:|
| 单策略亏损 | 6 FDUSD/对 | 16 USDT/机器人 |
| 单策略峰值回撤 | 3%/对 | 8%/机器人 |
| 组合亏损 | 24 FDUSD | 32 USDT |
| 组合峰值回撤 | 6% | 8% |

单策略熔断冷却6小时，组合熔断冷却12小时。权益峰值、触发事件和恢复阶段均持久化；
容器重启或当期盈亏回升不能绕过已经触发的状态。

### 7. 持仓保护

Grid：

- SELL成本利润底线避免低于移动平均持仓成本加当前普通止盈率卖出。
- 每对额外基础币容忍上限为约10 FDUSD。接近上限时，即使账本仍有现金且v22为
  Risk-On，BUY预算也可能低于10 FDUSD，因而合法地形成0张BUY。
- 新增库存享有24小时利润保护；最长持有48小时后进入库存退出流程。

DCA：

- 单executor止损为5%，普通止盈为2%。
- 部分成交同样受止损保护；executor自首次成交起最长持有5小时。
- Guard识别止损成交后进入30分钟恢复周期，阻止15秒内立即重新暴露风险。

## Grid实际订单数量机制

配置中的18格是候选拓扑，不是“始终9 BUY＋9 SELL”的承诺。每次刷新都按以下顺序形成
实际订单：

1. 以当前Grid中心和mid price划分上下逻辑层。
2. BUY预算取策略账本现金、100 FDUSD每侧上限、组合剩余资金和额外库存额度的最小值。
3. SELL预算取策略账本基础币、启动库存上限和交易所可用余额的最小值。
4. BUY数量为满足最低金额而按步长安全上取整；预算不足时从最远层删除并重新分配。
5. SELL数量始终向下量化，禁止超卖。
6. SELL执行价取“原Grid层、当前价止盈线、移动平均成本利润底线”三者最大值。
7. 多个逻辑SELL层若落到同一量化价格，合并为一张订单并汇总数量；总出售量不变。
8. 最后应用动态`PRICE_FILTER`、`LOT_SIZE`、`MIN_NOTIONAL/NOTIONAL`和Maker盘口检查。
   穿价层进入延迟队列，不改价追单，也不退化为Taker。

因此实际BUY/SELL数量可不对称，也会随库存、资金、成本、盘口和风控状态变化。
`EXPECTED_EMPTY`表示门控或恢复状态决定的预期无挂单；它与订单构建故障不同。
门控放行、阶段为`ACTIVE`且理论上应有订单但实际为0时，逐对按5/15/30/60秒退避
重建；持续失败只限制该交易对，不清仓、不锁存，也不影响另一交易对。

## DCA资金观察门

`gate_aggregate.capital`保留`free_quote`、`required_quote`、缓冲和数据年龄，供容量告警与
重入预检使用。其生产语义固定为：

```text
mode = alert_only
enforced = false
```

普通DCA最终权限不得与`buy_ready`做逻辑AND。原因是活动BUY订单会把USDT从free移到
locked；若把free余额当硬门，会产生撤单/重建反馈循环。真正提交订单时仍由交易所、
controller预算及归属边界校验可负担性；自动重入的市价建仓也必须单独通过资金预检。

## 状态真相与“正常交易”

排查顺序固定为：

1. Binance实际活动订单、余额和成交是经济事实。
2. 机器人SQLite记录订单和成交生命周期。
3. Grid Runtime或DCA controller说明预计层数、最终BUY/SELL和恢复阶段。
4. Guard合同解释v22、FOMC、熔断、完整性及库存归属。
5. Telegram、PNG和文档只用于审计，不能单独证明正在交易。

只有进程运行、合同新鲜、恢复阶段`ACTIVE`、controller/订单状态已落地且最终BUY/SELL
均放行时，报告才显示“正常交易”。Risk-Off下0单应显示“预期无挂单”；存在合法订单但
部分Maker层延迟时显示“正常交易（部分Maker层等待安全盘口）”。

## 费用与独立开关

- FDUSD普通`LIMIT_MAKER`按Maker 0%记账。
- 风险退出和自动重入使用市价单，按交易所实际Taker费用审计；暂时无法换算时保守按
  0.1%报价币费用，不得回退为0%。
- Binance现货BNB手续费抵扣必须关闭，任何通道不得假定使用BNB。
- 七类风险机制均展开为独立的 `GRID_RISK_<MECHANISM>_ENABLED` 和
  `DCA_RISK_<MECHANISM>_ENABLED`；`<MECHANISM>`分别为
  `V22_WEEKLY_GATE`、`FOMC_GATE`、`STRATEGY_LOSS_BREAKER`、
  `STRATEGY_DRAWDOWN_BREAKER`、`PORTFOLIO_LOSS_BREAKER`、
  `PORTFOLIO_DRAWDOWN_BREAKER`、`POSITION_PROTECTION`。
- 自动重入开关只控制重入动作，不解除熔断、`LATCHED`或完整性硬互锁；模型授权、
  generation完整性、账户指纹和库存归属也不能被普通开关绕过。

## 权益与库存计价

- Grid 单对权益固定为 `PairLedger.quote + PairLedger.base × 当前标记价`，资本基准为200 FDUSD。
- DCA 单机器人权益固定为 `显式报价币余额 + dca-equity-ledger-v1.owned_base × 当前标记价`，资本基准为190 USDT。
- 4小时、24小时、7天和累计收益均取连续权益端点差，并剔除明确资金注入/提取。
- 可成交库存与 Dust 分开显示；停止且已清仓时，权益只允许随真实 Dust 产生极小波动。
- 负 `net_base` 不代表现货空头；账本无法解释时显示 `EQUITY_UNRECONCILED`，不得绘制合成空头权益。
- Telegram各机制开关只控制是否通知，不改变对应机制是否执行；outbox或图片失败不得
  阻塞交易线程。
