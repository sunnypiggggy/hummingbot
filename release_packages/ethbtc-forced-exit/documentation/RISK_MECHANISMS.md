# Grid/DCA 风控机制全集

## 统一决策原则

Grid 与 DCA 各自计算自己的策略及组合风险，不跨策略混算。所有已启用 BUY 门采用逻辑 AND：任一门关闭，普通 BUY 即关闭。一个机制恢复不得覆盖其他仍生效的机制。

SELL、止损、取消订单、库存清理和紧急退出不受普通 BUY 门阻塞。进入 `EXITING/COOLDOWN/REENTRY` 时执行覆盖层会临时关闭双侧，以防退出期间产生新订单；该限制属于退出状态机，不等于 v22 获得主动交易策略权限。

## 七类机制

### 1. `v22_weekly_buy_gate`

- 输入：BTC-FDUSD、ETH-FDUSD 冻结周模型的概率、fold-local 阈值和4小时结构。
- Grid 直接消费 FDUSD 信号；DCA 映射 `BTC-USDT ← BTC-FDUSD`、`ETH-USDT ← ETH-FDUSD`。
- Risk-On：只表示技术门放行，最终 BUY 仍取决于其他六类机制。
- Risk-Off：取消该机器人订单并清理该交易对归属基础币；模型恢复后零固定冷却，但仍需其他门放行和连续3个健康周期。
- 模型缺失、过期、哈希错误、周缺口或未授权：Fail-Closed；完整性故障退出后进入 `LATCHED`。

开关：`GRID_RISK_V22_WEEKLY_GATE_ENABLED`、`DCA_RISK_V22_WEEKLY_GATE_ENABLED`。

### 2. `fomc_gate`

- 输入：FOMC 宏观事件租约及审批方向。
- 可限制 BUY、SELL 或双侧；无可信事件历史时不得推测区间。
- 租约结束后自动恢复，但恢复不会覆盖 v22 或熔断状态。
- 合同过期或事件源不健康时按配置 Fail-Closed。

开关：`GRID_RISK_FOMC_GATE_ENABLED`、`DCA_RISK_FOMC_GATE_ENABLED`。

### 3. `strategy_loss_breaker`

按单交易对/单机器人风险周期累计盈亏触发。

- FDUSD Grid：每对亏损达到 `6 FDUSD`。
- DCA：每个 BTC/ETH 机器人亏损达到 `16 USDT`。
- 触发后清理该策略归属库存；退出完成后冷却6小时。

开关：`GRID_RISK_STRATEGY_LOSS_BREAKER_ENABLED`、`DCA_RISK_STRATEGY_LOSS_BREAKER_ENABLED`。

### 4. `strategy_drawdown_breaker`

按单交易对/机器人风险周期权益峰值计算：`(peak-equity)/peak`。

- FDUSD Grid：每对回撤 `3%`。
- DCA：每机器人回撤 `8%`。
- 触发后清理该策略归属库存；退出完成后冷却6小时。
- 重启后从持久化峰值继续，不得重新计算绕过门槛。

开关：`GRID_RISK_STRATEGY_DRAWDOWN_BREAKER_ENABLED`、`DCA_RISK_STRATEGY_DRAWDOWN_BREAKER_ENABLED`。

### 5. `portfolio_loss_breaker`

在策略自身资金边界内汇总 BTC+ETH。

- FDUSD Grid：组合亏损达到 `24 FDUSD`。
- DCA：组合亏损达到 `32 USDT`。
- 组合触发要求 BTC、ETH 都完成退出；退出完成后冷却12小时。

开关：`GRID_RISK_PORTFOLIO_LOSS_BREAKER_ENABLED`、`DCA_RISK_PORTFOLIO_LOSS_BREAKER_ENABLED`。

### 6. `portfolio_drawdown_breaker`

- FDUSD Grid：组合峰值回撤 `6%`。
- DCA：组合峰值回撤 `8%`。
- 组合峰值持久化；组合恢复要求相关机器人全部退出并满足重入条件。
- 退出完成后冷却12小时。

开关：`GRID_RISK_PORTFOLIO_DRAWDOWN_BREAKER_ENABLED`、`DCA_RISK_PORTFOLIO_DRAWDOWN_BREAKER_ENABLED`。

### 7. `position_protection`

DCA：

- 单执行器止损固定 `5%`。
- 部分成交也启用止损；从首次成交开始计算最长持仓5小时。
- Guard 发现新的止损成交后进入单仓恢复流程，冷却30分钟，阻止立即创建新执行器。

Grid：

- 卖单成本底线避免低于受保护成本卖出。
- 额外库存容忍上限约 `10 FDUSD/交易对`。
- 新增库存有24小时利润保护期；最长持有48小时后进入库存退出。
- 该机制与 v22 技术退出、亏损和回撤熔断独立。

开关：`GRID_RISK_POSITION_PROTECTION_ENABLED`、`DCA_RISK_POSITION_PROTECTION_ENABLED`。

## 资金与交易参数

| 项目 | FDUSD Grid | DCA |
|---|---:|---:|
| 总资金上限 | 420 FDUSD | 每机器人200 USDT |
| 策略资金 | 400 FDUSD | 每机器人190 USDT |
| 现金储备 | 20 FDUSD | 每机器人10 USDT |
| 单对/机器人重入基础库存 | 约100 FDUSD | 约95 USDT |
| 普通 Grid Maker 费率模型 | 0% | 不适用 |
| 强制退出/重入 | 市价，使用实时交易所费率和过滤器 | 同左 |

## 订单生命周期

- Grid 普通挂单最长保留2小时；到期撤销并按当时有效参数重建，不代表触发止损或熔断。
- Grid 参数轮询为60秒，网格移动最短间隔30分钟；参数切换不重置累计盈亏、权益峰值或风险状态。
- DCA executor 从首次成交开始计算5小时期限；未成交 executor 的刷新周期同为5小时。
- DCA 普通退出规则保留2%止盈、5%止损；v22 和熔断强制退出优先于普通期限退出。

## 开关与硬互锁

七类普通机制默认开启，并分别提供 Grid/DCA 开关。模型授权、哈希、有效周和资金归属属于硬互锁，不能通过关闭普通机制开关绕过。`DCA_RISK_AUTO_REENTRY_ENABLED` 和 Grid 配置 `risk_auto_reentry_enabled` 只控制自动重入，不解除熔断或完整性锁。
