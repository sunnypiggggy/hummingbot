# 配置、合同与上线运维

## 容器职责

不新增服务：`grid-live-guard` 是唯一 v22 producer，加载模型、行情和连续状态并发布合同；Grid 直接消费。`dca-live-guard` 通过只读共享目录消费同一合同，不加载模型。

生产中不得同时运行第二个 v22 producer。完成切换后 v21 producer 必须关闭，不能作为故障回退。

## 核心合同

合同包含 `package_id`、执行策略版本、release/model/feature/strategy/data哈希、生成时间、有效期、审批回执哈希、激活时间和逐资产信号。

每对包含 `risk_off_active`、`recommended_buy_enabled`、实际 `buy_enabled`、`force_exit`、概率、fold-local阈值、模型周、转换、原因和事件ID。DCA 必须逐事件验证 FDUSD→USDT 映射一致。

## 风控开关

每个机制分别使用：

- `GRID_RISK_<MECHANISM>_ENABLED`
- `DCA_RISK_<MECHANISM>_ENABLED`

`<MECHANISM>` 为 `V22_WEEKLY_GATE`、`FOMC_GATE`、`STRATEGY_LOSS_BREAKER`、`STRATEGY_DRAWDOWN_BREAKER`、`PORTFOLIO_LOSS_BREAKER`、`PORTFOLIO_DRAWDOWN_BREAKER`、`POSITION_PROTECTION`。

其他关键配置：`GRID_V22_EXECUTION_MODE`、`GRID_V21_IN_GUARD_ENABLED`、`GRID_V22_PACKAGE_PATH`、`GRID_V22_AUTHORIZATION_PATH`、`DCA_V22_GATE_PATH`、`DCA_RISK_AUTO_REENTRY_ENABLED`，以及 Grid 策略配置 `risk_auto_reentry_enabled`。

## 观察与审批

观察模式真实刷新行情、概率、阈值和拟执行数量，但不修改 controller、不撤单、不成交。每个 release 至少观察24小时，要求合同持续新鲜、BTC/ETH覆盖完整、Grid/DCA事件一致、零完整性错误、行情可用率至少99.9%、归属数量不越界。

审批 CLI 绑定 release 与模型哈希、观察报告、账户预检、审批者和未来分钟边界。观察未满、报告失败、当前周到期或哈希变化时拒绝生成授权。

## 原子切换

切换前备份 Compose、环境、Guard状态、归属账本、机器人配置和数据库。关闭 v21 producer，将 v22 设置为 live 模式；先以未到激活时间的合同使普通 BUY Fail-Closed，再同批替换 Grid/DCA Guard 和加载新 Grid 配置。两个消费者在同一 `activate_at` 后执行。

## 激活后检查

10分钟、1小时、24小时分别检查：合同年龄、事件一致性、controller状态、订单取消、退出延迟、成交数量、手续费、滑点、dust、剩余风险、恢复阶段、归属余额和权益。

## Plotly 审计

Grid/DCA、BTC/ETH 分页展示各自权益，不使用组合权益冒充单对权益。七类机制各有独立阴影、标记、图例和复选框；v22 另有 BTC/ETH 子开关。隐藏机制图层不得隐藏价格、权益、峰值、回撤、概率或阈值。无可信事件显示“无数据”，不推测区间；HTML必须自包含UTF-8且不显示“机制事件数”或原始JSON大块区域。

## 回滚

先撤销授权并保持双侧关闭，再取消订单、复核和清理归属库存。回滚镜像不得自动恢复 v21、ROC、SQZMOM或旧周。只有新的有效周、完整预检和新审批完成后才可重新入场。

