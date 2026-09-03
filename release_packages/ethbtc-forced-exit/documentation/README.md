# ethbtc-forced-exit 机制文档

Telegram 频道告警、四小时单机器人 PNG、模型更新三窗口 PNG、Hermes 恢复提示和
OCI 运维见 [TELEGRAM_NOTIFICATIONS.md](TELEGRAM_NOTIFICATIONS.md)。

本目录是 `ethbtc-forced-exit` 发布族的 UTF-8 中文机制说明。它解释生产执行语义，不改变冻结模型、离线证据或已经生成的内容哈希 release。

## 权威定义与引用原则

- [RISK_MECHANISMS.md](RISK_MECHANISMS.md) 是机制名称、权限影响、退出动作、恢复条件、冷却、锁存、开关和真相源的唯一权威定义。
- 其余文档解释某一条链路的实现、合同或运维方法；若摘要表述与机制总表冲突，以机制总表和当前运行合同为准。
- OCI 的概率、fold、订单数和盈亏会持续变化。带日期的数值只能作为审计快照，不得复制为永久配置。

## 机制—文档索引

| 机制或执行互锁 | 权威定义 | 实现与运维说明 |
|---|---|---|
| v22 周度技术门与无故障切换 | [RISK_MECHANISMS.md](RISK_MECHANISMS.md) | [V22_WEEKLY_MODEL.md](V22_WEEKLY_MODEL.md)、[V22_ZERO_DOWNTIME_CUTOVER.md](V22_ZERO_DOWNTIME_CUTOVER.md)、[ONLINE_MODELS.md](ONLINE_MODELS.md) |
| FOMC、策略/组合亏损与回撤、持仓保护 | [RISK_MECHANISMS.md](RISK_MECHANISMS.md) | [FORCED_EXIT_AND_RECOVERY.md](FORCED_EXIT_AND_RECOVERY.md) |
| 基础设施/完整性与瞬时故障宽限 | [RISK_MECHANISMS.md](RISK_MECHANISMS.md) | [RESILIENCE_POLICY.md](RESILIENCE_POLICY.md)、[CONTRACTS_AND_RUNTIME_FLOW.md](CONTRACTS_AND_RUNTIME_FLOW.md) |
| `EXITING/COOLDOWN/REENTRY/LATCHED` | [RISK_MECHANISMS.md](RISK_MECHANISMS.md) | [FORCED_EXIT_AND_RECOVERY.md](FORCED_EXIT_AND_RECOVERY.md) |
| 统一库存、Dust、归属缺口与重复卖出优先级 | [RISK_MECHANISMS.md](RISK_MECHANISMS.md) | [ACCOUNT_INVENTORY.md](ACCOUNT_INVENTORY.md) |
| DCA 资金观察、Controller 落地 | [RISK_MECHANISMS.md](RISK_MECHANISMS.md) | [CONFIGURATION_AND_OPERATIONS.md](CONFIGURATION_AND_OPERATIONS.md)、[CONTRACTS_AND_RUNTIME_FLOW.md](CONTRACTS_AND_RUNTIME_FLOW.md) |
| Grid 参数、订单数量、最小金额、Maker 与零订单恢复 | [RISK_MECHANISMS.md](RISK_MECHANISMS.md) | [GRID_PAIR_PARAMETER_CUTOVER.md](GRID_PAIR_PARAMETER_CUTOVER.md)、[CONFIGURATION_AND_OPERATIONS.md](CONFIGURATION_AND_OPERATIONS.md) |
| 手续费与禁止 BNB 抵扣 | [RISK_MECHANISMS.md](RISK_MECHANISMS.md) | [NO_BNB_FEE_POLICY.md](NO_BNB_FEE_POLICY.md) |
| Telegram、四小时报告与 Hermes 提示 | [RISK_MECHANISMS.md](RISK_MECHANISMS.md) | [TELEGRAM_NOTIFICATIONS.md](TELEGRAM_NOTIFICATIONS.md)、[WEEKLY_APPROVAL_NOTIFICATIONS.md](WEEKLY_APPROVAL_NOTIFICATIONS.md) |

## 完整文档索引

- [GRID_PAIR_PARAMETER_CUTOVER.md](GRID_PAIR_PARAMETER_CUTOVER.md)：BTC中短期横盘、ETH长期波动的逐交易对合同、订单裁剪、混合回测和上线流程。
- [NO_BNB_FEE_POLICY.md](NO_BNB_FEE_POLICY.md)：禁止使用 BNB 支付现货手续费、Guard 启动互锁、Grid Maker/Taker 费用口径及历史修正。
- [ACCOUNT_INVENTORY.md](ACCOUNT_INVENTORY.md)：Grid/DCA 共享账户归属、事务租约、DCA 完整性清仓和无归属转 USDT。
- [REAL_SCENARIO_TESTING.md](REAL_SCENARIO_TESTING.md)：隔离HTTP/容器故障注入、真实30秒确认、重启幂等和验收报告。

- [ONLINE_MODELS.md](ONLINE_MODELS.md)：当前 OCI 实际启用的模型、release、哈希、周期、Grid/DCA映射和运行边界。
- [CONTAINERS_AND_SIGNAL_FLOW.md](CONTAINERS_AND_SIGNAL_FLOW.md)：当前 v22 live 容器拓扑、唯一 producer、共享目录和交易链路。
- [RISK_MECHANISMS.md](RISK_MECHANISMS.md)：七类风控和全部执行互锁的统一总表、阈值、权限、恢复及真相源。
- [V22_WEEKLY_MODEL.md](V22_WEEKLY_MODEL.md)：v22 周度模型、阈值、状态连续性、训练防泄漏和 Fail-Closed。
- [TELEGRAM_MODEL_PARAMETERS.md](TELEGRAM_MODEL_PARAMETERS.md)：管理Bot的Grid/DCA/风控参数与v22证据只读合同。
- [WEEKLY_APPROVAL_NOTIFICATIONS.md](WEEKLY_APPROVAL_NOTIFICATIONS.md)：每周候选、默认审批、原子切换时间及 12 张回测 PNG 的同步通知链路。
- [V22_ZERO_DOWNTIME_CUTOVER.md](V22_ZERO_DOWNTIME_CUTOVER.md)：两阶段预热、事务型 runtime generation、周状态继承、失败回滚与统一交易状态。
- [FORCED_EXIT_AND_RECOVERY.md](FORCED_EXIT_AND_RECOVERY.md)：强制退出、双通道接管、冷却、自动重入和人工解锁。
- [CONFIGURATION_AND_OPERATIONS.md](CONFIGURATION_AND_OPERATIONS.md)：容器依赖、合同字段、环境变量、观察、审批、切换、监控与回滚。
- [CONTRACTS_AND_RUNTIME_FLOW.md](CONTRACTS_AND_RUNTIME_FLOW.md)：所有 JSON 合同、producer/consumer、授权语义、运行链路和真实阻塞判定。

## 文档与 release 的关系

- `releases/<release_sha256>/` 是不可变生产候选，生成后不得原地补文件。
- 当前已生成 release 保持原哈希；本目录作为发布族级文档随代码保存。
- 后续通过 `stage_ethbtc_forced_exit_release.py` 生成的 release 会复制本目录，并把所有文档写入该 release 的 `MANIFEST.sha256`。
- 文档不能授予交易权限。交易权限只来自当前有效周、完整性检查、观察报告、账户预检和哈希绑定审批回执。

## 术语

- 普通 BUY：Grid 新买单或 DCA 新建仓执行器。
- Risk-Off：禁止普通 BUY，并由 `forced-exit-v2` 执行覆盖层取消订单、清理机器人归属基础币。
- 归属库存：Grid `capital_reservations` 或 DCA `managed_inventory + 机器人净成交` 证明属于该机器人的基础币。
- Fail-Closed：任何必要数据或完整性条件失败时不新增风险；完整性故障还会退出归属库存并锁存。
