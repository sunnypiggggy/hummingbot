# ethbtc-forced-exit 机制文档

Telegram 频道告警、四小时单机器人 PNG、参数 PDF/360 天证据、Hermes 恢复提示和
OCI 运维见 [TELEGRAM_NOTIFICATIONS.md](TELEGRAM_NOTIFICATIONS.md)。

本目录是 `ethbtc-forced-exit` 发布族的 UTF-8 中文机制说明。它解释生产执行语义，不改变冻结模型、离线证据或已经生成的内容哈希 release。

## 文档索引

- [ACCOUNT_INVENTORY.md](ACCOUNT_INVENTORY.md)：Grid/DCA 共享账户归属、事务租约、DCA 完整性清仓和无归属转 USDT。
- [REAL_SCENARIO_TESTING.md](REAL_SCENARIO_TESTING.md)：隔离HTTP/容器故障注入、真实30秒确认、重启幂等和验收报告。

- [ONLINE_MODELS.md](ONLINE_MODELS.md)：当前 OCI 实际启用的模型、release、哈希、周期、Grid/DCA映射和运行边界。
- [CONTAINERS_AND_SIGNAL_FLOW.md](CONTAINERS_AND_SIGNAL_FLOW.md)：当前 v22 live 容器拓扑、唯一 producer、共享目录和交易链路。
- [RISK_MECHANISMS.md](RISK_MECHANISMS.md)：Grid/DCA 七类风控、阈值、作用范围和开关。
- [V22_WEEKLY_MODEL.md](V22_WEEKLY_MODEL.md)：v22 周度模型、阈值、状态连续性、训练防泄漏和 Fail-Closed。
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
