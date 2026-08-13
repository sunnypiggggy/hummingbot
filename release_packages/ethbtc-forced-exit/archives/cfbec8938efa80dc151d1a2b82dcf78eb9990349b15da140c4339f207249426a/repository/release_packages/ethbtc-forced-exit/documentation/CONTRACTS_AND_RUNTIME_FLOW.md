# 系统合同、运行链路与阻塞判定

## 1. “合同”是什么

这里的合同（contract）不是法律合同，而是进程之间交换 JSON 数据时必须共同遵守的数据契约。它规定：

- 使用哪个 `schema` 或 `schema_version`；
- 哪些字段必须存在、类型和取值范围是什么；
- 数据由谁生成、谁只读消费；
- 文件多久失效、如何校验时间和 SHA-256；
- 信号、执行授权和实际交易动作之间如何换算；
- 验证失败时是继续交易、只观察，还是 Fail-Closed。

合同文件本身不自动等于交易权限。模型建议、审批回执、运行合同、策略状态和 Telegram 事件是不同层次，不能互相替代。

## 2. 总体链路

容器级完整拓扑、共享目录和当前 OCI 实际控制关系见
[CONTAINERS_AND_SIGNAL_FLOW.md](CONTAINERS_AND_SIGNAL_FLOW.md)。当前 OCI 已处于 v22
live 链；历史观察链只用于解释旧审计产物。

### 2.1 当前 OCI：v22 live 单 producer 链

```text
有效周 release + 审批回执 + 行情 ──► grid-live-guard（唯一 v22 producer）
                                      ├─► Grid 实盘策略
                                      └─► dca-live-guard（只读映射）
                                             └─► DCA controller BUY/SELL gate

FOMC/Hermes ──► macro state ──► Grid scheduler / DCA Guard
交易状态事件 ──► dca-live-report ──► Telegram（不回写交易）
```

v21 producer 已关闭；ROC/SQZMOM 不再作为线上独立技术 gate。真实下单权仍在三个
机器人容器，Guard 负责合同、最终门、紧急退出和恢复。

### 2.2 当前执行语义

```text
有效周 v22 release + 行情 + 连续状态 + 审批回执
                         │
                         ▼
                 grid-live-guard（唯一 producer）
                         │
                         ├── v22 live contract ──► Grid Guard/策略
                         └───────────────────────► DCA Guard 只读映射

v22 AND FOMC AND 亏损/回撤 AND 持仓保护 AND 恢复状态
                         │
                         ▼
                   最终交易权限/强制退出
```

原子切换已经完成，`grid-live-guard` 生成的 v22 合同进入最终权限计算。
`dca-live-guard` 不加载模型，只从只读共享目录读取合同，并执行固定映射：

- `BTC-USDT ← BTC-FDUSD`
- `ETH-USDT ← ETH-FDUSD`

Telegram 消息是审计和通知，不在交易决策链中；频道消息不能 reset、批准模型或改变 controller。

## 3. 合同登记表

| 合同 | Schema | Producer | Consumer | 主要作用 |
|---|---|---|---|---|
| v22 实时合同 | `ethbtc-forced-exit-live-contract-v1` | Grid Guard | Grid、DCA Guard | 模型信号、授权和强制退出语义 |
| v22 授权回执 | `ethbtc-forced-exit-authorization-v1` | OCI 审批 CLI | v22 producer | 将候选、观察、预检与激活时间绑定 |
| v22 observer 状态 | `ethbtc-forced-exit-observer-status-v1` | Grid/DCA observer | 健康检查、人工审计 | 证明两个消费者看到同一 release/事件 |
| 宏观主状态 | `schema_version=3` | Macro gateway/Hermes 流程 | Grid scheduler、DCA Guard | 宏观租约和 BUY/SELL 方向 |
| Grid FOMC gate | `grid-fomc-gate-v1` | Grid scheduler | Grid 策略 | 把宏观租约转换为是否暂停新单 |
| 风控恢复状态 | `ACTIVE/EXITING/COOLDOWN/REENTRY/LATCHED` | Grid/DCA 风控执行层 | 同一策略重启后的状态恢复 | 退出、冷却、重入和人工锁存 |
| Telegram 事件 | `ethbtc-telegram-event-v1` | Guard、策略、调度器、发布工具 | `dca-live-report` | 标准告警、报告和 Hermes 提示 |
| Telegram outbox | SQLite 本地表结构 | `dca-live-report` | 同一服务的发送循环 | 幂等、重试、限速和 message ID 审计 |

## 4. v22 实时合同

### 4.1 文件与时效

Grid producer 原子写入 `ethbtc_forced_exit_observation.json` 或 live gate 输出；DCA 通过只读挂载读取同一文件。合同约每30秒刷新，默认 `stale_after_seconds=150`。

顶层必须包含：

- 身份：`schema`、`package_id=ethbtc-forced-exit`、`model_version`、`execution_policy_version`；
- 时间：`generated_at`、`valid_until`、`stale_after_seconds`；
- 哈希：`release_sha256`、`model_sha256`、`feature_schema_sha256`、`strategy_schema_sha256`、`training_data_sha256`；
- 授权：`execution_authorized`、`observation_mode`、`activation_at`、`approval_receipt_sha256`；
- 安全语义：`market_sell_action=true`、`previous_model_fallback_allowed=false`；
- 状态：`source_healthy`、`runtime_action`、`reason`、`pairs`。

`pairs` 必须恰好只有 `BTC-FDUSD` 和 `ETH-FDUSD`。每对包含：

- `pair`、`source_pair`、`signal_ts`；
- `model_week`、`week_start`、`week_end`、`week_model_sha256`；
- `probability`、`entry_threshold`；
- `risk_off_active`、`recommended_buy_enabled`；
- `buy_enabled`、`force_exit`；
- `transition`、`reason`、`event_id`。

### 4.2 建议、授权和执行不是同一个字段

正常有效合同时：

```text
recommended_buy_enabled = NOT risk_off_active
buy_enabled             = execution_authorized AND recommended_buy_enabled
force_exit              = execution_authorized AND risk_off_active
```

因此：

- `recommended_buy_enabled=true` 只是模型建议 Risk-On；
- `execution_authorized=false` 时，即使模型建议 BUY，v22 也没有实盘权限；
- `risk_off_active=true` 只有在授权生效后才成为健康的 v22 强制退出动作；
- 合同缺失、过期、哈希错误、周过期或字段矛盾会生成 Fail-Closed 视图：禁止新增风险，live 执行层退出并锁存，不能回退旧周、v21、ROC 或 SQZMOM。

`event_id` 由 schema、release、交易对、信号时间和转换计算 SHA-256。Grid/DCA 应看到同一来源事件 ID；DCA 只改变交易对名称，不改变事件身份。

### 4.3 严格校验

消费端拒绝以下合同：

- schema、package、模型或执行策略版本不匹配；
- 任一必需 SHA-256 不是64位十六进制；
- `generated_at` 在未来超过10秒，年龄超过150秒，或超过 `valid_until`；
- BTC/ETH 缺失、多出其他交易对或映射错误；
- 概率/阈值不在 `[0,1]`；
- 信号不在签名周内，或签名周已经结束；
- Risk-Off 与建议 BUY 自相矛盾；
- 实际 `buy_enabled/force_exit` 与授权逻辑不一致；
- 声明允许回退旧模型，或没有声明市价退出能力。

## 5. 授权回执和观察状态

### 5.1 `ethbtc-forced-exit-authorization-v1`

审批回执至少绑定：`package_id`、release/model 哈希、操作人、显式确认串、`approved_at`、未来分钟边界 `activate_at`、签名周结束时间、24小时观察报告哈希、账户预检哈希、自动重入授权和 `consumed` 状态。

producer 只在以下条件同时满足时将 `execution_authorized` 设为 true：

- 回执与当前候选 release/model 完全匹配；
- 观察报告和预检均存在且通过；
- `activate_at` 是不早于审批时间的分钟边界；
- `activate_at < effective_end`；
- 当前时间已经到达 `activate_at`。

Telegram 的“候选请求审批”消息不是授权回执；Hermes 提示词也不是授权。授权只能由 OCI 本地审批流程生成的哈希绑定文件提供。

### 5.2 `ethbtc-forced-exit-observer-status-v1`

observer 状态只保存 release、循环次数、最近看见时间、来源/完整性错误数、`source_healthy`、`execution_authorized` 和逐资产事件 ID。它用于证明 Grid 与 DCA 看见相同信号，不直接控制订单。

历史观察阶段的 `docker-compose.ethbtc-observe.yml` 使用 `run_guard_with_v22_observation.py`：

- 在同一容器中启动未修改的 legacy Guard；
- 旁路运行无副作用 v22 observer；
- Grid observer 故意使用不存在的 `observation-mode-no-authorization.json`；
- 健康检查要求 `execution_authorized=false`。

该语义只适用于历史观察产物。当前生产是 live 模式：`execution_authorized=false`
表示授权不成立，必须 Fail-Closed；不能继续或回退 legacy 技术模型。

## 6. FOMC/宏观合同

宏观主状态 `schema_version=3` 保存 `decisions`、`leases`、`desired_gates`、`bot_gate_state`、`last_reconcile`、重试和审批回调。

Grid scheduler 将其转换为 `grid-fomc-gate-v1`：

- `pause_new_orders=true` 表示暂停 Grid 新单；
- `active_lease_ids` 必须与暂停状态一致；
- 只有已批准、未撤销且当前处于 `effective_at <= now < resume_at` 的 FOMC 租约生效；
- 合同缺失、schema错误、时间在未来或超过150秒时，Grid Fail-Closed 暂停新单。

DCA 直接读取宏观主状态的 `desired_gates.buy/sell`，再写入各 controller 的 `macro_buy_enabled`、`macro_sell_enabled` 和决策 ID。一个门恢复不得覆盖另一个仍关闭的门。

## 7. 七类风险机制与最终权限

七类机制为：

1. `v22_weekly_buy_gate`
2. `fomc_gate`
3. `strategy_loss_breaker`
4. `strategy_drawdown_breaker`
5. `portfolio_loss_breaker`
6. `portfolio_drawdown_breaker`
7. `position_protection`

普通 BUY 最终权限是所有已启用 BUY 门的逻辑 AND。SELL、止损、撤单和紧急退出不受普通 BUY 门阻塞；但在 `EXITING/COOLDOWN/REENTRY` 中，为防止退出与新订单竞态，执行覆盖层会临时关闭双侧。

详细阈值见 [RISK_MECHANISMS.md](RISK_MECHANISMS.md)。强制退出和恢复见 [FORCED_EXIT_AND_RECOVERY.md](FORCED_EXIT_AND_RECOVERY.md)。

## 8. 恢复状态合同

新恢复接口使用：

```text
ACTIVE → EXITING → COOLDOWN → REENTRY → ACTIVE
                  └──────────────► LATCHED（完整性/基础设施）
```

状态持久化 `phase`、`mechanism`、`scope`、触发时间和值、信号价格、退出目标、剩余库存、退出完成时间、`cooldown_until`、连续健康次数、重入状态和新风险周期基准。

冷却时间：技术门0、持仓保护30分钟、策略熔断6小时、组合熔断12小时；进入重入前还要求连续3个健康周期和其他所有门放行。`LATCHED` 不自动恢复。

当前 Grid runtime 已迁移到 schema 8，持久化逐对与组合恢复状态。旧备份仍可能只有
`ledger.halted` 和 `portfolio_tripped`；恢复旧备份时必须走兼容迁移，不能用旧文件
覆盖当前状态。无论 schema 版本，`ledger.halted=true` 都是真实交易阻塞，不能仅因
当前盈亏回到阈值内就假定已经解除。

## 9. Telegram 事件和发送合同

`ethbtc-telegram-event-v1` 是通知合同，不是交易状态合同。标准字段包括：

- `event_id`、`occurred_at`、`source`；
- `strategy`、`bot`、`pair`、`mechanism`；
- `transition`、`phase_from`、`phase_to`、`severity`；
- `reason`、`action`、触发值和阈值；
- release/model/parameter 哈希；
- `requires_manual_action`、附件和扩展详情。

允许的生命周期转换是 `TRIGGERED`、`EXITING`、`EXIT_COMPLETE`、`COOLDOWN`、`REENTRY`、`RECOVERED`、`LATCHED`、`EXIT_DELAY`、`ACTION_FAILED`，另有参数候选/激活/维持旧参数、证据缺失和四小时收益报告事件。

相同来源事件使用稳定 `correlation_id` 生成相同 `event_id`，避免每个 Guard 循环重复告警。`dca-live-report` 是唯一发送器：

- outbox 去重键为事件 ID、附件哈希、类型和频道 ID；
- 发送成功保存 Telegram message ID；
- 失败指数退避，重启继续；
- 附件发送前重新校验 SHA-256；
- 通知失败不进入交易决策链。

`LATCHED` 或需要人工处理的 `REENTRY` 可附 Hermes 提示词。提示词只发起私聊只读预检；频道消息本身没有 reset 权限。

## 10. 如何判断“现在是否被阻塞”

按以下优先级读取真实状态：

1. **策略/执行器状态**：Grid `ledgers.<pair>.halted`、`portfolio_tripped`、活动订单和 `grid_states`；DCA controller 实际 BUY/SELL 开关、executor 和止损状态。
2. **Guard 恢复状态**：`tripped`、`recovery.phase`、退出是否完成、剩余风险和冷却时间。
3. **宏观门**：Grid `pause_new_orders`；DCA `desired_gates` 和 `bot_gate_state`。
4. **技术合同**：先确认容器是 live 还是 observe；只有 live 且已授权的 v22 合同才有执行权。
5. **健康和完整性**：合同年龄、哈希、事件一致性、行情/API/数据库状态。
6. **Telegram 事件**：仅用于解释和审计，不能单独证明当前仍阻塞。测试消息更不能作为状态来源。

典型判读：

- live 合同 `execution_authorized=false`：授权失效，必须 Fail-Closed；历史 observer 文件不参与当前权限计算；
- `ledger.halted=true`：该 Grid 交易对真实停止，即使 v22、FOMC均放行；
- `DCA tripped=false`、controller BUY/SELL=true、恢复阶段 ACTIVE：DCA 未被熔断；
- FOMC无活动租约且合同新鲜：宏观门放行；
- TEST_ONLY Telegram 事件：永远不改变上述任一状态。

## 11. 运维原则

- 不通过编辑 Telegram 事件、observer 状态或报告来解除交易限制；
- 不用关闭普通机制开关绕过模型哈希、授权、资金归属或完整性互锁；
- 不在 v22 故障时回退上一周、v21、ROC 或 SQZMOM；
- reset 前必须复核退出完成、无活动订单、归属库存、合同新鲜度和其他门；
- 每次 schema 变化必须升级版本、同时更新 producer/consumer测试和本文档。
