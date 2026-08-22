# 配置、合同与上线运维

## 容器职责

当前 OCI 已使用 v22 `live` 模式。`grid-live-guard` 在现有容器内作为唯一 v22
producer 加载模型、行情和连续状态，发布有执行权的合同；Grid 直接消费，
`dca-live-guard` 通过只读共享目录映射消费且不加载模型。当前模型身份见
[ONLINE_MODELS.md](ONLINE_MODELS.md)，容器链路见
[CONTAINERS_AND_SIGNAL_FLOW.md](CONTAINERS_AND_SIGNAL_FLOW.md)。

生产中不得同时运行第二个 v22 producer。完成切换后 v21 producer 必须关闭，不能作为故障回退。

## 核心合同

合同包含 `package_id`、执行策略版本、release/model/feature/strategy/data哈希、生成时间、有效期、审批回执哈希、激活时间和逐资产信号。

每对包含 `risk_off_active`、`recommended_buy_enabled`、实际 `buy_enabled`、`force_exit`、概率、fold-local阈值、模型周、转换、原因和事件ID。DCA 必须逐事件验证 FDUSD→USDT 映射一致。

完整 schema、授权计算、FOMC合同、恢复状态、Telegram事件及线上阻塞判定见 [CONTRACTS_AND_RUNTIME_FLOW.md](CONTRACTS_AND_RUNTIME_FLOW.md)。当前是 live 模式；若 `execution_authorized=false`，必须 Fail-Closed，不能继续 legacy 交易。

## 风控开关

每个机制分别使用：

- `GRID_RISK_<MECHANISM>_ENABLED`
- `DCA_RISK_<MECHANISM>_ENABLED`

`<MECHANISM>` 为 `V22_WEEKLY_GATE`、`FOMC_GATE`、`STRATEGY_LOSS_BREAKER`、`STRATEGY_DRAWDOWN_BREAKER`、`PORTFOLIO_LOSS_BREAKER`、`PORTFOLIO_DRAWDOWN_BREAKER`、`POSITION_PROTECTION`。

其他关键配置：`GRID_V22_EXECUTION_MODE`、`GRID_V21_IN_GUARD_ENABLED`、`GRID_V22_PACKAGE_PATH`、`GRID_V22_AUTHORIZATION_PATH`、`DCA_V22_GATE_PATH`、`DCA_RISK_AUTO_REENTRY_ENABLED`，以及 Grid 策略配置 `risk_auto_reentry_enabled`。

## 周模型审批

默认每周在旧周边界前16小时生成候选，随后进入12小时复核窗口。频道通知包含报告、截止时间和 Hermes 提示词；明确拒绝会终止候选，明确批准可提前完成审批，无操作时仅在所有硬门槛持续通过后默认批准。审批等待不修改当前 release 或 controller，不撤单、不成交，也不暂停当前模型交易。

关键配置为 `V22_WEEKLY_AUTO_UPDATE_ENABLED=true`、`V22_WEEKLY_DEFAULT_APPROVAL_DELAY_SECONDS=43200`、`V22_WEEKLY_GENERATION_LEAD_SECONDS=57600`、`V22_WEEKLY_MINIMUM_RUNWAY_SECONDS=86400`、`V22_WEEKLY_RETAIN_OLD_RELEASES=3` 和 `V22_RUNTIME_ROOT=/workspace/state/v22-runtime`。候选在 `T-16h` 生成，保留 12 小时无人拒绝自动批准语义；随后在周边界前 35 分钟隔离预热、前 30 分钟原子提交 runtime generation，周边界不再切换实时文件。授权绑定 release、模型哈希、复核请求、账户预检、审批方式和未来周边界；完整性、连续性、资金归属、紧急通道或过滤器任一失败时拒绝默认通过。

## 原子切换

日常周切换不重启交易容器。候选批准后继续使用旧 release；到 `activate_at` 时，调度器原子替换发布族根目录的 `active_deployment.json`，该文件同时内嵌哈希绑定授权并引用不可变 release。Grid producer 每轮读取该单一指针，DCA 消费 Grid 发布的同一合同，因此不会先切模型后切授权，也不会因审批等待关闭当前交易。

健康周边界激活完成后才执行 release 保留：当前 release 不计入旧模型数量，另保留最近
3 个旧 release。被清理 release 对应的非活动 runtime generation 同步删除；当前和前一代
安全指针受保护。模型目录清理不删除 Telegram 证据 PNG、交付回执、审批记录或审计事件。
清理失败只产生告警并等待下次健康更新重试，不改变已激活模型和交易权限。

## 激活后检查

10分钟、1小时、24小时分别检查：合同年龄、事件一致性、controller状态、订单取消、退出延迟、成交数量、手续费、滑点、dust、剩余风险、恢复阶段、归属余额和权益。

## Grid 容器重启与订单对账

Grid 进程收到 Docker `SIGTERM` 或 `SIGINT` 后，必须先设置停止标志，立即禁止新的
Grid 订单，再由 Hummingbot `TradingCore.shutdown` 作为唯一撤单写入者撤销未成交订单、
停止连接器并落库。信号处理使用35秒内部上限，必须早于 Docker 的45秒强制终止期限
结束；正常结果为退出码0。禁止在策略 `on_tick` 与交易核心之间并发撤同一订单，以免
产生重复撤单和“订单不存在”的误告警。

每次进程启动都执行不可持久化绕过的订单对账：前30秒禁止创建网格，扫描
`BTC-FDUSD`、`ETH-FDUSD` 的策略活动订单和连接器恢复订单并撤销，随后还需连续三个
周期确认无遗留订单，才允许建立一组新网格。Grid 独占这两个 FDUSD 交易对；DCA 使用
USDT 交易对，因此该扫描不得扩大到账户其他交易对。

新网格的 BUY 名义金额不得超过交易所实时可用 FDUSD，SELL 数量不得超过交易所实时
可用 BTC/ETH，同时仍受 `capital_reservations` 归属账本限制。不得用账户总余额替代可用
余额，因为同一账户可能还存在 DCA 归属库存或其他锁定余额。

重启验收至少包含：停止前存在一组正常挂单；停止后退出码为0且两对交易所挂单均为0；
重启后12秒仍为0单；30秒对账期及三个静默周期完成后仅出现一组新网格；日志不得出现
余额不足、重复撤单或退出阶段重新建单。若优雅撤单失败，保持 Grid 停止并由
`grid-live-guard` 独立 Binance 通道只撤销上述两个 Grid 专属交易对，确认归零后才可重启。

## Plotly 审计

Grid/DCA、BTC/ETH 分页展示各自权益，不使用组合权益冒充单对权益。七类机制各有独立阴影、标记、图例和复选框；v22 另有 BTC/ETH 子开关。隐藏机制图层不得隐藏价格、权益、峰值、回撤、概率或阈值。无可信事件显示“无数据”，不推测区间；HTML必须自包含UTF-8且不显示“机制事件数”或原始JSON大块区域。

## 回滚

先撤销授权并保持双侧关闭，再取消订单、复核和清理归属库存。回滚镜像不得自动恢复 v21、ROC、SQZMOM或旧周。只有新的有效周、完整预检和新审批完成后才可重新入场。
