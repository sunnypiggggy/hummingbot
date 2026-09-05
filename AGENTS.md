# 项目 Agent 工作指南

本文件适用于整个仓库。项目是在 Hummingbot 基础上开发的 Grid、DCA、Stock 交易与风控系统，包含研究回测、OCI 运行组件和 Telegram 管理界面。使用中文沟通，先说明结果和交易影响，再提供必要证据。

## 开始工作与事实核验

- 先执行 `git status --short`、`git branch --show-current`，检查相关文件的未提交差异；保护用户和其他任务的修改，不覆盖、不顺手清理、不将无关文件纳入提交。
- 按任务定位模块，优先使用 `rg`；初次搜索排除大型结果、发布包归档和运行数据，只有任务涉及这些证据时再定向读取。
- 源码说明实现，配置说明意图，当前运行合同和实际进程说明生效状态。冲突时核对合同生成时间、来源、模型授权、容器镜像和加载版本，不凭某一个字段下结论。
- 历史日志不是当前故障；磁盘脚本已更新不代表 Python 进程已加载；配置默认值不代表运行参数；容器健康不代表正常交易。交易结论还需检查最终 BUY/SELL 权限、恢复阶段和控制器落地状态。
- 以当前任务及会话授权确定变更范围。查询、分析不等于授权下单、解锁、重训或部署；已有明确授权不重复索要，但不能把历史的一次实盘授权当成后续所有模型或操作的授权。
- 不将当前模型哈希、周数、库存、余额、订单数、开关状态或截止时间写成永久规则；报告这类信息时标明采集时间和证据来源。

## 模块与资料入口

| 工作内容 | 优先检查的代码 | 说明资料 |
|---|---|---|
| Hummingbot 核心、连接器、Executor | [hummingbot/](hummingbot/)、[controllers/](controllers/) | [README](README.md)、[上游贡献指南](CONTRIBUTING.md) |
| Grid 实盘策略与参数调度 | [实盘 Grid](scripts/walk_forward_portfolio_grid_live.py)、[Grid scheduler](scheduler/fdusd_live_grid_scheduler.py) | [Grid 参数与切换](release_packages/ethbtc-forced-exit/documentation/GRID_PAIR_PARAMETER_CUTOVER.md) |
| DCA 策略与控制器 | [DCA controller](controllers/market_making/dman_maker_v3_macro.py)、[DCA Guard](live_guard/dca_live_guard.py) | [DCA 止损与恢复](docs/DCA_STOP_LOSS_AND_SELL_TREND_RECOVERY.md) |
| 风控、退出、统一库存 | [Grid Guard](live_guard/grid_live_guard.py)、[库存账本](live_guard/account_inventory.py)、[恢复状态机](scripts/risk_recovery.py) | [机制总表](release_packages/ethbtc-forced-exit/documentation/RISK_MECHANISMS.md)、[库存归属](release_packages/ethbtc-forced-exit/documentation/ACCOUNT_INVENTORY.md) |
| v22 周模型、合同与发布 | [周模型](scripts/xgboost_long_risk_gate_v22.py)、[发布管理](scripts/v22_weekly_release_manager.py)、[消费合同](scripts/ethbtc_forced_exit_contract.py) | [周模型说明](release_packages/ethbtc-forced-exit/documentation/V22_WEEKLY_MODEL.md)、[原子切换](release_packages/ethbtc-forced-exit/documentation/V22_ZERO_DOWNTIME_CUTOVER.md) |
| Stock Runtime、PAPER 与订单 | [stocks_runtime/](stocks_runtime/)，重点为 router、ledger、async_orders | [Runtime](docs/BINANCE_STOCKS_RUNTIME.md)、[PAPER](docs/BINANCE_STOCKS_PAPER_TRADING.md) |
| Telegram 私聊管理 | [management_bot/](management_bot/) | [管理 Bot](docs/TRADING_MANAGEMENT_BOT_V3.md) |
| 频道报告、统一交易状态 | [报告服务](live_guard/dca_live_report.py)、[通知](live_guard/telegram_notifications.py)、[状态口径](live_guard/trading_status.py) | [通知说明](release_packages/ethbtc-forced-exit/documentation/TELEGRAM_NOTIFICATIONS.md) |
| FOMC 与人工恢复 | [macro_control/](macro_control/)、[Hermes](hermes/) | [宏观控制运维](ops/dca-macro/README.md)、[恢复审批](hermes/skills/dca-macro-control/references/risk-recovery-approval.md) |

完整生产机制资料见[发布族文档索引](release_packages/ethbtc-forced-exit/documentation/README.md)，容器依赖见[容器与信号链路](release_packages/ethbtc-forced-exit/documentation/CONTAINERS_AND_SIGNAL_FLOW.md)，合同语义见[合同与运行链路](release_packages/ethbtc-forced-exit/documentation/CONTRACTS_AND_RUNTIME_FLOW.md)。文档中的带日期状态也需重新核验。

### 历史资料的适用范围

[v22 Agent 操作说明](XGBOOST_V22_WEEKLY_RISK_GATE_AGENT_GUIDE.md)和[v22 Agent 交接](XGBOOST_V22_AGENT_HANDOFF.md)主要描述早期离线冻结包、BUY-only 研究语义及当时的 `NO-GO`。保留它们作为历史证据；不能据此断言当前生产授权、依赖关系、模型有效期或强制退出行为。

生产执行覆盖层与冻结模型是不同对象。解释实际行为时同时检查当前代码、发布族机制文档、有效运行 generation 和审批合同；发现冲突先报告并查证，不改写历史报告或凭文档修改授权。[CONTRIBUTING.md](CONTRIBUTING.md)描述上游贡献流程，不能机械用于本项目分支合并或 OCI 发布。

## 开发与交易边界

- 优先复用已有模块和合同；研究模型、参数实验与正式运行逻辑保持隔离，禁止因回测好看或测试通过自动开启实盘。不要未经任务要求切换分支、合并实验代码或增加容器。
- Grid、DCA、Stock 分别核算机器人归属库存和收益；FDUSD、USDT、USDC 不直接相加，策略 MTM 不冒充交易所账户总收益。
- 出售数量以归属账本及最新成交、余额、订单核对为依据，禁止按账户总余额清仓或借其他机器人的库存补缺口。跨交易对撤单、组合退出、并发租约及重试幂等均需验证。
- 模型授权、当前签名周、PAPER/LIVE、普通交易门和保护性退出权限分别检查。一个机制恢复不能覆盖其他阻塞；模型缺失或失效不能擅自回退旧模型。具体故障宽限、锁存和恢复规则见[韧性策略](release_packages/ethbtc-forced-exit/documentation/RESILIENCE_POLICY.md)与[退出恢复](release_packages/ethbtc-forced-exit/documentation/FORCED_EXIT_AND_RECOVERY.md)。
- 保持[禁止 BNB 抵扣手续费规则](release_packages/ethbtc-forced-exit/documentation/NO_BNB_FEE_POLICY.md)。费用比例、过滤器、限额取相应权威来源，不从截图或历史报告推断。
- Stock 的行情、白名单、资金、成交和风控最终校验由 Runtime 承担。PAPER 撮合应验证真实经济请求为零；不得为测试开启实盘或重置已有 Paper run。

## Telegram 与报告约定

- 私聊管理由 `sunnypiggy-trade-bot` 负责，周期频道通知由 `dca-live-report` 负责。两套 Token 和用途隔离；管理 Token 只能有一个更新消费者，通知服务不消费 `getUpdates`。
- 优先 Inline 导航和编辑原消息。股票、方向、金额及订单标识必须明确；展示真实预检、撤销和成交结果，超时不能当成功，查看和刷新不能隐式推进订单激活。
- 按机器人显示“正常交易／交易受限／停止交易／数据不可用”，区分普通订单权限与保护性退出；阻塞提供中文原因、解除条件和可信时间。冷却截止不等于恢复交易时间，模型等待及人工解锁不能伪造倒计时。
- 使用已有富文本发送方式并转义动态内容；长内容分页。未知参数不补当前默认值，缺少证据不借用旧模型图片，缺少行情不把持仓按零估值。
- 展示口径优先复用报告服务的状态与收益合同；检查数据年龄、窗口完整性和对账结果。告警但不阻塞的门不能渲染为停止交易，已恢复历史事件不能继续当当前故障。

## 本地环境与验证

Windows 工作区使用 PowerShell，OCI 使用 Linux shell；路径、引号和变量展开必须按执行环境处理。不要将 PowerShell 路径列表拼接给另一种 shell 做删除或移动。

测试命令从仓库根目录执行，先确认 Python 环境。基础运行依赖见[环境定义](setup/environment.yml)与[项目配置](pyproject.toml)；独立服务使用各自依赖，例如[管理 Bot](management_bot/requirements.txt)、[Guard](live_guard/requirements.txt)、[scheduler](scheduler/requirements.txt)。涉及 Hummingbot/Cython 的测试可能需要已编译环境，不能把缺依赖导致的收集失败报告为测试通过。

按改动选择相应组，而不是每次执行全仓测试：

```powershell
# Telegram 导航、风控详情和待开市订单
python -m pytest test/test_trading_management_bot.py test/test_management_risk_display.py test/test_management_scheduled_display.py -q

# Grid/DCA 执行、恢复与库存
python -m pytest test/test_grid_live_safety.py test/test_grid_live_runtime_risk.py test/test_risk_recovery.py test/test_account_inventory.py test/test_dca_live_safety.py -q

# v22 周模型与运行 generation
python -m pytest test/test_xgboost_long_risk_gate_v22_weekly.py test/test_v22_weekly_release_manager.py test/test_v22_runtime_generation.py -q

# Stock 配置、PAPER 撮合与异步计划
python -m pytest test/stocks_runtime/test_executor_config.py test/stocks_runtime/test_paper_broker.py test/stocks_runtime/test_async_orders.py -q

git diff --check
```

以上是测试入口，不是历史通过承诺。新增行为要验证真实数据流和失败路径；订单与清仓改动需覆盖部分成交、超时、重启、重复回调和资金边界。需要隔离 HTTP 或容器演练时按[真实场景测试说明](release_packages/ethbtc-forced-exit/documentation/REAL_SCENARIO_TESTING.md)执行，不使用生产凭证或生产状态。不要因小范围修改自动启动长时训练、全量回测或全仓格式化。

## 构建与 OCI 发布

当前 [Makefile](Makefile) 的 `make build` 会先执行 `git clean -xdf`，不能作为普通构建入口直接运行；这可能删除未跟踪的源码、配置和数据。使用指定目标的命令，例如：

```powershell
# 只构建指定组件；不启动容器
docker build -f Dockerfile.trading-management-bot -t hummingbot/trading-management-bot:local .
# 或按 Compose 指定服务构建
docker compose --profile telegram build sunnypiggy-trade-bot
docker compose --profile telegram config --quiet
```

发布必须在当前任务授权范围内，按以下步骤操作：

1. 从 SSH 配置和远端部署检查确认主机、目录、Compose service 与实际容器名称；不假设 OCI 有 Git 仓库。核对本地与远端相关源码差异，保存旧文件、容器 ID、镜像、启动时间及受影响状态的备份。SQLite 使用一致性备份方式，不把正在写入的单个数据库文件复制视为可靠备份。
2. 构建指定服务，检查相关依赖、挂载和配置；不要打印包含秘密的完整 Compose 展开或容器环境。`dca-live-report`、`dca-live-guard`、`dca-live-manager` 共用镜像标签；Telegram 管理 Bot 使用独立镜像。构建新标签不等于已运行容器更新。
3. 只更新本次明确涉及的服务。以下仅为管理 Bot 发布示例，需在已确认的 OCI 部署目录内执行：

   ```sh
   docker compose --profile telegram up -d --no-deps sunnypiggy-trade-bot
   ```

4. 检查健康、日志、合同新鲜度和实际页面或行为；核对部署文件 SHA256、镜像及进程加载版本。确认非目标交易容器 ID 和启动时间未改变。若策略脚本通过挂载更新，单有文件哈希一致仍不足以证明运行代码一致。
5. 出现异常使用本次备份对目标组件回滚，保留诊断证据；不要顺带启动停止的机器人、解除锁存或改变模型授权。交付说明改动、验证结果、部署/提交状态和仍未确认的事项。

## 密钥、数据与发布包

- 不输出或提交 Token、API Secret、私钥、完整环境文件或生产数据库。只读取任务所需字段；错误日志和命令不得泄露带 Token 的 URL。测试使用专用假凭证。
- 区分源码、运行状态、研究结果与不可变发布证据。`results/`、`data/`、`logs/`、构建目录和未跟踪文件不等于可删除垃圾，清理必须有明确范围。
- [发布包目录](release_packages/)中的内容寻址 release、manifest 和冻结报告不可原地改写。新增版本使用相应封包流程并校验依赖与哈希；发布族文档可以随源码维护，已封包副本保持不变。
- 提交前审查差异和文件范围；不自动 `git add .`、提交、推送或部署。文档整理任务不应改变交易代码、运行状态或历史发布包。
