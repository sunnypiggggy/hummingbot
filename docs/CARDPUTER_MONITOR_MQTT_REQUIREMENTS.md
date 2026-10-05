# Cardputer 交易监控 MQTT 需求与部署交接

更新日期：2026-10-05，Asia/Shanghai。

本文交给负责实现与部署的 Agent。用户已决定：由 OCI 的现有 `dca-live-report` 容器直接向 `sh.sunnypiggy.top:8883` 发布交易遥测，云端接收后继续通过 HTTPS 向 Cardputer 提供数据。**本方案不新增 OCI 的 `cardputer-monitor` 容器**；该名称指交易监控功能及可复用的本地模块。

现有独立采集器保留 HTTPS CLI。**2026-10-05 10:39（Asia/Shanghai）报告内 MQTT 发布的 v2 版已上线并实际收到两份连续真实快照**：四行状态/收益 FRESH、4h/24h/7d/累计窗口和 12h 收益/价格曲线符合协议，消息约 38 KiB；四条价格曲线各 73 个可信非空点。v2 修复按开盘时间查询 K 线时，12h 左端少取一根已闭合 K 线的边界问题。云端交易接收落库、设备接口及实机显示仍由接收 Agent 验证，不能把订阅成功当作完整链路验收。OCI 运行目录、版本、回退及证据见 [Report MQTT 部署交接](CARDPUTER_REPORT_MQTT_DEPLOYMENT.md)。

用户先授权仅上线公网 MQTT 服务及交易 Topic 权限，随后新增授权本任务适配并上线 OCI 报告容器；云端接收仍由其他 Agent 完成。**2026-10-05 Broker 已上线 64 KiB 限额、64 条有界队列与专用账号，真实公网验证通过**；接入文件、验证结果与回滚见 [交易 MQTT 服务上线记录](../../serverdocker/cloud/docs/MQTT_TRADING_DEPLOYMENT.md)。报告使用现成发布角色凭据，没有生成或轮换账号。

## 1 交付范围与界面行为

交付内容是报告容器中的异步 MQTT 发布模块、云端交易订阅与落库、部署配置及验证记录。Cardputer 继续使用原生 ESP-IDF 固件和 `/api/cardputer/v1` HTTPS 接口。

- 首页四宫格：Grid BTC-FDUSD、Grid ETH-FDUSD、DCA BTC-USDT、DCA ETH-USDT，各自只显示最近 4h 收益及币种。
- `1/2/3/4` 直接进入对应交易单元，显示最近 12h 收益曲线与对应交易对的市场价格曲线；进入交易 App 时提前拉取四组数据。
- 曲线标出实际观测到的交易停止或限制区间及原因，区分限制买入与全部交易停止；界面不添加“阴影：”字样。
- 每个详情页底部显示最终状态及原因，Enter 可查看风控详情，Tab 返回四宫格，G0 返回桌面。
- 首版沿用约 60 秒报告采样及设备交易刷新周期。MQTT 只改变传输方式，不提高源数据采样频率。

## 2 容器与数据来源

四个交易单元对应三个策略实例。实例由 `hummingbot-api` 动态创建，不是根 Compose 中的四个静态服务；部署时核验实际名称与挂载。

| 固定 ID | 交易单元 | 原始策略实例 |
| --- | --- | --- |
| `grid:BTC-FDUSD` | Grid BTC-FDUSD | `grid-live-fdusd-400` |
| `grid:ETH-FDUSD` | Grid ETH-FDUSD | `grid-live-fdusd-400`，与 BTC 共用实例 |
| `dca:BTC-USDT` | DCA BTC-USDT | `dca-live-btcusdt-200` |
| `dca:ETH-USDT` | DCA ETH-USDT | `dca-live-ethusdt-200` |

| 来源组件 | 职责及输出 |
| --- | --- |
| 三个策略实例 | 成交 SQLite、归属库存账本、实际 runtime/controller 状态，以及 `management_trading_snapshot.json` |
| `grid-live-guard` | Grid 归属库存估值、风控与恢复状态，输出 Grid `guard_state.json` |
| `dca-live-guard` | DCA 风控、归属库存及紧急调整、恢复状态，输出 DCA `guard_state.json` |
| `grid-live-fdusd-scheduler` | 网格参数和宏观门控辅助输入；不是四个交易单元收益及最终权限的统一来源 |
| `dca-live-report` | 综合上述数据，写统一收益库、最终交易状态和脱敏详情；作为本方案唯一交易快照发布者 |

报告容器内统一目录为 `/workspace/state/telegram`，本地 Compose 对应宿主机 `./dca-live-data/telegram`。主要文件为：

- `telegram_outbox.sqlite` 的 `profit_snapshot`：按策略和交易对保存累计 MTM、权益、回撤及报告采样时间 `observed_at`；该时间不等于底层成交或行情时间，底层数据年龄仍需由报告检查。
- `trading_status.json`：四行最终状态、普通买卖权限、阻塞原因和恢复阶段，采用 JSON 原子替换。
- `management_trading.json`：当前行情参考值、挂单/执行器等脱敏详情，可供以后扩展；当前界面不要求上传完整订单或成交列表。

本地另有正在修改的 `live_guard/risk_history.py`，报告周期已接入风险归档。其权限历史需要部署 Agent 核对运行版本；当前 `RiskHistoryReader.intervals` 主要面向指定 v22 门控，不能直接当作 Cardputer 全部 STOPPED/RESTRICTED 区间。**收益库没有最终权限历史，不等于整个报告服务没有风险归档。**

## 3 必需数据与口径

复用 `cardputer_monitor/collector.py` 的脱敏逻辑与 `OperationsReportReader`，避免另建收益计算口径。Grid 与 DCA 均使用策略归属 MTM，不能以成交额或账户总余额替代；FDUSD 与 USDT 分别显示，不直接相加。

| 数据组 | 必需字段或事实 | 来源及处理 |
| --- | --- | --- |
| 身份 | `id/name/bot_name/pair/quote_asset` | 固定四行映射；单行缺失仍保留位置 |
| 4h 收益 | `profit["4h"]`、窗口完整性、收益源时间 | 统一累计 MTM 与可信 4h 基准的差值；样本不足为 `null` |
| 其他收益 | `profit` 的 `24h/7d/all`、权益、回撤 | 保留现有协议，首页仍只显示 4h |
| 12h 收益曲线 | `profit_points`、窗口起止、收益源时间、完整性 | 从 `profit_snapshot` 读取历史，以 12h 左端可信值为基准；最多 73 点，每 10 分钟一点 |
| 12h 价格曲线 | `price_points`、行情源时间 | 四个交易对分别读取已闭合的公开 5m K 线，再按同一 10m 时间轴采样 |
| 最终状态 | `status/process_running/trade_mode/system_health/phase` | 读取统一状态合同；容器健康不等于正常交易 |
| 权限及原因 | `buy_enabled/sell_enabled/blockers/gates` | 普通买卖权限、中文脱敏原因；保护性退出权限不能冒充普通交易权限 |
| 恢复 | `recovery` 的阶段、冷却截止、退出完成时间、健康周期、重入阻塞原因 | 保留真实状态；冷却到期不等于已经恢复交易 |
| 停限区间 | `start_at/end_at/status/scope/reason`、`status_coverage_start_at/status_coverage_end_at` | 首版沿用云端连续可信状态观测，`scope=buy/all`；覆盖起点前、终点后及缺口保持未知 |
| 新鲜度 | 独立状态/收益/行情源时间、`sample_id/collected_at`、缺样标记 | 新的上传或 retained 消息不能更新旧源数据年龄 |

行情适配复用 `cardputer_monitor/history.py` 的固定交易对白名单。正式环境显式开启 `binance-public`，不需要交易所 API Key。报价超时或失败只降级价格曲线，不能丢弃有效收益、状态，也不能阻塞报告周期。

四行曲线一次组装为同一遥测快照；读取收益时使用独立只读 SQLite 连接及短事务，兼容 WAL，关闭事务后再执行网络请求。不要跨线程复用报告服务现有的 SQLite 连接或 HTTP session。当前报告逐行提交收益，状态 JSON 另行替换；保留每行来源时间及部分降级，不声称跨文件原子一致。

首版仅展示云端实际观测到的停限区间。首次连接可回取既有收益和市场行情，不能从当前状态倒推出过去 12h 的停止历史。若后续要求断网期间也完整回填，需要单独设计 OCI 风险归档读取、来源验证及云端回填协议；不要把上传方的 `history.pauses` 直接作为生产阴影依据。

## 4 MQTT 协议

以下 Topic 与 Broker 权限已准备好，消息格式、发布与接收逻辑尚需实现。

| 项目 | 要求 |
| --- | --- |
| 公网 Broker | `sh.sunnypiggy.top:8883`，原生 MQTT over TLS，协议 MQTT 3.1.1 |
| 发布 Client ID | `hummingbot-cardputer-report`，保证只有一个活跃发布实例 |
| 最新快照 Topic | `cardputer/v1/trading/hummingbot-main/snapshot` |
| 发布者状态 Topic | `cardputer/v1/trading/hummingbot-main/availability` |
| 消息设置 | 两个 Topic 均 QoS 1、retained |
| availability | `online/offline`，连接前设置 retained `offline` 遗嘱；正常退出也发布 offline |
| 快照格式 | UTF-8 JSON、`schema_version=1`、`source_id=hummingbot-main`，固定四行 `robots`，沿用现有遥测合同 |
| 最大快照 | 65,536 字节，序列化后检查；禁止 NaN/Infinity，缺失值用 `null` |
| 时间与幂等 | 原始 `collected_at` 及源时间不变；重试同一快照必须沿用相同 `sample_id` 和内容 |
| 刷新与重连 | 约 60 秒一帧；状态无变化也发布；重连退避建议 1–30 秒 |

完整合成示例使用 [trading-history.json](../cardputer_monitor/fixtures/trading-history.json)，基础无历史示例使用 [trading.json](../cardputer_monitor/fixtures/trading.json)。它们是测试数据，不得作为生产 retained 消息发布。新增传输不改变云端验证器要求的机器人名称、币种或字段。

MQTT availability 仅说明报告发布连接状态，不能把发布者 offline 渲染为策略已停止。设备通信状态还需结合最近有效快照时间；状态、收益、价格分别判断 300 秒过期，未来超过 30 秒的时间拒绝或独立降级，约 180 秒没有有效遥测表示通信中断。

QoS 1 允许重复投递，PUBACK 也不等于云端业务已经落库。云端交易 client 使用固定 ID、`clean_session=False`、`manual_ack=True` 与 Broker 持久化；SQLite 提交完成后才确认有效消息。无效或单调冲突的消息确认后丢弃并记录脱敏计数；临时数据库故障保留未确认消息并限时重试，必要时通过持久会话重连恢复投递，不能当作坏消息丢弃。重启后重新订阅和 retained 恢复均需验收；不宣称所有断网历史自动完整补齐。

## 5 报告容器中的实现要求

1. 在 `UnifiedTelegramReporting.update_snapshots` 完成收益记录和统一状态原子写入后，将刷新信号交给后台任务。后台任务组装曲线、取公开行情并发布；报告主循环不等待 MQTT 连接、PUBACK 或行情网络请求。
2. 使用一个有界后台任务和独立 SQLite 只读连接。待组装任务只保留最新请求；待发送快照可以由新快照替代，不能无界积压后倒序覆盖 retained。重试尚未被替代的快照时保留 ID、内容与时间。
3. MQTT 初始化失败、无凭据、DNS/TLS/认证错误及 Broker 中断均只影响遥测。异常隔离，报告、Telegram 通知、现有风险归档和审计继续运行。
4. MQTT 开关独立于 Telegram 开关；即使关闭 Telegram 通知，也能生成并发布交易快照。未显式启用时保持现有报告行为，`--healthcheck` 不建立 MQTT 连接。
5. 正常退出有限时清理后台任务及 MQTT 连接；`--once` 必须明确是否实际发布及是否获得确认，不能在消息未发出时报告成功。
6. 复用或抽取采集器的数据转换模块。现有 `cardputer_monitor/__main__.py --publish` 仍是 HTTPS CLI，不能直接当作 MQTT 发布入口；将所需 Python 包及模块显式打入报告镜像。

建议新增以下配置项，名称可由实现 Agent 统一调整并同步示例；**当前代码尚不识别这些配置**：

| 配置项 | 目标值或用途 |
| --- | --- |
| `CARDPUTER_MQTT_ENABLED` | 默认 false；生产显式设 true |
| `CARDPUTER_MQTT_HOST/PORT` | `sh.sunnypiggy.top` / `8883` |
| `CARDPUTER_MQTT_CREDENTIALS_FILE` | 独立只读 secret，内容为专用 MQTT username/password |
| `CARDPUTER_MQTT_CA_FILE` | 受信 CA 文件；保持证书链及域名验证 |
| `CARDPUTER_PRICE_PROVIDER` | 正式环境 `binance-public`，离线测试 `none` |

依赖可锁定为与现有云端一致的 `paho-mqtt==2.1.0`。不把新增 MQTT 密码放进普通 env、仓库、日志或设备配置；报告容器只获得自身发布账号。

报告与 `dca-live-guard`、`dca-live-manager` 共用 Dockerfile 和镜像标签。增加模块及依赖时核对其他任务的未提交修改，构建独立版本标签；部署只更新明确受影响的报告服务，不能因共享标签而顺带重建或重启交易、Guard、scheduler、manager。

## 6 云端接收与现有接口

现有 `cloud/cardputer/worker.py` 只订阅 GPU Topic，接收分支统一限制 1 KiB，且 client 默认不持久保存会话。必须新增交易订阅、交易消息分支及持久交易 client；仅改 ACL 不会自动接入交易。在同一 worker 中使用独立交易 client 可避免改变 GPU 订阅行为，不需要为交易另建云端订阅容器。云端仍使用已有 `api/cardputer-worker` 服务及共享 SQLite 数据卷。

- 订阅上述两个交易 Topic；交易快照白名单来源固定 `hummingbot-main`，与专用发布账号权限绑定。
- 复用 `cloud/cardputer/telemetry.py` 验证、`store.py` 的单调与幂等写入，再通过同一状态历史逻辑生成停限区间。
- 同 ID 同内容重复消息不更新接收时间或追加状态观测；同 ID 不同内容、迟到快照、相同采集时间的不同快照必须拒绝。
- 捕获协议验证错误、SQLite 冲突 `Conflict`、编码错误等预期异常，不能使 MQTT 回调线程退出；记录原因类型，不输出原始敏感 payload。
- 首版 MQTT 是该来源的唯一生产上传路径。现有 HTTPS 上传接口可保留作工具用途，但不同时启动另一个生产上传者，避免两条链路覆盖快照。
- `/trading`、`/trading/{id}`、`/trading/history/{id}?cursor=0` 保持现有协议；设备只读，无交易命令 Topic。
- HTTPS 摘要及详情/曲线分页保持单页不超过 8 KiB。曲线分页携带 `sample_id`，跨页更换快照时设备重新拉取，不能拼接不同帧。
- MQTT 接收、SQLite 及设备 API 均分别检查源时间；重新连接或收到 retained 不能将过期数据显示为实时。

通信健康还必须检查 `collected_at` 的年龄。新数据库第一次收到已采集超过 180 秒的 retained 快照时，不能因写入时间变成“现在”而恢复通信健康；即使其状态源仍在 300 秒内，也应显示通信中断。可在统一展示逻辑增加采集年龄检查或保守保存接收上界，HTTP 与 MQTT 两种入口都应遵循相同规则。

云端 API、worker、proxy 使用构建镜像，而不是源码挂载。单纯 restart 不会加载本次代码；需构建版本镜像并定向 recreate 使用新版本的 API 与 `cardputer-worker`，核对共享 `CARDPUTER_IMAGE` 对其他服务的影响及既有数据卷位置。现有基础部署脚本是否包含 Cardputer overlay 也必须核对，不能假设业务 API 已经上线。

## 7 Broker 限额与账号权限

更新前 Broker 的 `message_size_limit=1024`；本次已上线 `65536`。现有 `worker.py` 的 GPU 分支仍限制 1024 字节，**交易接收必须新增独立 64 KiB 分支，不能直接沿用 GPU 校验。**

生产 Broker 已采用 `message_size_limit=65536/max_queued_messages=64/max_inflight_messages=1`，保留顺序投递及有界队列，公网容量/权限/GPU 回归已通过。后续接收 Agent 保留 GPU Topic 的 1 KiB 校验，补交易分支与持久会话测试；这些队列设置不提供完整状态事件积压。

| 账号 | 最小权限 |
| --- | --- |
| 新专用账号，建议 `cardputer-report` | 仅 write 两个指定交易 Topic，无 read、通配符或设备控制权限 |
| 云端 `cardputer-cache` | read 两个交易 Topic，并保留既有 GPU Topic 的 read 权限 |
| 现有 GPU 发布/读取账号 | 保留原有账号、密码及 GPU Topic 权限 |
| Cardputer | 使用现有 HTTPS 设备 Token，不取得报告发布账号 |

实际启用的 ACL 可能是基础 `acl` 或 overlay 的 `acl.cardputer`，部署前检查挂载，只增量修改相关权限。`mosquitto/start.sh` 启动时将 ACL 和密码文件复制到 `/run/mosquitto`；只编辑宿主机挂载文件或 reload 不保证运行副本更新，需定向重启/recreate Broker 并核对实际副本。证书由现有公网 TLS 入口管理；OCI 不开放入站端口，云端不新增业务监听端口。Broker 重启后验证 GPU 数据重新连接并持续更新，保留 `mqtt_data` 数据卷。

## 8 实现与部署顺序

1. 核验 OCI 当前 `dca-live-report` 容器、报告目录、源码/加载版本、运行用户及实际 Compose 项目；核验云端 Broker、ACL、设备 API/worker 的实际部署状态。历史部署记录不能替代当前检查。
2. 在本地实现报告异步发布和云端交易接收，先用合成数据与隔离测试 Broker 验证。保护当前工作区已有的风险历史、数据库维护等其他任务修改。
3. 准备专用发布凭据、云端只读账号、CA、显式开关与版本镜像。先部署云端验证器、订阅、限额和 ACL，再启用 OCI 报告发布。
4. 首次真实消息验证四行身份、币种、源时间、曲线与实际状态；确认报告、Telegram、Guard、策略和 GPU 遥测仍按原周期运行。
5. 比对与记录改动前后容器 ID、启动时间、镜像和相关文件 SHA256；非目标交易容器应保持原运行状态。回滚报告版本/配置及云端相关变更时保留 MQTT 和 SQLite 数据卷。

实现后同步更新配置示例、Dockerfile/Compose、相关 README 和验证记录。本文中的待实现项只能在有实际证据后改为已完成；不沿用旧 HTTPS 测试成绩证明新 MQTT 链路已通过。

## 9 验收要求

| 场景 | 通过标准 |
| --- | --- |
| 四个交易单元 | 四个固定 ID、三实例映射、FDUSD/USDT 正确；缺失不填零，合法 0 保留 |
| 收益窗口 | 4h 依赖最新累计 MTM 与可信左端基准；12h 缺基准不可用，单点距前样本超过 300 秒时留 null 并保留其他有效段 |
| 曲线 | 四组提前可取，12h 时间轴一致，最多 73 点；公开行情只用已闭合 K 线 |
| 状态 | 最终权限、阻塞原因及恢复阶段与报告一致；告警不误当停止，冷却结束不误当恢复 |
| 停限区间 | 仅绘制可信连续观测，买入限制与全部停止有区别；coverage 起点前、终点后及采样缺口为未知 |
| SQLite 与 JSON | WAL 并发写入和 JSON 原子替换时读取稳定；后台连接不跨线程复用，网络不持有读事务 |
| 发布故障 | 断网、错误凭据、TLS/行情故障不阻塞报告和 Telegram，后台队列有界，恢复发布最新可信快照 |
| MQTT | TLS、QoS 1、retained、正常/异常离线、重连、发布/接收进程与 Broker 重启均验证 |
| 幂等与时序 | 重复不刷新旧数据年龄；迟到、篡改 ID 内容及非法字段被拒绝；新数据库收到旧 retained 仍显示通信中断 |
| 限额与权限 | 超过 1 KiB 的合法交易曲线成功；超过 64 KiB 拒绝；错误角色不能发布交易或控制消息 |
| GPU 回归 | 两个既有 GPU Topic、权限、UUID、1 KiB 校验及约 2 秒更新保持可用 |
| 设备接口 | 摘要、详情和曲线分页不超过 8 KiB，分页快照一致，数字键与 G0 行为符合要求 |

部署交付记录应包含实际容器/镜像、配置位置及权限、Topic、脱敏消息样本、四行对照、故障与重启测试结果、回滚入口及尚未验证项。实机显示与峰值内存没有实测时明确列为待联调，不写成通过。

以设备 HTTPS 摘要及历史接口实际返回的快照、源时间和曲线作为落库验收证据，不能只看发布端 `publish success` 或 PUBACK。现有 HTTP/GPU 测试不覆盖新交易 MQTT 链路；需增加真实测试 Broker → worker → SQLite → HTTPS 的集成验证。

## 10 代码与文档入口

| 工程 | 入口 |
| --- | --- |
| 报告集成 | [live_guard/dca_live_report.py](../live_guard/dca_live_report.py)、[mqtt_reporter.py](../cardputer_monitor/mqtt_reporter.py)、[专用报告 Dockerfile](../Dockerfile.dca-live-report-mqtt)、[报告 overlay](../ops/compose.cardputer-report-mqtt.yml)、[部署/回退记录](CARDPUTER_REPORT_MQTT_DEPLOYMENT.md) |
| 数据转换与收益 | [cardputer_monitor/collector.py](../cardputer_monitor/collector.py)、[management_bot/clients.py](../management_bot/clients.py) |
| 历史与公开行情 | [cardputer_monitor/history.py](../cardputer_monitor/history.py)、[cardputer_monitor/README.md](../cardputer_monitor/README.md) |
| 风险历史候选 | [live_guard/risk_history.py](../live_guard/risk_history.py)，核对该任务改动及实际部署版本 |
| 云端 | `H:\PycharmProjects\serverdocker\cloud\cardputer`，重点 `worker.py/telemetry.py/store.py/trading_history.py` |
| Broker | `H:\PycharmProjects\serverdocker\cloud\mosquitto`，重点 `mosquitto.conf/acl/acl.cardputer` |
| 固件与接口 | `H:\PycharmProjects\shanghai\cardputer_adv`，重点 `docs/API.md` 与交易 App 实现 |
| 验证入口 | `test/test_cardputer_monitor.py`、`test/test_cardputer_history.py`、`test/test_cardputer_price_boundary.py`、`test/test_cardputer_mqtt_reporter.py`、`test/test_cardputer_report_integration.py`、`cardputer_monitor/validation`；云端 `cloud/tests`。报告相关 221 项已通过；接收链路失败路径仍需独立验证 |
