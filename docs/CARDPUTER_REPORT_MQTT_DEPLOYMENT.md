# OCI Report → Cardputer MQTT 实现与部署交接

更新日期：2026-10-05，Asia/Shanghai。

**已上线：OCI `dca-live-report` 于 2026-10-05 10:39:39（Asia/Shanghai，02:39:39 UTC）启动 `20261005-mqtt-v2`，健康检查通过，真实公网 MQTT 连续两帧及在线状态验证通过；四条价格曲线均取得 73/73 个可信采样点。**仅更新报告容器；云端接收、SQLite 落库和设备 HTTPS 展示由接收任务另行验收，本次不声称端到端接入完成。

## 1 本次范围

在现有 `dca-live-report` 进程内增加一个后台 MQTT 发布任务，不新建 `cardputer-monitor` 容器。四个交易单元仍由原报告逻辑汇总，原 Telegram 通知、风险归档和审计继续运行。MQTT 模块不调用交易接口、不下单、不改变 Grid/DCA 策略参数或普通交易权限。

报告完成收益 SQLite、交易状态 JSON 及风险归档发布后，只向后台交付刷新信号。后台使用独立只读 SQLite 连接，读取已有报告、获取公开行情、编码快照和维护 MQTT 连接。主循环不等待 DNS、TLS、行情请求或 PUBACK。

首版约每 60 秒生成一个快照。后台只保留最新刷新请求和最新快照，Paho 队列、在途消息分别限制为 1；断线以 1–30 秒退避重试。重试同一帧保持 `sample_id`、内容和 `collected_at`，新报告到达可替换尚未发送的旧帧。网络故障只降级监控链路。

## 2 数据合同

| 固定 ID | 来源策略实例 | 报价币种 |
| --- | --- | --- |
| `grid:BTC-FDUSD` | `grid-live-fdusd-400` | FDUSD |
| `grid:ETH-FDUSD` | `grid-live-fdusd-400` | FDUSD |
| `dca:BTC-USDT` | `dca-live-btcusdt-200` | USDT |
| `dca:ETH-USDT` | `dca-live-ethusdt-200` | USDT |

报告目录为容器内 `/workspace/state/telegram`。`trading_status.json` 提供最终权限、运行状态、阻塞原因和恢复阶段；`telegram_outbox.sqlite/profit_snapshot` 提供机器人归属 MTM、权益及回撤。复用 `OperationsReportReader` 与 `SnapshotCollector`，不另行核算账户收益。

- 首页数据为各自 4h 收益，协议仍保留 24h、7d、累计收益及窗口完整性。FDUSD 与 USDT 分别显示，历史不足用 `null`；合法 0 保留。
- 每行附带最近 12h 收益和对应交易对价格曲线，最多各 73 点，10 分钟一格。收益缺失不插值、不补零；缺少左端可信 MTM 时曲线不可用。
- 正式 overlay 显式选择 `binance-public`，从固定交易对白名单读取已闭合 5m K 线，不使用交易所 API Key。行情失败可仅降级价格；异常历史不丢弃有效当前状态与收益。
- 读取 WAL 使用短只读事务，关闭事务后才执行网络请求。不能跨线程复用报告的写入连接或请求 session。
- 状态、收益、行情保留各自源时间，分别按 300 秒判断新鲜度。报告观察时间不等于底层行情或成交时间；底层健康仍由原报告检查。重新发布或收到 retained 不刷新旧源数据年龄。

### 停限区间的首版边界

本地风险归档任务已在 `risk_history.json` 提供权限历史。**本次 MQTT 出站有意不上传这些停限区间**：各行 `history.pauses=[]`，`status_coverage_start_at/status_coverage_end_at=null`。这使现有协议保持兼容，也不改变风险归档任务的文件、数据库或消费者。

接收端仍需使用连续、可信的最终交易状态观测，生成买入限制或全部停止区间及原因。首次连接前和断网缺口均为未知，不能从当前状态或当前模型概率倒推历史；不能把云端落库时间当成实际恢复时间。未来若需 OCI 风险历史回填，需要另行统一 `sell` 等权限范围、覆盖区间、来源验证和回填协议。

## 3 MQTT 与权限

| 项目 | 值 |
| --- | --- |
| 地址 | `sh.sunnypiggy.top:8883` |
| 协议 | 原生 MQTT 3.1.1 over TLS，验证证书链与域名 |
| 唯一发布 Client ID | `hummingbot-cardputer-report` |
| 快照 Topic | `cardputer/v1/trading/hummingbot-main/snapshot` |
| 在线状态 Topic | `cardputer/v1/trading/hummingbot-main/availability` |
| 发布参数 | 两个 Topic 均 QoS 1、retained |
| 消息限额 | UTF-8 编码后最多 65,536 字节，禁止 NaN/Infinity |
| 来源 | `schema_version=1`、`source_id=hummingbot-main`，固定四行 |
| availability | 纯 ASCII `online/offline`，不是 JSON；设置 retained `offline` 遗嘱，正常退出限时确认 offline |

沿用已经准备的专用 `cardputer-report` 发布账号，只允许写两个指定 Topic；它没有读取、通配控制或交易命令权限。云端订阅账号只读交易 Topic，并保留原 GPU 权限。Cardputer 继续使用 HTTPS 设备 Token，不取得 MQTT 发布密码。

发布账号 JSON 只包含 `username/password`，独立作为只读 Docker secret 挂载。密码不写入普通 env、源码、镜像、日志或部署交付记录；不重新生成或轮换已有账号。CA 是公开文件，可以只读挂载。

QoS 1 可能重复投递，接收端按 `sample_id` 与源时间做幂等和单调校验。PUBACK 仅证明 Broker 接收，不证明云端业务落库。availability 反映发布连接，offline 不能解释为策略停止。

## 4 实现与配置入口

| 文件 | 用途 |
| --- | --- |
| [mqtt_reporter.py](../cardputer_monitor/mqtt_reporter.py) | 后台采集、编码、TLS 发布、最新帧重试、健康与退出 |
| [requirements-mqtt.txt](../cardputer_monitor/requirements-mqtt.txt) | 本地 MQTT 开发依赖，固定 `paho-mqtt==2.1.0`，复用采集器 requirements |
| [dca_live_report.py](../live_guard/dca_live_report.py) | 风险归档之后触发；健康检查及主循环生命周期集成 |
| [Dockerfile.dca-live-report-mqtt](../Dockerfile.dca-live-report-mqtt) | 在已运行报告镜像上叠加本次白名单模块与 Paho 2.1.0 |
| [compose.cardputer-report-mqtt.yml](../ops/compose.cardputer-report-mqtt.yml) | 仅覆盖报告镜像、MQTT 配置和新增只读 secret/CA |
| [deploy_cardputer_report.py](../ops/deploy_cardputer_report.py) | 基线检查、私有部署目录、定向构建、发布及公网订阅验证 |
| [需求合同](CARDPUTER_MONITOR_MQTT_REQUIREMENTS.md) | 设备界面、接收与落库任务的完整需求 |

overlay 中生效的设置为：

| 配置 | 正式值 |
| --- | --- |
| `CARDPUTER_MQTT_ENABLED` | `true`，未加 overlay 的默认仍为关闭 |
| `CARDPUTER_MQTT_HOST/PORT` | `sh.sunnypiggy.top` / `8883` |
| `CARDPUTER_MQTT_CREDENTIALS_FILE` | `/run/secrets/cardputer_report_mqtt` |
| `CARDPUTER_MQTT_CA_FILE` | `/etc/cardputer/isrg-root-x1.pem` |
| `CARDPUTER_PRICE_PROVIDER` | `binance-public` |

主机路径由 `CARDPUTER_REPORT_CREDENTIALS_PATH`、`CARDPUTER_REPORT_CA_PATH` 传给 overlay，镜像由 `CARDPUTER_REPORT_IMAGE` 指定。只有路径和镜像名进入配置变量，密码仍从 secret 文件读取。

本地 MQTT 开发环境可以安装 `cardputer_monitor/requirements-mqtt.txt`。该文件用于明确依赖，不用于覆盖生产基础镜像的 requests 等包版本；生产发布镜像只从离线 wheel 安装新增的 Paho 2.1.0，继承原报告镜像已有依赖。

`--healthcheck` 不创建 MQTT 连接；Telegram 关闭时仍采集并发布。MQTT 启用时，`--once` 只有报告周期成功且快照获得 PUBACK 才返回 0；禁用时不等待 MQTT。**不要在现有发布容器旁启动使用同一 Client ID 的临时发布进程**；测试应使用隔离 Broker，线上验证使用只读订阅者。

报告增加线程后，参数图片子进程使用 spawn，避免 fork 继承网络线程锁。SIGTERM/SIGINT 唤醒主循环，finally 在有限预算内停止后台任务；初始化、刷新和关闭失败均记录固定错误类型，不附原始凭据或 payload。

## 5 定向发布与后续维护

本次 v2 部署基线记录位于本地 `H:\PycharmProjects\serverdocker\cloud\artifacts\oci-report-before-mqtt-v2.json`，目标报告前身为 v1；OCI 主仓源码哈希仍为原版。基线包含 SSH 现场核验出的目录、Compose 项目、目标和非目标容器及相关文件 SHA256。历史首次部署基线 `oci-report-before.json` 对应未启用 MQTT 的原报告，不能用于 v2 发布。

当前基线目录为 `/home/ubuntu/extra_drive/hummingbot`，项目 `hummingbot`，基础 Compose 为该目录下 `docker-compose.yml`。当前正式发布号 `20261005-mqtt-v2`，调用 helper 必须明确版本与对应基线，不依赖默认参数：

- 私有部署目录：`.cardputer-report-rollouts/20261005-mqtt-v2/`，权限 0700。
- 新镜像：`hummingbot/dca-live-report:cardputer-mqtt-20261005-mqtt-v2`。
- 保留旧镜像标签：`hummingbot/dca-live-report:pre-cardputer-20261005-mqtt-v2`，对应前一版 MQTT v1 的实际镜像 ID。
- 本地清单：`H:\PycharmProjects\serverdocker\cloud\artifacts\oci-report-mqtt-20261005-mqtt-v2\manifest.json`。

以下命令记录本次已执行的准备、构建、发布与验证，使用 Windows PowerShell。以后发布新版本需采集新的运行基线并使用新的 release；不能把已部署前的历史基线当成现在仍有效。日常重建目标容器使用下文保存的 `deploy.sh`。

```powershell
Set-Location "H:\PycharmProjects\hummingbot"
$reportPython = "H:\PycharmProjects\serverdocker\cloud\.venv\Scripts\python.exe"
$reportBaseline = "H:\PycharmProjects\serverdocker\cloud\artifacts\oci-report-before-mqtt-v2.json"
& $reportPython ops/deploy_cardputer_report.py prepare-build --release 20261005-mqtt-v2 --baseline $reportBaseline
& $reportPython ops/deploy_cardputer_report.py deploy --release 20261005-mqtt-v2 --baseline $reportBaseline
& $reportPython ops/deploy_cardputer_report.py verify-runtime --release 20261005-mqtt-v2 --baseline $reportBaseline
& $reportPython ops/deploy_cardputer_report.py verify --release 20261005-mqtt-v2 --baseline $reportBaseline
```

`prepare-build` 先比较 OCI 源文件哈希和容器身份，保存旧 Compose/报告源码，复制显式白名单源码并安装预先下载的纯 Python Paho wheel。构建使用原报告镜像的固定标签及 `--network=none`，随后在无网络、无生产挂载的临时容器内检查真实导入。无需重建 Guard/manager 的共享镜像。

`deploy` 私下校验合并配置，确认其他服务定义、原报告命令、健康检查、网络及原挂载/通知 secret 保留。它只执行目标报告服务的：

```sh
docker compose --project-directory /home/ubuntu/extra_drive/hummingbot \
  -p hummingbot -f /home/ubuntu/extra_drive/hummingbot/docker-compose.yml \
  -f /home/ubuntu/extra_drive/hummingbot/.cardputer-report-rollouts/20261005-mqtt-v2/compose.cardputer-report-mqtt.yml \
  up -d --no-deps --force-recreate --no-build dca-live-report
```

上面命令需要已设置新镜像与 secret/CA 路径变量。**后续维护优先执行已生成的 `deploy.sh`**，它保存完整版本参数，不需要再次手工指定：

```sh
sh /home/ubuntu/extra_drive/hummingbot/.cardputer-report-rollouts/20261005-mqtt-v2/deploy.sh
```

仅使用基础 Compose 的报告 `up` 可能移除 overlay 并换回原镜像；维护时应保留两个 `-f`、项目名和版本参数。不执行整栈 `up`、`down`、`make build`，不连带更新 Guard、策略、scheduler、manager 或任何数据库卷。

检查实际加载版本采用目标容器镜像 ID 与镜像内源文件 SHA256，不能只看磁盘脚本或一次 restart。部署前后还需核对非目标容器 ID、镜像、启动时间和 restart count。helper 在健康失败或非目标运行状态改变时回滚报告；原文件与现场状态发生并发变化则停止部署，先重新核验。

`verify-runtime` 是可重复执行的只读运行核验：校验目标镜像及健康、非目标容器身份、10 个加载源码 SHA256、原风险/通知/模型概率模块、原挂载来源及读写属性。它进一步检查发布凭据和 CA 的实际容器挂载均只读、报告没有新增端口、原网络不变，并将脱敏结果保存在该版本 `runtime-verification.json`。它不读取或打印凭据内容，也不启动第二个发布进程。

## 6 回滚

在已核验的 OCI 主机执行：

```sh
sh /home/ubuntu/extra_drive/hummingbot/.cardputer-report-rollouts/20261005-mqtt-v2/rollback.sh
```

v2 的 `rollback.sh` 恢复先前已核验的 v1 完整 MQTT overlay 配置，包含发布开关、只读凭据和 CA；镜像固定为 `pre-cardputer-20261005-mqtt-v2`（SHA `909ea2…`），不依赖可能被重新打标的 v1 普通标签。它仅 recreate `dca-live-report`，同样使用 `--no-deps --no-build`；不删除报告 SQLite、风险历史、消息或行情数据。回滚后核对目标镜像及健康、真实 MQTT 消息、Telegram 持久队列和非目标容器。保留部署目录、旧镜像标签和基线清单供恢复与诊断。

v2 基线记录 `rollback_compose_command`，来自 v1 manifest 中的完整 Compose 命令；v2 manifest 保留回滚命令及 `rollback_config_verified_at`。本次已修正回滚脚本并在真实 OCI Compose 合并配置中核验：前镜像 ID、MQTT 开关/地址/端口/行情提供方、网络/命令/健康检查、只读 secret/CA，以及其他 13 个服务均符合原配置。该核验没有执行回滚、没有重启当前 v2 容器。`repair-rollback` 是已完成的运维修正，交接者不需要重复执行。

若明确需要回到最初未启用 MQTT 的报告，使用仍保留的 v1 回滚脚本：

```sh
sh /home/ubuntu/extra_drive/hummingbot/.cardputer-report-rollouts/20261005-mqtt-v1/rollback.sh
```

它对应 `pre-cardputer-20261005-mqtt-v1` 的原镜像 `2661fe…`，恢复最初没有启用 MQTT 的报告。v2 回滚到 v1 MQTT 与 v1 回滚到初始报告是不同选择，应核对目标镜像、配置与预期发布状态。

SQLite 备份使用一致性备份接口；正在写入的单个主数据库文件复制不能作为可靠备份。本次容器替换不需要删除或重建现有报告数据库。

## 7 验证证据与部署状态

### 已完成的本地验证

执行者本轮报告 221 项相关测试通过，覆盖采集/历史、MQTT 发布、报告集成、风险归档、报告收益和 Telegram；其中 18 项新增验证真实行情接口按 K 线开盘时间筛选时的曲线左端边界。测试使用合成数据及模拟网络，不访问生产 Broker。

新增集成测试真实执行报告 SQLite、状态 JSON 和风险归档写入，证明通知触发时四行及源时间可读。另覆盖 Telegram 禁用、MQTT 通知异常隔离、健康检查不初始化、单次发布成功/失败/无发布者、SIGTERM 和异常退出清理。发布模块覆盖断线重试、重复内容、限额、TLS/凭据失败、最新帧替换和停限历史归一化。

行情边界修复将拉取起点提前到曲线起点前 600 秒，覆盖接口按 open time 筛选与界面按 close time 采样的差异，避免非整 5 分钟起点漏掉左端可用闭合 K 线。完成态筛选、300 秒年龄限制和真实缺口的 null 表示保留。该修复的正式版本与实际行情点数由下面现场证据确认。

### 已完成的现场验证

| 项目 | 当前记录 |
| --- | --- |
| OCI 目标容器 | `dca-live-report`，ID `90fa0537ee03c326028b9838a462054dfb533ad0148020fa39294d8b4f8fffc5` |
| 启动与健康 | `2026-10-05T02:39:39.870133771Z`，`healthy`，restart count 为 0 |
| 新镜像 | `hummingbot/dca-live-report:cardputer-mqtt-20261005-mqtt-v2`，实际镜像 SHA256 见下文 |
| 镜像导入与加载源码 | 无网络临时容器真实导入通过；10 个复制代码文件 SHA256 与版本清单一致 |
| 原报告模块 | 原扁平 `risk_history.py`、`model_probability_history.py`、`telegram_notifications.py` SHA256 保留 |
| Telegram/报告 | 周期继续更新；本次采样 pending=0、retrying=0，profit_report_error 为空 |
| 非目标运行容器 | 其他 10 个容器 ID、镜像、启动时间、restart count 全部不变 |
| Compose 全 profile 配置 | 其他 13 个服务配置不变，原报告挂载/通知 secret/命令/健康检查保留 |
| 实际挂载/网络/端口 | 新增凭据与 CA 均只读，原挂载来源及读写属性不变，网络不变，未新增报告端口 |
| 公网 Broker | 只读账号实际收到 retained 快照和下一周期新快照，availability 为纯 ASCII `online` |
| 四行身份与新鲜度 | 固定四行及 FDUSD/USDT 正确；状态、收益均 FRESH；4h、24h、7d、累计收益有有效值 |
| 曲线 | 每行收益/行情各 73 点；最新帧 DCA ETH 收益有 71 个有效点，所有行情均有 73 个有效点，真实收益缺失保留 null |
| 云端业务落库与 Cardputer HTTPS 接口 | 接收任务另行验收 |
| 实机显示、语音及峰值内存 | 设备联调任务，不在本次范围 |

当前 v2 镜像实际 ID 为 `sha256:db680a0997b1a35698f835ede51cc49273b929be9ad5631208a7fe28123d84f9`。前一版 MQTT v1 镜像 `sha256:909ea2f9de290b636c2c470e4098f5e909b766ec9482994190487b2ac921b0bb` 保留为 `hummingbot/dca-live-report:pre-cardputer-20261005-mqtt-v2`。最初报告镜像 `sha256:2661fe8378a66ad34622ca02d3dee9167c87d2b931b962cb03cbe35bddc76617` 仍保留为 `hummingbot/dca-live-report:pre-cardputer-20261005-mqtt-v1`。

**OCI 主仓的原报告源码文件没有覆盖。**新增源码位于私有 `.cardputer-report-rollouts/20261005-mqtt-v2/context/`，并已复制进上述版本镜像；运行 SHA256 以镜像内文件和 manifest 比对，不能把 OCI 主仓磁盘上的旧文件当成新进程版本。

真实 MQTT 帧证据：

| 快照 ID | 原始 `collected_at` | 编码字节数 | 接收方式 |
| --- | --- | --- | --- |
| `5a9ac32e-4c4f-4e6b-bacc-7d7d92a576b4` | `1791167983.0657597` | 38,457 | 新只读订阅者收到 retained |
| `62b0fc46-804b-4bae-aaf3-bbfe4decb332` | `1791168045.4301193` | 38,432 | 下一周期实时消息，非 retained |

两帧采集时间递增，相隔约 62.364 秒，均小于 64 KiB。第二帧状态与收益源时间均为 `1791168043.498691`，行情最后闭合时间为 `1791167999.999`，验证器据此独立判断新鲜度。四条行情曲线均为 73/73 个可信采样点，验证了左端拉取边界修复；仅使用已闭合 K 线，不使用尚未收盘的价格补点。后续真实行情或收益缺口仍应保持 null。

采样时 Grid BTC-FDUSD、DCA BTC-USDT 为 RESTRICTED，两个 ETH 单元为 NORMAL；这是 **2026-10-05 10:40 左右的只读状态**，本次未改变、恢复或解锁任何交易权限。DCA ETH 曲线缺少两个可信 MTM 采样点，保留缺口不填零，不将整个窗口伪装为完整。

本地证据文件如下，不保存完整生产盈亏 payload 或密码：

- [版本与基线清单](../../serverdocker/cloud/artifacts/oci-report-mqtt-20261005-mqtt-v2/manifest.json)：版本、原/新镜像、白名单源码 SHA256、部署路径与目标身份。
- [运行版本与容器比对](../../serverdocker/cloud/artifacts/oci-report-mqtt-20261005-mqtt-v2/runtime-verification.json)：镜像内加载文件、保留模块、非目标容器及报告健康摘要。
- [v2 公网 MQTT 连续帧验证](../../serverdocker/cloud/artifacts/oci-report-mqtt-20261005-mqtt-v2/mqtt-verification.json)：两帧身份、源时间、新鲜度、收益窗口可用性、曲线点数及 availability。

### v1 历史记录

v1 于 2026-10-05 10:29:10 上线，先验证原报告内后台 MQTT 发布、连续两帧及权限隔离。实际行情只有 72/73 点，排查发现接口以 K 线 open time 筛选，原请求起点只提前 300 秒，非整 5 分钟边界可能漏掉左端所需的已闭合前一根 K 线。v2 通过扩大取数边界修复，未以进行中的价格或插值填补。

v1 的 [版本清单](../../serverdocker/cloud/artifacts/oci-report-mqtt-20261005-mqtt-v1/manifest.json)、[运行核验](../../serverdocker/cloud/artifacts/oci-report-mqtt-20261005-mqtt-v1/runtime-verification.json) 和 [MQTT 验证](../../serverdocker/cloud/artifacts/oci-report-mqtt-20261005-mqtt-v1/mqtt-verification.json) 均保留。历史验证文件与 v2 分开保存，不能将 v1 72 点记录冒充当前版本结果。

公网 `verify` 创建临时只读 MQTT subscriber，观察至少两个不同且时间递增的真实快照及 `online`，并使用现有云端验证器独立校验每行历史。脱敏验证结果写入 `H:\PycharmProjects\serverdocker\cloud\artifacts\oci-report-mqtt-verification.json`，记录身份、状态、采集时间、字节数和曲线点数，不保存完整生产消息或密码。主循环的 `cardputer_mqtt` 日志是固定字段健康摘要；不能据此代替真实订阅或云端落库验收。

发布端验证通过后，接收 Agent 仍应核验 `Broker → worker → SQLite → /api/cardputer/v1/trading` 及历史分页。只有设备 API 实际返回相同快照及正确源时间，才能确认端到端接入完成。约 180 秒未取得有效新快照为通信中断；收到旧 retained 不能使旧数据变为实时。

本次未为验收强制中断生产 Broker 或报告网络；断网、重试和异常清理由隔离测试验证。生产 Broker 重启、云端 subscriber 持久会话恢复、业务落库幂等与设备显示仍需由对应上线任务验证。
