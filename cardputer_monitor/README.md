# Cardputer 只读交易遥测

2026-10-05 用户选定由现有 `dca-live-report` 容器直接发布 MQTT，生产不新增独立
`cardputer-monitor` 容器。实现与部署需求见
[Cardputer 交易监控 MQTT 需求与部署交接](../docs/CARDPUTER_MONITOR_MQTT_REQUIREMENTS.md)。
2026-10-05 已将异步 MQTT 发布集成并上线到 OCI 的现有报告容器；四个交易单元的真实
快照与下一份更新已由只读订阅端收到。部署、版本、回退及接收端剩余工作见
[Report MQTT 部署交接](../docs/CARDPUTER_REPORT_MQTT_DEPLOYMENT.md)。
以下说明保留现有 HTTPS CLI 和可复用数据模块用法；生产由报告容器独占发布，勿同时运行 HTTPS 上传者。

默认只采集并输出脱敏 JSON，必须明确传入 `--publish` 才会通过 HTTPS 上传。
固定展示 Grid BTC-FDUSD、Grid ETH-FDUSD、DCA BTC-USDT、DCA ETH-USDT 四个交易单元；
Grid 两行来自同一进程。采集器不会读取交易凭证或 Telegram Token，不调用交易 API，
不修改策略、风控合同、报告或数据库。可选公开报价适配器只读取市场 Kline，
没有 API key、账户查询或交易操作。

## 数据与运行

复用 `management_bot.clients.OperationsReportReader`，读取报告目录中的
`trading_status.json` 和 `telegram_outbox.sqlite`。状态与收益分别检查来源时间：
超过 300 秒标记 `STALE` 并隐藏当前值，缺失、非法或未来超过 30 秒标记
`UNAVAILABLE`。缺少的一行仍保留固定身份，金额为 `null`，不会补零。
源状态 `UNKNOWN` 展示不可用，`ALERT_ONLY` 不误当停止交易。

收益是策略归属 MTM，包括持仓估值。4h、24h、7d 窗口由既有 Reader 的标量历史
计算；样本不足保持 `null/window_complete=false`。FDUSD 和 USDT 分别标注，不合计。
最终 BUY/SELL 权限不能解释为允许新增空单；保护性退出和普通卖单不同。
恢复只输出白名单字段，冷却结束不承诺恢复交易。

在仓库根目录运行：

```powershell
python -m pip install -r cardputer_monitor/requirements.txt
python -m cardputer_monitor --config cardputer_monitor/config.example.json --once
```

配置是 JSON，只有 `reports_dir`、`token_file`、`telemetry_url`、`interval_seconds`、
`request_timeout_seconds`、`price_provider` 和 `price_timeout_seconds`。
本地运行需将 `reports_dir` 改为报告目录的绝对路径。
Token 只能存在独立文件中，不能写入配置、命令参数或日志。默认每 60 秒采集，
每次上传最多两次请求，连接或 5xx 错误重试一次；下一周期重新采集最新数据，
没有历史上传积压。403、验证错误、重定向不重试，TLS 证书校验始终启用。

有独立、仅授权 `hummingbot-main` 写入的遥测 Token 后，显式发布命令为：

```powershell
python -m cardputer_monitor --config cardputer_monitor/config.example.json --once --publish
```

目标是 `POST https://sh.sunnypiggy.top/api/cardputer/v1/telemetry/trading`。
云端必须按 `sample_id` 幂等、拒绝旧采集时间覆盖新快照，并为 Token 绑定来源。
设备仅有读取交易状态的权限；收到的新心跳不能延长旧源数据的有效期。
云端/设备按源时间持续重算 300 秒年龄，180 秒未收到有效遥测则标记通信中断。

## 12 小时曲线与暂停记录

同一遥测快照的每行附带可选 `history`。时间轴为 `collected_at` 的整数秒之前
12 小时，每 600 秒一个点，共 73 点；`profit_points` 的值是各时点策略归属
累计 MTM 减去左端基准。读取现有 `profit_snapshot.mtm_quote`，每个时点只选
不晚于该时点、相距不超过 300 秒的观测，且四行所有曲线查询在同一只读事务中。
这不是成交收益求和。缺少可信左端基准时整条收益曲线为 `null`；中间缺口也保留
`null`，不插值、不延续上一笔。`window_complete` 专指所有 73 个收益点齐全。
当前快照与历史查询为独立事务，分别保留来源时间，不声称跨事务原子帧。

现有 v2 收益库只保留 MTM 标量审计，没有市场价格或最终交易权限的历史。
`management_trading.json` 只有当前策略参考价/中间价；不把成交价或模型回放当作
市场历史。设置 `price_provider` 为 `binance-public` 后，只有显式 `--publish`
运行才启用公开报价读取；默认 `none` 和所有 dry-run 均不打开报价网络连接。
适配器固定访问 `https://data-api.binance.vision/api/v3/klines`，四个固定交易对
分别请求 5m UTC Kline，只使用已闭合 Kline 的 close 及其真实 close time，
再按同一 10m 时间轴抽样。不会读取交易凭证，不允许自定义目标 URL；每对每周期
一次有时限请求，禁止重定向，启用证书校验，最多 150 根/128 KiB。
`price_timeout_seconds` 在 1–10 秒之间，默认 5 秒。报价故障只影响价格曲线。
参考 [Binance 官方公开行情说明](https://developers.binance.com/en/docs/products/spot/faqs/market_data_only)
及 [Kline 接口及字段](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)。

`profit_observed_at` 和 `price_observed_at` 分别是收益记录/闭合 Kline 的实际来源
时间，超过 300 秒的来源不得因为新上传心跳恢复新鲜。总 `data_state` 表示至少一个
曲线源可用；云端及设备仍需分别用两个来源时间控制相应曲线的新鲜度。
风控历史只读消费Report的`risk_history.json`，不另建风险历史数据库。合同缺失或
陈旧时为`pauses=[]/status_coverage_start_at=null`，不从当前状态推算历史。
暂停区间和连续覆盖起止均来自Report已保存的实际观测；早于观测起点及采样
中断区间保持未知。阴影必须区分 `buy` 和 `all`，限制买入不等于停止全部交易。
云端将 `status_coverage_start_at` 定义为最近连续可信状态观测的起点，并额外输出
可选 `status_coverage_end_at`（最后可信源状态时间）；起点之前、终点之后均未知。
遇到超过 180 秒的采样间断或 `UNAVAILABLE`，只展示新连续可信后缀。
通信中断或当前状态源过期时覆盖起止均为 `null`。源采集器无需发送终点字段。
`pauses` 最多 8 项，原因最多 48 字符；完整遥测不得超过 64 KiB。

## 可选容器

不修改根 Compose。以下命令仅构建并运行新采集器的 dry-run：
Dockerfile 专用 `.dockerignore` 使用明确白名单，构建上下文只含采集器与
复用的状态读取模块，不传输仓库环境文件、私钥、运行数据或发布包。

```powershell
docker compose -f docker-compose.yml -f cardputer_monitor/compose.yml --profile cardputer-monitor build cardputer-monitor
docker compose -f docker-compose.yml -f cardputer_monitor/compose.yml --profile cardputer-monitor run --rm cardputer-monitor --once
```

正式运行按部署目录的实际报告位置调整只读目录挂载。整个报告目录必须挂载为
`/reports:ro`，不能单文件挂载；JSON 的原子替换要求重新打开文件，SQLite WAL
还需要可读的 `-wal/-shm` 文件。UID 10001 需能读取源目录和独立 Token 文件，
不要改成以 root 运行来绕过权限。

显式发布可用 `run --rm -v /absolute/token-file:/run/secrets/cardputer_telemetry_token:ro`
追加只读 Token 挂载，随后传 `--config /app/cardputer_monitor/config.example.json --publish`。
省略 `--once` 时每 60 秒运行。此命令不会更新其他服务，采集器不开放入站端口。
配置长期服务时同样必须明确添加 `--publish` 与专用 Token 的只读挂载。

SQLite 在只读 `BEGIN` 事务中完成所有收益查询并关闭，再发网络请求。不要复制
正在写入的单个数据库，不使用 `immutable=1`，不改 journal mode，也不允许
采集器创建或写入报告目录。辅助文件暂时不可读时单独降级收益。
状态 JSON 和收益库没有跨文件事务，分别保留源时间，不声称同一原子帧。

## 合成合同与验证

`fixtures/trading.json` 是可共享的合成合同，包含正常、受限、停止、不可用，
以及合法零值和缺少窗口。固定时间为 2026-10-04 00:00 UTC，测试新鲜度时需注入
该时间或整体平移时间。所有 robot 字段固定存在，未知事实为 `null`。
`fixtures/trading-history.json` 扩展相同合同，含 73 点收益和市场价格曲线、真实零值、
中间缺口及完整未知行。其中暂停阴影和覆盖起点全部是明确的合成演示数据，
不是对实盘风险历史的断言；生产采集器不发送这些合成区间。

`blockers` 最多 8 项，每项 120 字符；`gates` 最多 16 项，只导出已知门控字段，
`mechanism/label/state/health/reason` 长度最多 64/40/32/16/120。
源日志和任意原因字符串不会透传，只输出固定中文说明；未知字段丢弃。
`recovery` 白名单为 `mechanism/phase/cooldown_until/healthy_cycles/exit_completed_at/reentry_block_reason`。

```powershell
python -m pytest test/test_cardputer_monitor.py test/test_cardputer_history.py -q
```

测试采用临时报告和模拟 HTTPS，不使用生产数据、Token 或真实上传。
Docker Desktop Linux 验收已通过：UID 10001 能读整个报告目录的只读挂载，
WAL 并发写入期间收益保持事务一致，并能重新打开原子替换后的 JSON。
部署时仍需核对实际 Linux 源目录和辅助文件权限、源报告重启、网络断连与恢复。

可重跑独立容器验收：

```powershell
docker build -f cardputer_monitor/Dockerfile -t hummingbot/cardputer-monitor:acceptance .
python cardputer_monitor/validation/docker_acceptance.py
```

该脚本只创建 `cardputer-monitor-acceptance` 临时容器，禁用网络，挂载纯合成报告
为只读，使用宿主机合成 writer 并发更新 WAL 与原子 JSON。测试结束仅清理自己
创建的容器和临时目录，不删除 volume。已有同名容器时拒绝运行，不影响其他服务。

独立真实模块联测（需安装云端开发依赖 Flask，并明确指定云源码目录）：

```powershell
python cardputer_monitor/validation/gateway_acceptance.py --cloud-root H:/PycharmProjects/serverdocker
```

该验收实际运行本采集器、公开 Kline 适配器的模拟 HTTP 客户端及云端真实 Flask
路由/SQLite Store；报告、价格响应和两个权限 Token 全部为新建临时合成数据，
不加载生产配置或凭证，不连接外部服务器。已验证四行固定币种、4h/24h/7d
MTM 窗口、73 点收益基准差/价格曲线、缺失基准与中间缺口、源时间的 300/301 秒
边界、两个曲线源独立降级、实际收到状态的暂停区间及未知尾部、重复上传不刷新
通信时间和读写权限隔离；历史分页响应最大 5,951 字节。
本目录单元测试 65 项通过；Linux UID 10001 的只读 WAL/原子 JSON/12h 曲线
并发验收为 80 个一致样本、80 个 writer 版本。真实设备麦克风、TF 拔卡及上网
稳定性仍需硬件验收。上述结果对应独立采集器的模拟验收；当前生产已采用报告内
MQTT 模块，云端交易接收、落库与设备接口仍由接收 Agent 验收。
