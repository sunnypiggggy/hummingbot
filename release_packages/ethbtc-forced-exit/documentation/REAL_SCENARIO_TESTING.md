# Grid/DCA 真实场景测试

## 目标与隔离边界

测试验证从 v22、FOMC 和熔断信号到 Grid/DCA 聚合门、统一库存账本、Binance 紧急执行和
Telegram 通知的运行时链路。它不是主网演练，不使用 Binance Testnet，也不允许访问
生产密钥、生产 Docker socket 或生产状态目录。

`docker-compose.risk-scenarios.yml` 是独立 Compose 项目，网络固定为 `internal: true`，
镜像使用 `:scenario` 标签。生产 Compose 不包含任何场景服务。只有同时设置
`GUARD_SCENARIO_MODE=true` 和非空 `GUARD_SCENARIO_ID`，Guard 才接受 loopback 或 Docker
内部模拟器端点；场景模式漏配模拟地址不能回落正式端点，生产模式下端点覆盖也会导致
启动失败。

## 真实执行内容

- Binance 模拟器校验 HMAC、时间戳和 API key，并维护余额、挂单、订单、成交、动态过滤器
  和第三资产手续费；撮合端独立执行 `MARKET_LOT_SIZE/LOT_SIZE` 与 `MIN_NOTIONAL`，非法
  数量会按交易所错误拒绝且不得生成成交。
- DCA 场景进程运行真实30秒的三次稳定确认；Grid场景进程运行生产 Guard 中的行情、过滤器
  和账户客户端。
- 清仓使用SQLite尝试日志和确定性 `clientOrderId`。成交后断连接时查询原订单；终态部分成交
  只对剩余量创建 `-rN` 子订单。
- 单次交易所请求阻塞超过租约TTL时，后台心跳仍持续续租；并发执行者不能在请求进行中
  抢占同一资产。
- 容器验收在确认期间向DCA场景容器发送 `SIGKILL`，重启后继续原SQLite/WAL状态。
- OCI容器验收还在共享卷上持有真实SQLite写锁，并主动断开DCA场景容器的internal网络；
  Guard必须非零退出、接回网络后从同一状态恢复。成交响应固定在经济成交后断开，恢复只能
  查询原 `clientOrderId`，不能产生第二笔成交。
- Telegram使用真实HTTP和持久化outbox，但只发送到隔离sink。容器验收让首条请求固定返回
  HTTP 429，再重启报告容器；退避结束后必须恰好送达7条、零pending、零重复。
- 交易所故障层覆盖过滤器临时变化、撤单延迟、成交后余额延迟、429/5xx、时间戳偏移，
  并分别核算基础币、报价币和BNB手续费；任一复核未完成都不能写入 `COMPLETED`。

脱敏8月10日场景固定保留以下事实：

- DCA归属 BTC `0.001499762327138578`、ETH `0.050528420907064937`；Grid归属为0。
- 只出售 `0.00155 BTC`，得到 `101.133005 USDT`，记录 `0.00012469 BNB`手续费。
- ETH `0.0022` 不足5 USDT，必须记为dust且零下单。
- 既有DCA `LATCHED`库存保持 `pending_manual_existing_dca_inventory`。

## 执行命令

```bash
pytest -q test/test_real_risk_scenarios.py test/test_risk_scenario_compose.py
python3 scripts/run_risk_scenario_acceptance.py
python3 scripts/run_risk_scenario_acceptance.py --soak-seconds 7200
```

运行器结束时删除场景容器、网络和卷，只保留JSON报告、按UTC排序的Markdown时间线及
脱敏状态副本。报告记录全部状态文件和时间线的SHA-256。每次运行前后对比生产容器集合，
发现变化立即判失败。

## 硬验收标准

- 未授权成交、超范围出售、重复经济成交、Risk-Off新增BUY均为0。
- 同一资产同一时刻只有一个退出租约；归属缺口、锁定余额、数据不新鲜或活动挂单均
  Fail-Closed。
- `COMPLETED`必须保存订单、余额、无活动挂单和请求数量四项复核结果。
- SQLite写锁、事务中 `SIGKILL`、WAL恢复和状态JSON临时文件半写均有进程级回归；半写
  `.tmp` 不能替换上一份完整合同。
- 七类机制逐项关闭时聚合BUY门必须关闭；除BUY-only v22外，退出/恢复阶段同时关闭SELL。
- Telegram最终送达且同一事件、附件、频道保持幂等。
- 快速随机层使用5个固定种子、每个500步；OCI验收使用20个种子。每步从BUY/SELL成交、
  外部入账、数据故障、活动订单、归属缺口、SQLite重启和七类门信号切换中固定种子抽取，
  同时核对余额归属与最终BUY逻辑AND。

## 最近一次容器验收

- 本地最终相关回归：运行时/风控160项通过，v22周模型/发布管理10项通过；仅有既有
  Pydantic弃用告警，无测试失败。
- 两小时长稳：`2026-08-09T17:23:12Z` 至 `19:24:49Z`，实际连续121分36秒，
  216次DCA容器重启；`PASS`，报告SHA-256
  `1841c6b2880807bd72b266fd14e49b9b031435f4ccff6ed825489793bc8c1d92`。
- 最终源码容器演练：`PASS`，8项硬检查通过；报告SHA-256
  `4bea78bb3c135af18e34e62b6520fe2b2dcb7aeeb7e21d47e6431280ddcb3688`，时间线SHA-256
  `696df0c4dbaed8f86d8ec53451672740e9819d3d772d1d8ca273f51d22bda1b1`。
- 两次均只有一笔 `0.00155 BTC` 成交，到账 `101.1330050 USDT`；ETH零订单，
  DCA既有LATCHED库存未出售，生产容器集合不变。最终演练还确认Telegram首发429后
  重启报告进程恰好送达7条、零pending、零重复。
- OCI随机验收：20个种子 × 500步，70,000项断言通过；内容SHA-256
  `0f33a49e47a7e4d81d75cecad37d8699872c643986d992d25fb1efc0a507feea`。

验收期间发现并修复两个运行器问题：`docker compose start` 会连带启动已退出的依赖容器，
现改为按容器ID执行原生 `docker start`；Telegram outbox 实际位于 `dca/telegram/`，现使用
只读SQLite URI验收，路径错误时不会创建空数据库掩盖问题。发现问题的那轮结果未计为通过，
上述最终哈希来自修正后的完整重跑。
