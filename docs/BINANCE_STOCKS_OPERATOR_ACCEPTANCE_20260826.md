# Binance Stocks 管理授权与 PAPER 全链路验收

## 结论

2026-08-26 验收通过。OCI 上 Stocks Runtime 保持 `PAPER`，LIVE 未授权，Binance 真实经济请求计数为 0。最终已生成新的 PAPER 运行周期，现金与权益均为 2000 USDC，无持仓、无挂单、无活动 Executor，账本对账差额为 0。

Telegram SQLite 按操作方要求不备份、不放入发布包。本次只让 Bot 自动退休 Paper Reset 后已不存在的异步计划订阅，避免每 25 秒重复产生 404 日志。

## 运行授权边界

| 能力 | 状态 |
|---|---|
| 管理员配置维护 | 已开放 |
| PAPER 创建、撤销、平仓、减仓 | 已开放 |
| Stocks LIVE 交易 | 未授权 |
| Runtime 模式 | PAPER |
| 真实下单/撤单请求 | 0 |

身份校验、管理员私聊、二次确认、幂等 ID、资金、行情、交易时段、库存归属和风险检查均保留。

## 实际执行的验收

1. 白名单：读取 AAPL/TSLA/SPY/QQQ；临时加入 MSFT、修改额度、取得真实 Binance Quote 后直接删除；删除后新 BUY 被阻止。
2. 限额：运行中修改为 499/999/1999/199，确认 PostgreSQL 即时生效，再恢复为 500/1000/2000/200；Bot 重启后数值未丢失。
3. 行情：AAPL、TSLA、SPY、QQQ 的 Quote、交易状态、方向和新鲜度均通过；没有使用全市场批量 ticker。
4. OrderExecutor：真实行情驱动的不可成交 LIMIT 挂单与撤单、可成交 LIMIT BUY、MARKET SELL 均完成。
5. PositionExecutor：MARKET 开仓、保护性 LIMIT、部分减仓、容器重启恢复、手动平仓均完成。
6. 收益：5 笔模拟成交，成交额 80.00290549982 USDC，费用 1.75 USDC，净收益 -1.75096866890 USDC；账户收益与逐仓收益差额为 0。
7. 重启：未成交 LIMIT 和 Position 状态在实际容器重启后恢复，未生成重复成交。
8. Reset：有风险敞口时返回 409；全部撤单和平仓后成功生成新 `paper_run_id=80ca559ac03649d994dc30dbf7a3332e`。
9. 隔离场景：真实 PostgreSQL 中运行 FIFO 部分成交、重复 Quote、不虚构流动性、重启恢复、MARKET 超时，以及异步队列持久化/取消；临时测试 Schema 和数据库随后删除。

## 验收发现并修复的问题

- Binance calendar 只在阶段转换时推送，旧交易日状态曾被跨日复用；现要求持久化交易日与当前纽约日期一致。
- Paper Reset 先删父表导致异步计划外键冲突；现先删除依赖记录。
- Position 部分减仓后父 Executor 仍按原始数量平仓，可能模拟超卖；现按归属 lot 同步剩余数量，并在成交时再次限制 SELL 数量。
- SELL 库存已被策略账本预留后，Hummingbot BudgetChecker 重复扣减并误报余额不足；现由账本预留作为唯一预算判断，Connector 注册和成交归属检查仍保留。
- Paper Reset 删除异步计划后，Telegram 仍轮询旧 ID；现收到 404 后一次性退休订阅，不重复告警。

## 证据与哈希

机器可读摘要：`results/binance_stocks_operator_acceptance_20260826/summary.json`。

OCI 原始分阶段 JSON 保存在：

`/home/ubuntu/extra_drive/hummingbot/results/binance_stocks_operator_acceptance_20260826/`

| 文件 | SHA-256 |
|---|---|
| cleanup-reset.json | `83436dd2a0987e36e0d50d6476b00854f43abc87d90b8ec2df5cab5d74f739ab` |
| configuration.json | `7be81254835678aaf6c7342446c85631ba1bba83b725a4fab2e6b484e1692d5b` |
| trade-prepare.json | `d57ca3205ec572ca05d74e9b39c050de2fc41a8d581a21d92a719b7c12d058be` |
| trade-finish.json | `a2b27a807436ea2c95a4d09f31cf709d6bb41a81503ced984807438c1a29e984` |

## 最终状态

- Stocks Runtime：running / healthy
- `runtime_mode=PAPER`
- `live_authorized=false`
- `economic_http_request_count=0`
- 当前权益：2000 USDC
- 持仓：0
- 活动订单：0
- 活动 Executor：0
- Grid/DCA 三个实盘机器人未被本次验收重启或修改

