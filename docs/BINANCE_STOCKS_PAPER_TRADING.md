# Binance Stocks Paper Trading

`binance-stocks-runtime` 的 `PAPER` 模式使用 Binance Stocks 官方一档 Bid/Ask 行情，
但订单、成交、资金和持仓全部保存在本地 PostgreSQL。Paper connector 没有 Binance
下单或撤单网络路径，`BINANCE_STOCKS_LIVE_AUTHORIZED` 必须保持 `false`。

## 账户与成交口径

- 每个 Paper run 初始为 2000 USDC，所有 OrderExecutor 和 PositionExecutor 共享资金。
- LIMIT 遵循 `EXTENDED + DAY`；MARKET 只在 `MARKET_OPEN` 生效。
- 新订单默认延迟1秒。一个Quote事件的一档数量只消费一次，订单按创建顺序FIFO成交。
- MARKET一档不足时等待后续新Quote，5秒后取消余量。无盘口数量时不伪造成交。
- BUY按ask、SELL按bid；LIMIT成交价格不会差于订单限价。
- 单订单累计成交额不超过350 USDC收0.35 USDC，超过350 USDC按0.1%；部分成交不会重复收固定费，且不使用BNB。

账户权益为：

```text
cash_balance + Σ(long_position × latest_bid)
```

净收益为当前权益减去该run的初始2000 USDC。BUY费用计入持仓成本，SELL费用从回款
扣除。主要查询接口为：

- `/stocks/paper/account`
- `/stocks/paper/positions`
- `/stocks/paper/orders`
- `/stocks/paper/trades`
- `/stocks/paper/performance?window=4h|24h|7d|all`
- `/stocks/paper/equity`

## 数据隔离与恢复

Paper 使用 `binance_stocks_paper` schema，不读取或认领真实美股持仓，也不会把Paper
持仓迁移为实盘归属。订单、成交、Executor checkpoint和权益快照跨容器重启保留。
重启时恢复未完成订单、部分成交、止损/止盈、追踪止盈和时间退出的原计时基准。
恢复失败会取消未成交入场、保留虚拟持仓并设置 `PAPER_RECOVERY_REQUIRED`，禁止新增风险。

安全重置只允许在没有活动Executor、活动订单、活动intent和持仓时执行：

```json
{"confirmation":"RESET PAPER ACCOUNT TO 2000 USDC"}
```

重置创建新的`paper_run_id`，旧run的订单、成交和权益历史继续保留。

## OCI检查

运行正常时 `/stocks/health` 应满足：

- `runtime_mode=PAPER`
- `live_authorized=false`
- `economic_requests_enabled=false`
- `economic_http_request_count=0`
- `connector_ready=true`
- `paper_recovery_required=false`

官方实时行情需要独立的 Binance Stocks 只读凭证。凭证缺失时服务保持启动，但健康状态为
`degraded`、`connector_ready=false`，不得使用静态价格生成虚拟成交。

