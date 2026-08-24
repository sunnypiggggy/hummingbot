# Binance Stocks Paper Trading

`binance-stocks-runtime` 的 `PAPER` 模式使用 Binance Stocks 官方一档 Bid/Ask 行情，
但订单、成交、资金和持仓全部保存在本地 PostgreSQL。Paper connector 没有 Binance
下单或撤单网络路径，`BINANCE_STOCKS_LIVE_AUTHORIZED` 必须保持 `false`。

PAPER 不等待实盘最终授权。创建请求会启动真实 Hummingbot `OrderExecutor` 或
`PositionExecutor`，并经历与实盘相同的规则校验、下单状态、部分成交、止盈、止损、
追踪止盈、时间退出、撤单、费用、持仓和重启恢复流程。唯一替换点是最后的交易所经济
动作：实盘调用 Binance place/cancel；PAPER 把该动作提交给本地持久化撮合器。

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
- `/stocks/paper/summary`

`/stocks/paper/summary` 是面向管理端的一次性一致快照。它在同一个 `paper_run_id`
内返回账户、4h/24h/7d/累计收益、逐股票成本和收益、活动订单/Executor及对账结果。
窗口未完整覆盖时返回 `window_complete=false`；管理端不得把该值描述为完整窗口。
逐股票净收益口径为：

```text
realized_pnl + market_value - remaining_cost - fees
```

逐股票合计必须与 `equity - initial_usdc` 对账。缺少持仓估值或对账失败时
`valuation_complete=false` 或 `reconciliation.ok=false`，管理端必须隐藏收益数值。
休市时允许使用该run持久化的最后可信Bid估值并标记
`MARKET_CLOSED_LAST_TRUSTED`，不得将缺少行情的持仓按0估值。

## 下单接口

OrderExecutor预检使用 `POST /stocks/order-executors/preview`，PositionExecutor预检使用
`POST /stocks/position-executors/preview`。预检与创建共享同一服务端配置构造和权威策略，
客户端不自行拼装Hummingbot枚举。

人类可读的 LIMIT OrderExecutor：

```json
POST /stocks/order-executors
{
  "id": "paper-aapl-limit-0001",
  "symbol": "AAPL",
  "side": "BUY",
  "amount": "0.5",
  "order_type": "LIMIT",
  "price": "220.00"
}
```

包含完整保护的多头 PositionExecutor：

```json
POST /stocks/position-executors
{
  "id": "paper-aapl-position-0001",
  "symbol": "AAPL",
  "amount": "0.5",
  "entry_order_type": "LIMIT",
  "entry_price": "220.00",
  "stop_loss": "0.02",
  "take_profit": "0.04",
  "time_limit": 14400,
  "trailing_activation": "0.02",
  "trailing_delta": "0.005"
}
```

百分比参数使用小数形式：`0.02 = 2%`。LIMIT 必须有价格；MARKET 不得带价格且仅在
`MARKET_OPEN` 执行。Position 只允许 BUY 开多，`stop_loss` 和 `time_limit` 必填。

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

官方 REST 行情需要独立的 Binance Stocks MARKET_DATA API Key；PAPER 不要求交易 Secret。
凭证缺失时服务保持启动，但健康状态为
`degraded`、`connector_ready=false`，不得使用静态价格生成虚拟成交。

Compose 的 secret 源路径在项目级环境变量解析阶段确定。OCI 操作必须显式使用
`docker compose --env-file .env.control ...`，不能只依赖 service 的 `env_file`；否则
Compose 会挂载默认空凭证文件。只读行情 secret 只保存 `api_key`，`api_secret` 为空，
并保持 `economic_requests_enabled=false`。

周末或交易日历暂未给出有效市场阶段、但最新行情仍可信时，Paper 收益合同返回
`MARKET_STATE_UNAVAILABLE`。管理 Bot 显示“行情已接入，等待交易时段状态”，不会把
这种业务时段状态误报成行情断线；存在持仓时仍使用最后可信行情估值。

## 真实进程 Smoke

`scripts/smoke_binance_stocks_paper_api.py` 只能对
`BINANCE_STOCKS_SCENARIO_MODE=true` 的隔离 runtime 执行。它通过官方协议形状的有状态
行情服务器驱动真实 API、ExecutorService、Connector 与 PostgreSQL，并覆盖全部 Stocks
业务接口。验收要求：

- LIMIT BUY/SELL 完成一轮正收益；
- Position 完成开仓及止盈，取消和手动关闭能进入终态；
- 费用、现金、库存、权益峰值守恒，最终可安全创建新 run；
- runtime 的 `economic_http_request_count=0`；
- 模拟 Binance 的 place/cancel/其他私有经济端点命中数全部为 0；
- 日志中没有 `ERROR/Traceback/CRITICAL`。
