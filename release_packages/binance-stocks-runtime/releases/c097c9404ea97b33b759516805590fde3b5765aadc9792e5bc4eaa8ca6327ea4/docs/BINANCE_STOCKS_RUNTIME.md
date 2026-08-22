# Binance Stocks 专用 Executor Runtime

## 1. 安全定位

`binance-stocks-runtime` 是单一、独立的美股执行容器。它复用 Hummingbot API 的
`ExecutorService`，但不复用加密资产的 Binance connector、机器人状态或数据库。

- 仅允许 `binance_stocks`。
- 仅允许 `order_executor` 和 `position_executor`。
- REST 只绑定 OCI `127.0.0.1:8001`，通过独立 Basic Auth + SSH 隧道访问。
- PostgreSQL 复用现有实例，但使用独立数据库 `hummingbot_stocks`。
- 不挂载 Docker socket，不具备创建 Grid/DCA/其他机器人容器的能力。
- 原始 `/trading`、账户写入、持仓记录删除、Grid/DCA 等接口全部由中间件禁止。

系统不会调用 Binance 免责声明签署接口。实盘必须同时满足 `LIVE`、显式授权和
`disclaimer_confirmed=true`。Shadow 只返回验证结果；Paper 会执行完整的本地模拟成交，
但不会产生 Binance 经济订单。

## 2. 归属口径

`ManagedLedgerEquityPositionProvider` 不是 Binance 全账户持仓接口。它只证明：

1. 请求 ID 已写入本容器 PostgreSQL；
2. connector 生成的 `clientOrderId` 使用 `x-HBSTK` 前缀；
3. Binance 确认成交增量属于该请求；
4. SELL 数量来自相同管理批次或 OrderExecutor 的未分配管理库存。

状态始终显示：

```text
position_source=managed_ledger_non_authoritative
external_positions_unknown=true
```

GOOG、MU、NVDA 等部署前仓位属于外部库存，不导入、不认领、不出售。部署后若发现
非本前缀的新美股成交，系统锁存新增 BUY、撤销未成交入场风险，但已有仓位的止损、
止盈和时间退出继续工作。

## 3. 风险限制

| 限制 | 行为 |
|---|---|
| 单次订单/开仓 | 名义金额不超过 200 USDC |
| 单标的敞口 | 默认不超过 200 USDC，并且不得超过该 ticker 的白名单限额 |
| 管理敞口 | 管理持仓 MTM + 未成交 BUY + 费用预留不超过 2000 USDC |
| 日内净亏损 | Binance `tradingDate` 内达到 200 USDC 后只阻止新增 BUY |
| SELL | 只能使用本容器账本可用份额；不允许做空 |
| 费用 | 仅 USDC；不使用 BNB |

默认白名单为 `AAPL/TSLA/SPY/QQQ`。`exchangeInfo` 中的其他直接美股/ETF仍可查询，需先经
`/stocks/whitelist/{symbol}` 显式启用后才允许新增 BUY。删除或禁用白名单只阻止新增 BUY，
不会删除账本，也不会阻止已有管理仓位的保护性退出。运行时限额只能向下收紧，不能超过
环境变量定义的硬上限。日亏损门不会阻止撤单、SELL、止损、时间退出或追踪止盈。

## 4. Executor 请求

所有创建请求必须提供稳定 `id`。超时重试必须复用同一 ID；同一 ID 配置不同会被拒绝。

### OrderExecutor LIMIT

```json
{
  "account_name": "stocks_managed",
  "controller_id": "stocks-runtime",
  "executor_config": {
    "id": "order-aapl-20260821-01",
    "type": "order_executor",
    "connector_name": "binance_stocks",
    "trading_pair": "AAPL-USDC",
    "side": "BUY",
    "amount": "0.25",
    "price": "220.00",
    "execution_strategy": "LIMIT"
  }
}
```

LIMIT 固定为 `EXTENDED + DAY`。MARKET 仍用基础股数表达；connector 使用 10 秒内的
ask 换算为 Binance `notional`。MARKET SELL 使用 `quantity`，且 MARKET 仅在 RTH 放行。

### PositionExecutor

```json
{
  "account_name": "stocks_managed",
  "controller_id": "stocks-runtime",
  "executor_config": {
    "id": "position-aapl-20260821-01",
    "type": "position_executor",
    "connector_name": "binance_stocks",
    "trading_pair": "AAPL-USDC",
    "side": "BUY",
    "amount": "0.25",
    "entry_price": "220.00",
    "triple_barrier_config": {
      "stop_loss": "0.03",
      "take_profit": "0.05",
      "time_limit": 14400,
      "trailing_stop": {"activation_price": "0.03", "trailing_delta": "0.01"},
      "open_order_type": "LIMIT",
      "take_profit_order_type": "LIMIT",
      "stop_loss_order_type": "MARKET",
      "time_limit_order_type": "MARKET"
    }
  }
}
```

PositionExecutor 仅允许 BUY 开多，且 `stop_loss`、`time_limit` 必填。首次部分成交即开始
保护和时间计时。风险退出先确认撤销剩余入场与止盈单，再按最终剩余管理数量退出：

- RTH：MARKET。
- PRE/POST：best bid 下 1 tick 的 EXTENDED DAY LIMIT，每 5 秒查询、撤单、按剩余量追价。
- CLOSED/OVERNIGHT：`EXIT_PENDING`，下一 PRE 开始限价退出；进入 RTH 后转 MARKET。

## 5. REST 范围

标准 `/executors` 创建、搜索、详情、日志和性能查询保留；Executor 类型注册表只包含上述
两类。美股专用接口：

- `GET /stocks/health`
- `GET /stocks/markets`
- `GET /stocks/whitelist`
- `PUT /stocks/whitelist/{symbol}`
- `DELETE /stocks/whitelist/{symbol}`
- `GET /stocks/limits`
- `PUT /stocks/limits`
- `GET /stocks/quotes/{symbol}`
- `GET /stocks/market-status/{symbol}`
- `GET /stocks/account-summary`
- `GET /stocks/managed-positions`
- `GET /stocks/open-orders`
- `GET /stocks/order-history`
- `GET /stocks/trade-history`
- `POST /stocks/executors/preview`
- `GET /stocks/executors`
- `POST /stocks/executors`
- `POST /stocks/order-executors`
- `POST /stocks/position-executors`
- `GET /stocks/executors/{id}`
- `POST /stocks/executors/{id}/reduce`
- `POST /stocks/executors/{id}/cancel`
- `POST /stocks/executors/{id}/close`

全账户订单/成交查询会标记 `managed_by_this_runtime`。取消/关闭只能操作
`stocks_managed + binance_stocks` 的 Executor；标准 keep-position 绕过接口被禁用。

## 6. 配置与密钥

独立 Docker secret JSON：

```json
{"api_key":"...","api_secret":"..."}
```

API Key 必须禁止提现、绑定 OCI IP，并与 Grid/DCA/紧急清仓 Key 分离。Paper 默认使用仓库内
空 JSON 占位，不包含秘密。关键环境变量见 `.env.control.example`。

## 7. 发布顺序

### Paper

```bash
docker compose --profile stocks build binance-stocks-runtime
docker compose --profile stocks up -d binance-stocks-runtime
```

保持：

```text
BINANCE_STOCKS_RUNTIME_MODE=PAPER
BINANCE_STOCKS_LIVE_AUTHORIZED=false
BINANCE_STOCKS_DISCLAIMER_CONFIRMED=false
```

有状态模拟器运行完整 PRE → RTH → POST 和故障场景。Paper REST 创建请求会启动真实
Executor 生命周期，订单只进入本地 PostgreSQL 撮合器；Binance place/cancel 请求恒为0。

### Shadow 24 小时

配置独立只读/交易资格 Key 后设 `BINANCE_STOCKS_RUNTIME_MODE=SHADOW`，授权仍为 false。
Shadow 读取真实行情、Funding USDC、订单及成交，Executor 创建返回
`SHADOW_VALIDATED_NO_ORDER`。验收外部活动识别、市场时段、动态规则、200/2000/200 门和通知，
不得出现本前缀新经济订单。

### 实盘（另行人工授权）

只有 Shadow 报告通过、外部活动已审计、管理账本为零且 Binance 法律确认已由用户完成后，
才能在另一次变更中同时设置：

```text
BINANCE_STOCKS_RUNTIME_MODE=LIVE
BINANCE_STOCKS_LIVE_AUTHORIZED=true
BINANCE_STOCKS_DISCLAIMER_CONFIRMED=true
```

本发布不会设置这些值。

## 8. 测试

```powershell
C:\Users\sunny\anaconda3\envs\hummingbot\python.exe -m pytest -q -p no:cacheprovider `
  test/hummingbot/connector/exchange/binance_stocks test/stocks_runtime
docker compose --profile stocks config
```

必须验收：重复经济订单、外部库存 SELL、做空、超 200 单笔、超 2000 敞口、日亏损后
新增 BUY 均为 0；部分成交和退出订单的累计数量/费用重启前后一致。
