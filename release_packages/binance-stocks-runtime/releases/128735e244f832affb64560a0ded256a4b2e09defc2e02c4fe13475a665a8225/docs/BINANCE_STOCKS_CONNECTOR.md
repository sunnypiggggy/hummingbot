# Binance Stocks Trading Hummingbot Connector

## 1. 交付范围

`binance_stocks` 是独立于 `binance` 现货 connector 的直接美股/ETF connector。首版定位为行情、Paper Trade、DCA 和低频订单生命周期验证。

支持：

- 内部交易对 `AAPL-USDC`，API ticker 映射为 `AAPL`。
- 一档 Bid/Ask 行情；每条 quote 都是完整快照，不伪装 L2 深度。
- `LIMIT` 和 `MARKET`。
- HMAC-SHA256、确定性 `clientOrderId`、订单查询、撤单和累计成交/费用对账。
- `calendar`、`tradingStatus`、`tradability` 和 `orderReport` WebSocket。
- Funding Wallet (`CARD`)；BUY 请求显式携带，SELL 按 Binance 规则默认使用 CARD。

不支持：

- bStocks、`tokenize`、mint/redeem。
- Grid、做市、`LIMIT_MAKER`、GTC 和 24H session。
- L2 深度、逐笔公共成交、VWAP。
- 使用 Binance Spot 钱包余额或本地成交历史推算股票持仓。

## 2. 文件结构

```text
hummingbot/connector/exchange/binance_stocks/
├── binance_stocks_exchange.py                 ExchangePyBase 实现和交易门
├── binance_stocks_auth.py                     SAPI HMAC
├── binance_stocks_web_utils.py                REST/WS factory 和时间同步
├── binance_stocks_constants.py                路由、限速和订单状态
├── binance_stocks_utils.py                    配置、发现和响应解包
├── binance_stocks_order_book.py               一档快照
├── binance_stocks_api_order_book_data_source.py
├── binance_stocks_api_user_stream_data_source.py
└── binance_stocks_position_provider.py        权威持仓 provider 合同
```

目录会被 Hummingbot 的现有 connector 扫描器自动发现，因此无需修改 Binance Spot 或维护静态注册表。

## 3. 配置

```yaml
connector: binance_stocks
binance_stocks_api_key: <secret>
binance_stocks_api_secret: <secret>
quote_asset: USDC
wallet_type: CARD
trading_session: EXTENDED
time_in_force: DAY
disclaimer_confirmed: false
```

用户必须先在 Binance 完成美股法律声明确认，再手动把 `disclaimer_confirmed` 设为 `true`。connector 不会调用 `POST /sapi/v1/equity/account/disclaimer`。

首版会本地拒绝非 USDC、MAIN、RTH/24H 限价配置、GTC 和 `LIMIT_MAKER`。

## 4. 行情与市场状态

启动时通过 `exchangeInfo` 获取 ticker、精度、最小订单、碎股能力和可交易方向。Stocks `MARKET_DATA` REST 不需要签名，但仍携带 `X-MBX-APIKEY`；REST quote 用于初始快照，combined WebSocket 消费：

- `{SYMBOL}@quote`
- `calendar`
- `{SYMBOL}@tradingStatus`
- `{SYMBOL}@tradability`

一条新 quote 会整体替换旧 Bid/Ask。交易时段内 quote 超过 10 秒未更新时禁止下单。`MARKET_CLOSED` 是正常业务状态，不标记为基础设施故障，但任何订单仍会被本地阻止。

ticker 订阅发生增删时会断开并重建 URL 型 combined stream；不存在动态 SUBSCRIBE RPC。

## 5. 下单语义

LIMIT 请求固定为：

```text
orderType=LIMIT
tradingSession=EXTENDED
timeInForce=DAY
quoteAsset=USDC
walletType=CARD  # BUY only
```

MARKET 请求不带 `price`、`tradingSession` 或 `timeInForce`，且只在 `MARKET_OPEN` 放行。

`POST order/place` 和 `POST order/cancel` 的 `S/F` 只是请求回执，不会直接把订单标为 OPEN/CANCELED。最终状态来自 `orderReport` 和 `order/detail`。下单超时或断线后，connector 先按确定性 `clientOrderId` 查询；不会直接重复 POST。

取消订单属于降低风险动作，不依赖持仓 provider，但仍由 `orderReport`/REST 对账确认最终状态。

## 6. 费用

预估费用口径：

- 名义金额不超过 350 USDC：固定 0.35 USDC。
- 名义金额超过 350 USDC：0.1%。

成交后以 `order/detail` 顶层累计 `fee` 为准。connector 以已持久化的累计成交量、成交额和 USDC fee 做差分，避免部分成交、乱序回报或重启后重复记账。BUY fee 计入成本，SELL fee 从回款扣除；不会产生 BNB fee。

## 7. 权威持仓合同与 Fail-Closed

`EquityPositionProvider` 必须返回：

```python
EquityAccountSnapshot(
    positions={"AAPL": EquityPosition(total=..., available=...)},
    quote_total=...,
    quote_available=...,
    source="authoritative-source-name",
    timestamp=...,  # Unix seconds
)
```

默认 `UnavailableEquityPositionProvider` 永远不提供快照。不能用订单历史、手工 CSV、加密钱包余额或策略本地库存替代这个合同。

专用 `binance-stocks-runtime` 是显式例外：它可注入
`ManagedLedgerEquityPositionProvider`，但该 provider 只授权出售本容器前缀订单的确认成交，
并始终标记 `managed_ledger_non_authoritative`、`external_positions_unknown=true`。它不声称是
账户权威持仓，也不导入部署前或人工交易产生的美股库存。运行细节见
[`BINANCE_STOCKS_RUNTIME.md`](BINANCE_STOCKS_RUNTIME.md)。

实盘 place 请求发出前必须同时满足：

1. API/user stream 验证成功；
2. `disclaimer_confirmed=true`；
3. 市场阶段、ticker 状态和 BUY/SELL 方向允许；
4. quote 在 10 秒新鲜度内；
5. 权威持仓快照在 30 秒新鲜度内；
6. SELL 的可用股票数量或 BUY 的可用 USDC 足够。

任何一项失败都会在 HTTP place 之前本地阻止。由于 Binance 当前没有官方股票持仓查询接口，默认 provider 使实盘 BUY/SELL 持续 Fail-Closed；只读行情与 `binance_stocks_paper_trade` 不受影响。

未来出现官方持仓接口时，应新增 provider 实现，不修改订单簿、策略或订单生命周期。

## 8. 运行状态

`status_dict` 在 Hummingbot 基础状态之外增加：

- `account_eligible`
- `disclaimer_confirmed`
- `market_state_initialized`
- `quotes_fresh_or_market_closed`
- `position_reconciliation_ready`

`position_reconciliation_ready=false` 是预期的安全状态，不表示可以通过关闭检查绕过。

## 9. 测试

```powershell
$env:PYTHONUTF8='1'
C:\Users\sunny\anaconda3\envs\hummingbot\python.exe -m pytest -q `
  -p no:cacheprovider test/hummingbot/connector/exchange/binance_stocks
```

测试包含：

- HMAC、配置发现和一档快照替换。
- 无权威持仓时网络调用前 Fail-Closed。
- LIMIT/MARKET 参数、交易时段、碎股、USDC fee 和累计成交差分。
- 本地有状态 HTTP/WebSocket 模拟器；真实 RestAssistant 对模拟器完成签名、place、detail 和 cancel，全程不使用主网密钥。

可选公共行情 smoke test 后续必须显式启用，不进入默认 CI；严禁在测试中调用免责声明接口或提交真实订单。

## 10. 官方依据与限制

- [Stocks Trading Quick Start](https://developers.binance.com/en/docs/products/stocks/quick-start)
- [Stocks Trading General Info](https://developers.binance.com/en/docs/products/stocks/general-info)
- [Stocks Trading Common Definition](https://developers.binance.com/en/docs/products/stocks/common-definition)
- [Stocks Trading Change Log](https://developers.binance.com/en/docs/products/stocks/change-log)

实现依据 2026-08-12 的字段变更：`order/detail` 可按 `clientOrderId` 查询；逐笔 trade 不再提供 fee，已成交订单的顶层累计 fee 才是记账依据。
