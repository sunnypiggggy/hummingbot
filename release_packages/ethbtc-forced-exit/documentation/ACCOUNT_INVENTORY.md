# Grid/DCA 统一库存归属与清仓

Grid 与 DCA 共用同一个 Binance 现货账户，因此不能分别把账户余额当作自己的库存。
生产环境使用共享 SQLite 账本和 `account_inventory_status.json` 合同统一核对 BTC、ETH。

## 归属公式

- Grid：`capital_reservations + Grid 净成交 + 已确认紧急调整`。
- DCA：`managed_inventory + DCA 净成交 + 已确认紧急调整`。
- 无归属：`Binance 实际总余额 - Grid 归属 - DCA 归属`。
- 归属缺口：`Grid 归属 + DCA 归属 - Binance 实际总余额`；大于零时 Fail-Closed，禁止用其他库存补差。

原有 JSON 账本继续作为证据输入，但不再单独决定出售数量。共享账本记录账户指纹、
证据哈希、逐资产归属、余额快照、确认次数、资产租约、清仓任务和交易所订单。

## 并发与幂等

BTC、ETH 分别使用事务型资产租约。Grid、DCA 或无归属清仓必须先取得对应租约，
再刷新 Binance 余额，并在下单瞬间用共享账本复核总归属和当前机器人归属。归属证据
缺失、总归属超过实时余额或请求数量超过该机器人归属时均零下单 Fail-Closed；不能靠
先卖一个策略的库存来掩盖账户级归属缺口。每次市价单使用确定性 `clientOrderId`；请求超时或容器重启后
先查询该订单，确认不存在才允许重试，防止重复卖出形成反向仓位。

执行器在同步 Binance 请求和成交复核期间由独立心跳线程持续续租，而不是只在请求前后
续租。心跳丢失会停止残余补单并强制重新核对账户；长时间 API 阻塞不能让第二个 Guard
取得同一资产租约。

## DCA 完整性退出

DCA 完整性熔断必须出售 `managed_base_target + net_base - 已确认退出量`，不能因为
`net_base=0` 将启动库存标记为 `not_required`。只有归属库存低于动态最小成交门槛、
无活动订单或 executor、成交与余额复核完成且 `exit_completed_at` 已写入时，才能记录
`action_complete=true`。

本次迁移保留现有 DCA 归属库存并标记为 `pending_manual_existing_dca_inventory`；不会
补做历史 DCA 清仓，也不会解除 `LATCHED`。

## 无归属库存

无归属 BTC/ETH 必须在连续三个独立余额周期、跨度至少 30 秒且证据不变后，分别通过
`BTC-USDT`、`ETH-USDT` 市价卖成 USDT。首次迁移数量受批准快照上限约束；余额增加的
部分进入新的确认周期，不会并入首次订单。低于动态 `MARKET_LOT_SIZE` 或
`MIN_NOTIONAL/NOTIONAL` 的残余记录为 dust。

状态合同为 `account-inventory-status-v2`，Telegram 生命周期事件包括发现、开始清仓、
完成、失败、归属缺口和恢复健康。库存清仓不会启动机器人、解除锁存或覆盖其他风控门。

`v2` 将活动挂单纳入顶层 `healthy` 计算。清仓作业只有在订单/成交、余额、活动挂单和
请求数量四项复核全部通过后才能进入 `COMPLETED`。`liquidation_attempts` 保存主订单与
确定性残余子订单；响应超时或 Guard 重启后必须先查询原 `clientOrderId`。
