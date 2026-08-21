# 交易系统维护管理 Bot V3

## 职责边界

`sunnypiggy-trade-bot` 是独立的 Telegram 私聊管理容器。它负责展示、
向导、预检、二次确认和执行结果跟踪，不承担行情计算、策略决策、库存归属
或最终风险判断。

数据和动作链路如下：

```text
Telegram 主账号
  -> sunnypiggy-trade-bot
       -> Hummingbot API（Grid/DCA 状态、停止、启动）
       -> Binance Stocks Runtime（白名单、限额、预检、Executor）
       -> Guard JSON 合同（只读风控与库存状态）
       -> 模型审批公开目录（只读请求和证据）
       -> 模型审批决定目录（只写哈希绑定决定）

dca-live-report -> Telegram 通知频道（独立 Bot Token）
```

管理 Bot 没有 Docker socket、Binance API Key、紧急清仓凭证和通知 Bot
Token。交易所是否合法、是否有余额、行情是否新鲜以及是否触发风险限制，
均由对应权威容器再次检查。

## Telegram 交互

界面优先使用 InlineKeyboard，并编辑原消息。只有股票代码、金额、价格、
仓位参数和拒绝原因需要文本回复。会话15分钟过期；callback只包含随机短
会话ID，完整参数保存在独立SQLite。

所有可变更操作遵循：

```text
选择 -> 输入 -> 权威预检 -> 中文影响预览 -> 二次确认 -> 执行结果
```

SQLite同时保存 Telegram update offset、已处理update、经济动作幂等键和
审计索引。重复回调或容器重启不会再次创建相同订单或审批决定。

## Stock 接口合同

Stock Runtime提供以下受控接口：

- `GET/PUT/DELETE /stocks/whitelist[...]`
- `GET/PUT /stocks/limits`
- `POST /stocks/executors/preview`
- `GET/POST /stocks/executors`
- `GET /stocks/executors/{executor_id}`
- `POST /stocks/executors/{executor_id}/cancel`
- `POST /stocks/executors/{executor_id}/reduce`
- `POST /stocks/executors/{executor_id}/close`

白名单只约束新增BUY。停用或移除股票不会自动卖出已有仓位，SELL、减仓和
平仓仍由归属账本校验后执行。

活动限额保存于Stock Postgres：单笔金额、单股票持仓和Stock总持仓。
环境变量是不可突破的硬上限；运行时必须满足：

```text
0 < 单笔限额 <= 单股票限额 <= 总持仓限额
```

## 模型审批合同

Scheduler将候选请求和脱敏状态原子发布到 `approval_public`。管理 Bot只挂载
这个目录并扫描 `approval-request-*.json`，展示候选、硬门槛、模型周、
有效期和证据附件哈希。批准或拒绝写入独立决定目录，决定至少绑定：

- 模型类型、candidate ID、release和model哈希；
- approval request哈希；
- Telegram user/chat/update/callback ID；
- 操作人、决定、原因和时间。

Scheduler只消费与当前待审批release完全匹配的决定，并重新执行全部硬校验。
12小时默认批准保持不变；证据缺失或硬门槛失败时不会自动批准。候选审批、
预热和失败不会覆盖当前健康runtime generation。

## 部署与运维

1. 将管理Token保存到OCI文件
   `/home/ubuntu/secrets/sunnypiggy_trade_bot_token`。容器固定以
   `UID/GID 10001`运行，因此主机文件应由运维用户持有、组设为`10001`，
   权限设为`0640`；这样只有主机运维用户和管理Bot进程可读。
2. 创建可写目录 `telegram-management-data` 和
   `results/ethbtc_forced_exit_weekly/decisions`；后者必须仅允许scheduler与
   管理Bot写入。
3. 先保持 `TRADING_MANAGEMENT_MUTATIONS_ENABLED=false` 启动
   `sunnypiggy-trade-bot`，验证私聊鉴权、状态合同和Stock预检。
4. 首次启动会删除旧webhook并丢弃Condor遗留update；之后从SQLite offset
   恢复，不再次清空消息。
5. 观察通过后再开启变更总开关。Stock真实订单仍额外要求Stock Runtime为
   `LIVE`且已独立授权。

Condor迁移记录见 `CONDOR_MIGRATION_ARCHIVE.md`。
