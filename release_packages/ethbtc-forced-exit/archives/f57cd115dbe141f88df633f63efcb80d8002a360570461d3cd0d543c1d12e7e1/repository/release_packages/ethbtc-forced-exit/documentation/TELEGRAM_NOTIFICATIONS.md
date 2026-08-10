# Telegram 频道通知与移动端报告

## 边界与职责

`dca-live-report` 是唯一 Telegram 发送器，不新增 Compose service。Grid Guard、
DCA Guard、Grid 参数调度器和 v22 发布工具只写标准 JSONL 事件，不持有 Telegram
凭证，也不直接联网发送。

通知 Bot 与 Hermes 审批 Bot 必须使用不同 Token。通知 Bot 只允许向配置的频道
调用 `sendMessage`、`sendPhoto` 和 `sendDocument`；不得调用 `getUpdates`。频道消息
没有审批或恢复权限，所有人工恢复仍在 Hermes 私聊完成。

## 配置

`dca-live-report` 只加载仓库根目录的 `telegram-notify.env` 作为通知专用、非秘密
配置。该文件持久保存总开关、四小时报告开关和频道 ID，避免 `docker compose up`
时因调用 shell 未导出变量而回退到关闭状态。禁止给该服务加载 `.env.control`；Bot
Token 仍只能通过 Docker secret 注入。

- `TELEGRAM_NOTIFY_ENABLED`：总发送开关，默认 `false`。
- `TELEGRAM_NOTIFY_BOT_TOKEN_FILE`：Docker secret 文件，默认
  `/run/secrets/telegram_notify_bot_token`。
- `TELEGRAM_NOTIFY_CHANNEL_ID`：唯一允许的目标频道 ID。
- `TELEGRAM_PROFIT_REPORT_ENABLED`：四小时收益报告开关，默认 `true`。
- `TELEGRAM_ALERT_<MECHANISM>_ENABLED`：七类机制各自通知开关，默认 `true`。

七类机制为 `v22_weekly_buy_gate`、`fomc_gate`、`strategy_loss_breaker`、
`strategy_drawdown_breaker`、`portfolio_loss_breaker`、
`portfolio_drawdown_breaker` 和 `position_protection`。完整性、API、行情和数据库
故障使用 `infrastructure_integrity_breaker`，不能被七类普通通知开关绕过。

## 事件合同和生命周期

事件 schema 为 `ethbtc-telegram-event-v1`，包含事件 ID、UTC 时间、来源、策略、
机器人、交易对、机制、旧/新阶段、严重级别、原因、动作、触发值、阈值和
release/model/parameter 哈希。

通知覆盖 `TRIGGERED → EXITING → EXIT_COMPLETE → COOLDOWN → REENTRY →
RECOVERED`，以及 `LATCHED`、`EXIT_DELAY` 和 `ACTION_FAILED`。生产者使用稳定的
源事件 ID 或触发时间作为 `correlation_id`。SQLite outbox 以事件 ID、附件哈希和
频道 ID 幂等去重；重启不会重复发送已经成功的内容。

每条 Grid/DCA 风控事件在原始“原因”字段后增加自然语言解释。DCA 的 v22 事件
必须明确列出：v22 BUY 门当前为“放行（Risk-On）”还是“阻止（Risk-Off）”、
恢复阶段、最终 DCA 聚合门的 BUY/SELL 状态、controller 更新是否已落地，以及
“交易正常”或“交易受限”的结论。只有聚合 BUY/SELL 均放行、恢复阶段为
`ACTIVE` 且 controller 更新已确认时，才允许写“交易正常”；缺字段时必须写明
“不能据此判断”，不得猜测。其他机制的后续影响按
`TRIGGERED/EXITING/EXIT_COMPLETE/COOLDOWN/REENTRY/RECOVERED/LATCHED/EXIT_DELAY/ACTION_FAILED`
分别生成；`RECOVERED` 只表示当前机制解除，不能表述成全部风控门已经放行。

Telegram 失败不进入交易决策链。待发送项指数退避重试，并保存发送时间、消息
ID、重试次数和最终错误；发送器按单频道限速。

## 每四小时收益报告

北京时间每天 `00:00/04:00/08:00/12:00/16:00/20:00` 发送。发送槽持久化，
重启只处理当前最近一个时段，不补发多条历史时段。

每次为四个机器人分别生成 1440×2400 PNG：Grid BTC-FDUSD 和 ETH-FDUSD 每对
资金基准 200 FDUSD；DCA BTC-USDT 和 ETH-USDT 每机器人资金基准 190 USDT
（95 现金 + 95 基础币）。PNG 只展示单机器人策略归属 MTM，不合并 FDUSD 与
USDT，也不称为 Binance 账户收益。

报告包含 4 小时、24 小时、7 天、上线以来收益，权益、峰值、回撤、归属基础币、
费用、买卖成交数、活动订单/executor、v22/FOMC 门、恢复阶段、数据年龄和告警。
数据缺失时显示“无可信数据”并继续准时发送降级报告。

## 模型更新回测图片

v22 候选、真实重训验证和后续模型更新不向频道发送 `.joblib` 模型、JSON 审计文件
或参数 PDF。频道只接收 12 张 1440×2400 竖版 PNG：Grid BTC、Grid ETH、
DCA BTC、DCA ETH 各三张，分别覆盖：

- 截止时间前最后 360 个完整 UTC 自然日；
- `2026-01-01 00:00 UTC` 至 `2026-03-01 00:00 UTC`（1–2 月）；
- `2026-05-01 00:00 UTC` 至 `2026-07-01 00:00 UTC`（5–6 月）。

每张图片只展示一个机器人，包含价格、连续权益、回撤、v22 概率和逐周阈值；
黄色区间表示 v22 Risk-Off，进入和恢复使用不同标记，签名模型未覆盖区间使用
红色边框，无可信成交回放证据的区间使用灰色。Grid/DCA、BTC/ETH 权益不合并。

完整性检查必须把生产模型哈希、历史证据模型哈希、执行策略版本和图片 SHA-256
写入事件与 manifest。历史周身份无法验证或回放证据缺失时，不推测信号、不伪造
图片，改发严重告警。图片失败不改变 v22 当前周、哈希、观察期和审批门槛，也不
阻断本来已满足旧部署门槛的 Grid 参数更新。

耗时回放/渲染由报告服务内的持久化、低优先级、单工作进程执行，XGBoost、
OpenBLAS、OMP 和 MKL 线程上限为 2，不阻塞五分钟采集、四小时调度或 outbox。

v22 周模型候选通过硬校验后，频道还会发送 `MODEL_APPROVAL_PENDING`，包含
release/model 哈希、12小时默认审批截止时间、当前结论和可复制到 Hermes 私聊的提示词。
提示词要求明确选择批准或拒绝，并说明无人拒绝且全部硬门槛持续通过时默认批准。
候选训练、PNG 渲染、频道发送和审批等待不改变当前 release、当前授权或交易合同；
只有未来周边界的原子部署指针切换才使新模型生效。硬校验失败发送
`MODEL_UPDATE_BLOCKED`，且绝不因超时自动放行。

## Hermes 恢复提示

事件进入 `LATCHED`，或在 `REENTRY` 因自动重入开关关闭需要人工处理时，频道
发送一次可复制提示词。提示词只含机器人、机制、事件 ID、release/model 哈希、
阶段和阻塞原因，不含 Token、账户余额、CLI 密钥或直接恢复命令。

Hermes 私聊依次完成：读取事件；只读检查退出完成、活动订单、资金归属、合同与
过滤器新鲜度和其他风控门；生成一次性事件哈希绑定审批；通过 `clarify` 按钮取得
精确批准；执行对应 reset/恢复授权；再次只读复核。任一步不可用或任一其他门仍
关闭，都必须停止，不能从频道直接 reset。

## OCI 上线检查

1. 将独立通知 Bot 加入目标频道并仅授予发帖权限。
2. 将 Token 写入 `/home/ubuntu/secrets/telegram_notify_bot_token`，权限设为仅部署
   用户可读；`.env.control` 只保存频道 ID 和非秘密开关。
3. 重建现有 `dca-live-report`、Grid Guard、DCA Guard 和 Grid scheduler 镜像，
   不新增服务。
4. 先保持总开关关闭，检查事件 JSONL、outbox、12 张测试 PNG 和目标频道 ID。
5. 开启总开关，模拟 info、附件失败和重复事件，确认只向目标频道发送、重试有效
   且不重复。
6. 确认通知端无 `getUpdates`，Guard 环境无通知 Token，Hermes 凭证没有挂载到
   `dca-live-report`。
