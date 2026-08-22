# v22 每周审批与绘图通知

## 固定时间

- 签名周边界：每周日 `15:00 UTC`，即北京时间周日 `23:00`。
- 候选生成：周边界前 16 小时，即北京时间周日 `07:00`。
- 默认审批截止：候选生成后 12 小时，通常为北京时间周日 `19:00`。
- 生效时间：通过全部硬门槛后仍等到北京时间周日 `23:00` 原子切换。

## 审批与通知绑定

候选封包完成后，周度管理器在同一次生成流程中写入一个哈希绑定的
`MODEL_APPROVAL_PENDING` 事件。该事件同时包含 release/model 哈希、审批截止时间、
可复制到 Hermes 私聊的审批提示词，以及 `report_request=v22_png_windows`。

`dca-live-report` 是唯一 Telegram 发送器，最迟在 60 秒内读取该事件。频道先收到审批
文字，然后收到与同一事件 ID 绑定的 12 张手机竖版 PNG：Grid BTC、Grid ETH、DCA BTC、
DCA ETH 各三张，分别覆盖过去 360 个完整 UTC 自然日、2026 年 1—2 月和 2026 年 5—6 月。
消息与附件按事件 ID、频道 ID 和文件哈希幂等去重。

绘图在独立低优先级进程中执行，因此附件可能晚于审批文字，但不会丢失与候选的哈希绑定。
绘图或 Telegram 故障不暂停当前模型和交易；系统会发送证据缺失告警并保留持久化重试记录。
审批等待期间继续使用当前有效模型，只有未来周边界的原子指针切换才使新模型生效。

## 旧模型保留

新模型在周边界完成健康激活后，周度管理器执行一次内容寻址清理：

- 保留当前 release 和按签名有效期排序的最近 3 个旧 release，总计通常最多 4 个。
- 同步删除只引用被清理 release 的非活动 runtime generation，当前 generation、前一代
  安全指针及其依赖始终受保护。
- 清理不在审批、候选生成、预热或不健康激活阶段执行。
- `V22_WEEKLY_RETAIN_OLD_RELEASES=3` 控制旧 release 数量；数字不包含当前 release。
- 12 张证据 PNG、Telegram message ID、交付回执、审批记录和事件日志位于独立审计目录，
  不随旧模型清理。
- 清理采用目录原子改名后删除；删除失败会恢复目录、发送
  `MODEL_RETENTION_FAILED`，不回滚当前模型或中断交易，并在下一次健康更新后重试。

每次清理结果写入 `V22_WEEKLY_WORK_PATH/release_retention.json`，成功删除时发送
`MODEL_RETENTION_PRUNED`，其中列出保留 release、删除 release 和同步删除的 runtime
generation。
