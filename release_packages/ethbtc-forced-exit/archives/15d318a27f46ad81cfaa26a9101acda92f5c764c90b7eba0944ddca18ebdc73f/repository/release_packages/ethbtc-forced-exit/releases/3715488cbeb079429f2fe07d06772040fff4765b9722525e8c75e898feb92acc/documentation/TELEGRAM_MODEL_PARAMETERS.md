# Telegram 模型与参数只读模块

管理 Bot 只读取 `dca-live-report` 生成的脱敏合同，不读取交易所密钥、connector
配置或原始模型文件。页面不能修改参数，也不能直接批准模型。

## 导航

```text
模型与参数
├─ Grid 参数（BTC / ETH）
├─ DCA 参数（BTC / ETH）
├─ 风控门参数（Grid / DCA × BTC / ETH）
├─ 当前模型
├─ 候选模型
└─ 历史模型（最近3个曾实际上线且已下线的版本）
```

`当前模型`显示系统健康、签名模型周及北京时间有效区间、BTC/ETH Risk-On 或
Risk-Off、普通 BUY 影响，以及四个机器人的最终交易状态。模型有效期来自签名周
`week_start/week_end`，不能使用短周期刷新合同的 `valid_until`。

当前模型只有在证据同时精确绑定 release 和 model 时，才显示 Grid BTC、Grid
ETH、DCA BTC、DCA ETH 四个 360 天图片按钮。缺失时只显示“当前模型精确360天
回测：缺失”，不借用历史图片、不跳转历史、不自动生成。

`候选模型`只显示尚未激活的当前候选，包括目标周、有效期、审批/预热/激活状态、
自动审批倒计时、硬门槛和四张证据准备情况。审批操作仍跳转到独立“模型审批”。

`历史模型`只来自可信的生产 generation 切换记录。发布目录修改时间、被拒绝候选、
未激活候选和普通参数快照都不能生成历史模型。最近保留3个；没有可信生命周期或
精确证据时明确显示“无可信记录”。

## 后台合同

- `management_parameter_catalog.json`：生效参数、风控门、当前/候选/历史模型的
  人类展示字段和后台身份。
- `model_evidence_catalog.json`：按
  `release + model + strategy + pair + window` 索引并校验附件。
- `management/model_lifecycle.json`：已观察到的健康生产 generation 生命周期。
- `management/history/<sha256>.json`：仅供后台审计的内容寻址参数快照。

Release、Generation、模型/特征/参数哈希继续保留在后台合同、完整性校验和审批
决定中，但 Telegram 参数页、模型页、审批详情和图片说明均不展示。旧证据缺少
release 绑定时标记为 `UNBOUND_LEGACY`，保留审计但不进入 Telegram 导航。
