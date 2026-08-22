# Telegram 模型与参数只读模块

管理Bot的“模型与参数”页面只读取报告服务生成的脱敏合同，不读取交易所密钥、
connector配置或原始模型文件。

## 合同

- `management_parameter_catalog.json`：Grid/DCA生效参数、配置/运行哈希、风控门
  阈值与实时权限、v22当前/候选/最近3个版本。
- `model_evidence_catalog.json`：经manifest和SHA256验证的PNG/PDF索引。
- `management/history/<sha256>.json`：内容寻址参数快照，最多保留最近3个。

Guard通过 `mechanism_parameters` 发布实际阈值和恢复规则；报告服务按字段白名单
汇总。配置和运行参数不一致时合同必须标记 `MISMATCH`，不得回退展示默认值。

## Telegram导航

```text
模型与参数
├─ Grid参数（BTC/ETH）
├─ DCA参数（BTC/ETH）
├─ 风控门参数（Grid/DCA × BTC/ETH）
├─ v22当前模型
├─ 候选与最近3个版本
└─ 模型证据（策略 → 资产 → 时间窗口）
```

页面完全只读。模型批准/拒绝仍在“模型审批”完成；普通Grid参数修改继续使用
原有参数发布流程。

证据模型哈希与当前模型一致时标记为精确证据；不一致时只能作为历史覆盖证据，
Telegram标题必须明确说明，不能暗示为当前release的精确回放。
