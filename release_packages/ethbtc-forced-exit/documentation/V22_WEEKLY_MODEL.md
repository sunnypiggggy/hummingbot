# v22 周度模型与参数更新

## 三种周期

- 模型与 fold-local 阈值：每7天更新一次。
- 风险概率：每个完整1小时K线更新。
- 结构确认：每个完整4小时K线更新。
- live 合同：Guard 约每30秒刷新，有效期150秒。

周区间采用 `test_start <= signal_ts < test_end`。当前发布族锚点为周日15:00 UTC，即北京时间周日23:00。

## 每周变化的内容

BTC、ETH 独立重新训练并签名：模型树及权重、最佳树数量、校准分布、`entry_threshold`、周模型哈希、总模型哈希、行情哈希、fold编号和有效区间。

阈值来自当前 fold 校准集：BTC 使用概率98%分位数，ETH 使用98.5%分位数。阈值不是固定常量，也不能从前一周继承。

## 冻结的门控结构

| 参数 | BTC-FDUSD | ETH-FDUSD |
|---|---:|---:|
| 概率连续确认 | 1个1小时信号 | 2个1小时信号 |
| 候选武装时间 | 48小时 | 48小时 |
| 最短 Risk-Off | 48小时 | 48小时 |
| 普通恢复确认 | 3个4小时周期 | 3个4小时周期 |
| 强恢复确认 | 2个4小时周期 | 2个4小时周期 |
| 恢复后冷却 | 48小时 | 24小时 |

概率达到阈值只会武装 Risk-Off 候选。真正进入还要求新的4小时结构满足：ROC<0、SQZMOM<0，并由 DI、EMA20斜率、低于EMA20比例确认持续弱势。

普通恢复要求 ROC、SQZMOM 相比前一4小时结构改善，并至少两项结构转好；强恢复要求 ROC或SQZMOM非负、DI>0、EMA斜率非负且低于EMA比例<0.50。

## 状态连续性

周模型切换不重置 `active/since/above_entry_count/armed_until/cooldown_until/recovery_count`、概率历史或4小时结构历史，也不重置资金、库存、权益峰值和累计盈亏。

缺周、重复覆盖、周不连续、BTC/ETH覆盖不一致均 Fail-Closed。禁止回退上一周、v21、ROC或SQZMOM。

## 训练防泄漏

- 新 cutoff 必须严格等于已签名 manifest 的上一周结束。
- 两对5分钟行情必须无缺口、无非法OHLCV，并覆盖 cutoff。
- 标签为 `long_event_72h`，成熟延迟96小时。
- Purge为120小时。
- 校准使用成熟记录最后14天。
- Early stopping使用最终拟合前的14天开发集。
- 最终拟合只使用 cutoff 前已成熟且排除校准的数据。

## 完整性与审批

模型、特征顺序、策略规范、训练行情和 release 均有 SHA-256。每个新 release 默认 `deployment_allowed=false`；必须完成观察、账户预检和哈希绑定审批后才产生执行授权。授权必须发生在 `effective_end` 前。

