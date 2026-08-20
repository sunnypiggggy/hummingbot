# v22 无故障周切换与统一交易状态

## 目标

v22 发布包切换与周模型 fold 切换是两个不同动作。候选发布包在周边界前完成隔离推理和原子热切换；到周边界时，已激活发布包只在内部选择下一周 fold，不再改写 release 指针、授权文件或实时合同路径。

这套流程消除切换期间“新 release + 旧状态”或“旧 release + 新授权”的混读。它不降低完整性门槛：周边界到达后若没有健康且已签名的下一周模型，`model_signal` 必须为 `UNAVAILABLE`，系统进入完整性 Fail-Closed，不能回退过期模型。

## 固定时间线

- `T-16h`：生成候选、360 天与重点窗口回测、发送频道审批材料；保持 12 小时默认审批窗口，无人拒绝且硬门槛持续通过时约在 `T-4h` 自动批准。
- `T-60m`：复核模型、特征、策略、行情、审批、过滤器和库存归属。
- `T-35m`：在 `runtime/staging/<id>/` 生成候选状态和合同；不得写线上合同。
- `T-30m`：候选与当前发布包使用相同 `signal_ts` 和当前周模型，通过语义一致性后只原子替换 `runtime/current.json`。
- `T`：当前 generation 自然选择下一周 fold；随后更新仅供展示和运维使用的 `active_deployment.json`、`current` 链接及授权副本。

## Runtime generation 合同

每个 generation 位于 `runtime/generations/<generation_sha256>/`，至少包含：

- `manifest.json`：release、授权、前驱 release、fold 边界、预热状态/合同哈希；文件哈希就是 generation ID。
- `gate_state.json`：该 generation 独立的 BTC/ETH 状态机状态。
- `shadow_contract.json`：隔离推理结果，不是消费者读取的线上合同。

唯一实时提交点是 `runtime/current.json`。消费者每个周期先读取一次该指针，然后固定使用同一 generation。指针通过临时文件、落盘和 `os.replace` 提交；因此读取者只会得到完整旧 generation 或完整新 generation。

线上 v22 合同增加：

- `runtime_generation`
- `predecessor_release_sha256`
- `state_lineage_sha256`
- `cutover_phase`
- `fold_boundary`
- `model_signal`：`RISK_ON`、`RISK_OFF` 或 `UNAVAILABLE`
- `system_health`

`RISK_OFF` 是健康模型的风险判断；`UNAVAILABLE` 是模型或完整性不可用。两者不得混为同一状态。

## 预热硬门槛

预热必须同时满足：

- 当前线上合同健康，其 release 精确等于候选声明的 predecessor。
- BTC/ETH 的 `signal_ts`、当前周编号、当前周模型哈希和 Risk-Off 状态与线上一致。
- 候选 release、授权、模型、特征、策略、行情与状态血缘哈希完整。
- Grid 的 `BTC/ETH-FDUSD` 与 DCA 的 `BTC/ETH-USDT ← BTC/ETH-FDUSD` 映射完整。

任一条件失败只产生 `MODEL_CUTOVER_PRECHECK_FAILED`，保留当前健康 generation 并重试，不覆盖线上合同、不触发 Risk-Off、不清仓。

## 状态继承规则

- 已经 Risk-Off：继承触发时间、冷却和恢复进度。
- 当前 Risk-On：在新 fold 清除 `above_entry_count`、`armed_until` 等旧周入场证据。
- 新 fold 必须得到切换后的完整 1 小时概率和新的完整 4 小时结构确认；release、阈值或 fold 编号变化本身不能触发 Risk-Off。

## 回滚与真正故障

- `T` 前候选运行失败：若前驱仍有有效签名覆盖，原子恢复上一 generation；当前交易权限保持不变。
- `T` 后禁止回退前驱或上一周模型。
- `T` 后无有效签名周：输出 `signed_week_unavailable`，`model_signal=UNAVAILABLE`，完整性门关闭并按既有退出/锁存规则处理。
- 未提交的 staging generation 可删除；producer 重启只从已提交 generation 恢复。

## 四小时状态报告

四张单机器人 PNG 固定为 `1440×3200`。顶部必须显示系统健康、是否正常交易、最终 BUY/SELL、generation、release、模型周及切换阶段。门控表包含七类风险机制以及完整性、资金预算、库存归属、恢复阶段和控制器落地门。

只有同时满足以下条件才显示“正常交易”：进程运行、数据和合同新鲜、恢复阶段为 `ACTIVE`、控制器已落地、最终 BUY 与 SELL 均放行。候选预热期间显示“候选预热中／当前模型继续交易”，不能显示为 Risk-Off 或系统故障。

## 上线顺序

1. 先部署支持 generation 指针的 Grid/DCA 消费者和报告器。
2. 影子运行一个完整预热周期，核对两消费者看到同一 generation。
3. 再启用 producer 的原子提交流程。
4. 上线验收观察完整性事件、非模型 Risk-Off、BUY/SELL 虚假跳变均为零。

本文件是发布族文档；已生成的不可变 release 不原地修改。后续候选封包时由打包流程纳入 manifest。
