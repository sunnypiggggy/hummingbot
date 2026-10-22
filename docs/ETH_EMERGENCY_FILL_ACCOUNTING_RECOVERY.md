# ETH 紧急退出成交记账与恢复

## 修复范围

2026-10-11 修复 DCA 独立退出中“成交已发生，复核失败导致漏记账”的问题。只修复经济流水消费、订单证据刷新和异常展示，不修改模型、本金、收益偏移、风险峰值或风控阈值。

## 成交与退出完成是两个事实

- `liquidation_attempts` 保存确定性订单、原始成交响应和查询结果。
- `economic_fills` 按账户绑定、交易对、交易所成交 ID 唯一登记。数量、成交价格、手续费资产和费用不可因余额复核失败而清零；身份冲突拒绝重复归属。
- `economic_consumption` 保存消费者进度。计算仍读取完整经济事实，消费回执不会导致重启后漏算。
- `EXECUTED_PENDING_VERIFICATION` 表示成交已入账、退出仍待核验；并非完成、Risk-On 或交易授权。
- `COMPLETED` 仍须订单/成交、余额、活动订单/Executor及剩余风险四项核验通过。各项结果独立保存，原错误保留。

Guard 先用 GET 查询未结清任务的原客户端订单 ID，再重算归属。缺口门不能阻止查询和补记，但新卖单仍须通过归属、余额、精度及资产租约检查。未知订单不得重复提交；终态部分成交仅沿原任务的确定性剩余子订单继续执行。

DCA 和 Report 使用同一份已登记成交事实，排除普通 `TradeFill` 和旧紧急调整中已存在的相同成交/订单。基础币费用减少基础币，报价币费用扣报价币一次；第三币费用缺乏可信估值时降级，不启用 BNB 抵扣。

生产共享 SQLite 使用 WAL，Report 的挂载为只读；没有可用 SHM 时不能直接打开该数据库。DCA 每 15 秒从一致性事务导出 `economic_fills_readonly.json`，Report 优先读取该只读合同，校验 schema、内容哈希及 180 秒新鲜度。合同失败时报告降级，不授予 Report 写库权限，也不使用 `immutable=1` 忽略正在写入的 WAL。经济事实及消费进度仍以共享 SQLite 为权威，导出不产生第二笔经济流水。

## 本次事故的补记预览

2026-10-08 的 ETH-USDT 市价退出已成交 0.057 ETH，分为 0.005 和 0.052 ETH 两笔，成交均价 2431.84 USDT：

| 项目 | 经济变动 |
|---|---:|
| ETH | -0.057 |
| 成交收入 | +138.61488 USDT |
| 实际手续费 | -0.13861488 USDT |
| 净现金 | +138.47626512 USDT |

本次只消费原成交，不再次下单，不改写启动库存或通过调整本金凑平。剩余 DCA 归属由完整账本重算，不写死事故数量。

## Grid 与审批

Grid 交易所订单快照与完整紧急预检分开发布。账户绑定及来源检查成功即可更新真实订单证据；归属缺口仍使紧急就绪和普通权限失败。缺口修复后须通过现有连续健康检查，ETH 才可自然重建单代网格，BTC Risk-Off 不受本次修复解除。

模型审批保留 12 小时默认批准、证据送达及全部硬门槛。候选包、证据和授权不因记账修复而改写；达到审批时间前仍正常等待，预热/激活遵循现有调度。

## 异常与运维

- 持续退出或成交/归属待复核进入 `exit_reconciliation:<机器人>` 故障周期，频道与当前异常共用同一来源。触发与恢复各通知一次，不按检查循环重复创建事件。
- `reconciliation_checks` 为复核次数，`exit_order_attempts` 为共享订单任务中的真实请求尝试数，不将数万次检查解释为卖单。
- Report 只读读取共享经济事实；保留现有 Cardputer MQTT wrapper、secret、CA 和覆盖配置。
- 本次只更新 DCA Guard、Grid Guard、Report，不重启交易机器人、代理或 Stock，不解除有效冷却与锁存。
- Caddy 的 `context canceled` 不能单独证明业务故障；需授权客户端的真实链路证据。未完成该验收时明确保留结论，不通过降低日志级别或放宽客户端证书掩盖。

## 回归测试

```powershell
python -m pytest test/test_emergency_fill_recovery.py test/test_account_inventory.py test/test_dca_live_safety.py test/test_grid_guard_shadow.py test/test_dca_live_report.py test/test_telegram_notifications.py test/test_v22_runtime_generation.py test/test_v22_weekly_release_manager.py -q -p no:cacheprovider
python -m pytest test/test_real_risk_scenarios.py -q -p no:cacheprovider
```

覆盖事故精确补记、重复消费、真实 SQLite 重启、普通成交重叠、错误成交身份、未知订单不重发、余额延迟、部分成交、活动执行器和真实缺口阻止完成，以及 Grid 快照刷新但归属门不绕过。

生产备份、只读预览、构建日志与定向 Compose 覆盖保存在 OCI `.eth-exit-rollouts/20261011-fill-accounting-v1/`。SQLite 使用一致性备份；期初、旧任务及不可变发布包保持原样。

## 2026-10-11 生产执行记录

时间均为北京时间。16:53:03 自动补记原订单的两笔成交，0.057 ETH 减量及 138.47626512 USDT 净现金只记一次。原任务通过四项核验后为 COMPLETED，保留原失败原因及补记审计；BTC、ETH 归属缺口均为零。DCA ETH 进入组合保护 COOLDOWN，冷却截止为 2026-10-12 04:53:03；这不是保证恢复交易时间，仍须其他门及原恢复条件共同放行。

16:54:08 更新 Grid Guard 后，ETH 自然重建 3 BUY / 10 SELL。17:10 的交易所 GET 核验：13 个订单 ID 与运行账本完全一致，无同方向同价格重复单；SELL 剩余合计 0.04 ETH，不超过归属 0.040198 ETH。BTC 仍为 Risk-Off、无活动订单。Grid 及两台 DCA 交易容器 ID 和启动时间均未改变。

17:09 更新 DCA Guard 与 Report 的只读成交合同兼容版本；报告已正确展示 DCA ETH 归属残余 0.00001442090706493699903730061 ETH，警告列表为空。Report MQTT wrapper、secret、CA 和覆盖配置均继承原部署，未使用基础 Guard 镜像替代。

候选的九项运行预检及 12 张 PNG 送达回执均通过；截至本记录仍为 AWAITING_APPROVAL，继续等待 19:00:37 的原 12 小时审批截止，未伪造批准、预热或激活结果。当前签名周 23:00 到期，后续审批和周切换仍由 scheduler 执行。

本地已通过 201 项目标回归、45 项有状态真实场景和 61 项 Grid/恢复/管理展示测试，共 307 项。`test_grid_live_runtime_risk.py` 在本地因缺少 `eth_account` 无法收集，不计为通过；未为测试开启实盘或安装替代生产依赖。新只读合同增加损坏、陈旧及无 WAL 访问回归验证。涉及本地故意断开的 HTTP 场景使用 localhost 的 NO_PROXY，避免系统代理把断连替换为 503，不削弱场景断言。

代理只读分类确认：样本中的取消全部位于 xHTTP `/api/v3/updates` 路由、HTTP/2，错误为 `context canceled`，不能据此推断 Hermes 或交易 API 故障。用户确认忽略代理客户端验收，因此本次不修改 Caddy、不降低日志级别、不放宽 mTLS，也不宣称授权客户端链路已验证。

17:24:27 最终只读验收通过，覆盖最后一次 Report 更新后 919 秒（15 分 19 秒）：三个目标服务健康，交易容器身份不变，两个资产归属缺口为零，原清仓成交仍恰好两条，没有新增经济下单尝试；ETH 仍为 3 BUY / 10 SELL、无重复价位、SELL 覆盖通过。DCA ETH 保持 COOLDOWN，BTC 保持模型限制；收益报告无警告、当前异常为空，MQTT 已发布 13 帧且无错误。审批仍未到截止点，本次不宣称候选已批准或周切换已完成。

验收证据为 OCI 私有发布目录 `acceptance-latest.json`，SHA256：`b09499a770ffb31610aec41797c82f26d5de2beb291677c8ff9dfbdee80b7174`。定向发布文件为该目录下 `context/compose.eth-exit.yml`，须与现有 Cardputer overlay 同时使用；不可用基础 Compose 覆盖 Report 版本。未提交 Git、未修改模型授权、冻结包或 Stock 状态。
