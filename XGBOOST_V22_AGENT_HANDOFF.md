# XGBoost v22 周度长期 Risk-off：Agent 交接说明

## 1. 当前结论

v22 的目标是精确实现旧 v21 Plotly 所使用的概率语义：

- 每个周折重新训练 BTC/ETH 独立 XGBoost 模型。
- 每个周折使用各自最后14天成熟校准样本生成概率阈值。
- 周边界只切换模型和阈值，Risk-off 状态机连续运行，不重置状态。
- 模型只给出是否暂停对应交易对普通 Grid BUY 的建议。
- 不撤销 SELL、不触发 Taker 卖出、不影响48小时额外库存退出和风控恢复 BUY。

当前结论仍为 `NO-GO`：

- `deployment_allowed=false`
- `promotion_authorized=false`
- `market_sell_action=false`
- `short_spike_enabled=false`
- `mechanism1_fallback_allowed=false`

不得因为代码或测试通过而改变这些字段。

历史250天精确回放：

- 净收益：`-4.10205176515791 FDUSD`
- 拼接最大回撤：`-15.452621089548902%`
- BTC收益：`+3.566830609864212 FDUSD`
- ETH收益：`-7.668882375022065 FDUSD`
- 单对停止：25次
- 组合停止：1次
- Risk-off：`1777.9166666666667 pair-hours`

因此v22目前只能用于研究和影子信号，不能接管真实Grid。

## 2. 与v21的关键区别

v21最终应用包使用“截至2026-07-31全量重拟合模型 + 单个固定绝对阈值”。旧v21 Plotly则使用“周度walk-forward模型 + 每周变化阈值”。两套概率语义不同，是此前应用结果和Plotly不一致的根因。

v22选择旧Plotly语义，并把下列内容一起写入模型包：

- BTC/ETH各36个历史周模型，共72个模型。
- 每周训练截止点、有效开始和结束时间。
- 每周原始校准阈值和执行阈值。
- 每周精确树数、模型SHA-256和标签成熟审计。
- BTC/ETH特征顺序及独立状态机配置。
- 完整策略说明及策略Schema哈希。

主模型包：

`results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/shadow_package/models/xgboost_long_risk_gate_v22_weekly.joblib`

当前模型SHA-256：

`3a0eea362858830757db5bf3de3892b88923e5cbce52b5f3e72c8cdabbecc338`

策略SHA-256：

`5de7c72a35911bbaad43fe342a068cb80a276c2b145ded8bfce79bd5eb1f29cb`

## 3. 固定的交易对配置

BTC：

- 模型：`long_event_72h / full_structure / xgb_34`
- 进入分位数：98%
- 连续确认：1根完整1小时K线
- 武装时间：48小时
- 最短Risk-off：48小时
- 冷却：48小时
- 恢复确认：3个完整4小时周期
- 进入：`persistent_bearish`
- 恢复：`adaptive_relief`

ETH：

- 模型：`long_event_72h / directional_persistence / xgb_16`
- 进入分位数：98.5%
- 连续确认：2根完整1小时K线
- 武装时间：48小时
- 最短Risk-off：48小时
- 冷却：24小时
- 恢复确认：3个完整4小时周期
- 进入：`persistent_bearish`
- 恢复：`adaptive_relief`

状态机代码在：

`scripts/xgboost_long_risk_gate_v22.py`

v22复用v21经过测试的状态数据结构和进入/恢复实现，但事件ID使用v22模型版本。

## 4. 重要文件

- `scripts/xgboost_long_risk_gate_v22.py`
  - v22策略Schema、模型包校验、周模型选择及连续状态执行器。
- `scripts/freeze_xgboost_long_risk_gate_v22.py`
  - 使用6进程重新拟合并冻结历史周模型。
- `scripts/build_xgboost_v22_weekly_report.py`
  - 使用冻结模型包进行Grid精确回放并生成Plotly。
- `scripts/build_xgboost_v22_shadow_signal.py`
  - v22影子信号生产器。
- `scripts/grid_xgboost_shadow_gate_v22.py`
  - 严格、不可授权的v22信号契约。
- `scripts/append_xgboost_v22_signed_week.py`
  - 从完整最新K线训练下一周模型，写入独立待审核包，不修改源包。
- `test/test_xgboost_long_risk_gate_v22_weekly.py`
  - 周模型、哈希、状态一致性和fail-closed测试。

冻结锁：

`results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/shadow_package/shadow_lock.json`

主Plotly：

`results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/xgboost-grid-long-risk-gate-v22-weekly-250d_plotly.html`

应用回放摘要：

`results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/application_bundle/summary.json`

## 5. 从头冻结历史v22包

在项目根目录运行：

```powershell
& 'C:\Users\sunny\anaconda3\python.exe' -B scripts\freeze_xgboost_long_risk_gate_v22.py --workers 6 --xgb-threads 2
```

冻结脚本会：

1. 从v21锁读取BTC/ETH获胜候选。
2. 从v19特征面板读取无前视训练数据。
3. 对BTC/ETH每个周折独立重新训练模型。
4. 对比旧缓存概率和阈值。
5. 嵌入模型、阈值、树数、时间边界和全部哈希。
6. 原子写入模型包及影子锁。

XGBoost多线程直方图训练存在数个`1e-8`量级的浮点归约差异。冻结流程允许概率最大误差`1e-6`，但最终必须满足：

- Risk-off状态差异为0。
- `enter/recover`事件差异为0。
- 执行阈值相对旧阈值的最大浮点调整不超过`1e-8`。

不要删除“执行阈值”调整。部分模型分数存在大量并列值，直接使用重新拟合后的原始分位数会改变`probability >= threshold`边界。模型包同时保留：

- `calibration_threshold`：原始周度分位数。
- `entry_threshold`：保证旧超阈判定不变的最小浮点调整值。
- `execution_threshold_adjustment`：两者差值。

## 6. 重新生成应用回放和Plotly

```powershell
& 'C:\Users\sunny\anaconda3\python.exe' -B scripts\build_xgboost_v22_weekly_report.py
```

脚本必须使用冻结包重新计算概率、状态、Grid交易和指标，不允许直接复制旧CSV作为v22结果。

验收字段位于`application_bundle/summary.json`：

```json
"old_walk_forward_parity": {
  "risk_state_mismatches": 0,
  "transition_mismatches": 0,
  "maximum_probability_absolute_error": 2.501754758910124e-08,
  "maximum_threshold_absolute_delta": 3.725290395606429e-09
}
```

Plotly包含BTC/ETH价格、周度概率、周度阈值、精确进入/退出标记，以及BTC/ETH Risk-off阴影独立开关。

## 7. 运行一次影子信号演练

历史确定性演练示例：

```powershell
& 'C:\Users\sunny\anaconda3\python.exe' -B scripts\build_xgboost_v22_shadow_signal.py `
  --cache-dir results\backtests\xgboost_grid_long_risk_gate_v22_weekly_250d\signal_test\candles `
  --seed-cache-dir results\backtests\eth_xgboost_long_risk_gate_v15_250d\extended_candles `
  --output results\backtests\xgboost_grid_long_risk_gate_v22_weekly_250d\signal_test\signal.json `
  --state results\backtests\xgboost_grid_long_risk_gate_v22_weekly_250d\signal_test\state.json `
  --observed-at 1785509700
```

生产器规则：

- 只使用完整1小时和4小时K线。
- 每个新完整1小时收盘才推进概率和状态。
- 状态跨周连续保存。
- 当前时间必须由唯一签名周模型覆盖。
- 缺少签名周、模型/策略/状态哈希不一致、非法概率或过期时双对fail-closed。
- 影子契约的`buy_enabled`永远为`false`。
- 反事实建议只能读取`recommended_buy_enabled`。

不要把v22影子文件直接配置成当前Grid的已授权信号源。

## 8. 训练并签名下一周模型

每周新模型必须写入新的待审核目录，禁止原地修改当前包。例如：

```powershell
& 'C:\Users\sunny\anaconda3\python.exe' -B scripts\append_xgboost_v22_signed_week.py `
  --source-package results\backtests\xgboost_grid_long_risk_gate_v22_weekly_250d\shadow_package `
  --candle-dir PATH_TO_COMPLETE_FDUSD_5M_CANDLES `
  --cutoff '2026-08-02T15:00:00Z' `
  --output-package results\backtests\xgboost_grid_long_risk_gate_v22_weekly_250d\staged_week_37 `
  --xgb-threads 2
```

续签要求：

- `--cutoff`必须精确等于当前锁的`effective_end`。
- BTC/ETH 5分钟K线必须连续且覆盖到截止点前最后一根完整K线。
- 完整历史必须足够重建扩展训练集，不能只提供最近45天。
- 所有训练记录必须满足`label_ready_ts <= cutoff`。
- 校准集为最后14天成熟记录，与最终拟合数据互斥。
- 新模型有效期固定为`[cutoff, cutoff + 7天)`。
- 输出包继续保持`deployment_allowed=false`。
- 新包必须重新运行回放、状态迁移和哈希检查后才能人工替换影子包。

当前本地FDUSD数据只到`2026-07-31 15:00 UTC`附近，不能训练`2026-08-02 15:00 UTC`截止的第37周模型。禁止通过放宽K线完整性、提前截止或沿用第36周模型绕过此限制。

## 9. 测试命令

Windows默认临时目录存在权限问题，必须显式指定工作区内的pytest临时目录：

```powershell
& 'C:\Users\sunny\anaconda3\python.exe' -B -m pytest -q `
  --basetemp .pytest_tmp_v22_agent `
  test\test_xgboost_long_risk_gate_v22_weekly.py `
  test\test_xgboost_long_risk_gate_v21_shadow.py
```

当前结果：`19 passed`。

同时执行：

```powershell
git diff --check
```

只关注新增修改导致的问题；工作区已有许多用户修改和换行警告，不要重置、覆盖或清理无关文件。

## 10. 后续Agent不得做的事情

- 不得把`historical_verdict`改成通过。
- 不得设置`deployment_allowed=true`或`promotion_authorized=true`。
- 不得用v21最终固定模型替代缺失的v22周模型。
- 不得在缺失当周模型时沿用上一周模型。
- 不得恢复机制1作为故障回退。
- 不得增加短期插针通道。
- 不得让Risk-off信号产生SELL、Taker减仓或撤销已有SELL。
- 不得把BTC状态合并进ETH或把ETH状态合并进BTC。
- 不得重置周边界处的Risk-off、武装、冷却或恢复计数。
- 不得为了获得旧图指标而直接读取预生成状态CSV驱动应用；应用必须从签名模型包重新计算。
- 不得修改当前运行Grid服务或授权锁，除非用户另行明确要求且所有前向验收通过。

## 11. 推荐的下一步

1. 获取截至`2026-08-02 15:00 UTC`且连续完整的BTC-FDUSD、ETH-FDUSD 5分钟历史。
2. 使用续签脚本生成第37周待审核包。
3. 从该周第一个完整小时开始运行独立影子服务。
4. 连续积累至少8个完整、未参与调参的UTC周。
5. 报告信号可用率、模型周切换一致性、Grid反事实收益、回撤和停止事件。
6. 另行人工签署晋级锁；v22自身永远不能自动授权。

