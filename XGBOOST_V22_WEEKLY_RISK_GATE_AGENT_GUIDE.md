# XGBoost v22 周度 Risk-off：Agent 操作说明

## 1. 目的与当前结论

v22 用于复现旧 v21 Plotly 中真正采用的策略语义：**每周 walk-forward 重训模型，并使用该周独立校准的概率阈值**。

它不是 v21 最终全量重拟合、固定绝对阈值的模型。两者不可混用。

当前历史结论仍为 `NO-GO`：

- 250天净收益：`-4.102051765 FDUSD`
- 拼接最大回撤：`-15.452621%`
- BTC收益：`+3.566831 FDUSD`
- ETH收益：`-7.668882 FDUSD`
- 单对停止：25次
- 组合停止：1次
- `deployment_allowed=false`
- `promotion_authorized=false`

因此，其他Agent只能进行研究、重放、生成影子信号和构建待审核周模型，不能授权或接管真实Grid。

## 2. 必须保持的策略语义

每个交易对独立运行长期Risk-off：

- BTC：`long_event_72h / full_structure / xgb_34`，进入分位数98%。
- ETH：`long_event_72h / directional_persistence / xgb_16`，进入分位数98.5%。
- BTC/ETH模型、概率、阈值和状态完全独立。
- 周边界切换到该周签名模型和阈值，但Risk-off状态、武装期、冷却和恢复计数不能重置。
- 进入使用`persistent_bearish`结构确认。
- 恢复使用`adaptive_relief`，普通恢复需要3个完整4小时结构改善，强恢复可缩短至2个周期。
- Risk-off只建议暂停对应交易对的普通Grid BUY。
- 不撤销SELL，不触发市场卖出，不影响48小时库存退出，不影响风控恢复基准库存的BUY。
- 不包含短期插针通道，不允许机制1回退。

缺少当周签名模型、模型或Schema哈希不匹配、K线缺失、概率非法、状态损坏或信号过期时：

- BTC和ETH均fail-closed。
- `buy_enabled=false`。
- 不产生SELL、Taker减仓或其他交易动作。
- 禁止沿用上一周模型，也禁止回退v21最终模型。

## 3. 关键代码

| 文件 | 用途 |
|---|---|
| `scripts/xgboost_long_risk_gate_v22.py` | v22策略契约、周模型选择和连续状态机执行器 |
| `scripts/freeze_xgboost_long_risk_gate_v22.py` | 从历史周折重新拟合并冻结BTC/ETH周模型 |
| `scripts/build_xgboost_v22_weekly_report.py` | 使用冻结包精确回放Grid并生成Plotly |
| `scripts/build_xgboost_v22_shadow_signal.py` | 生成影子心跳和反事实BUY建议 |
| `scripts/grid_xgboost_shadow_gate_v22.py` | 严格的非授权信号契约与校验器 |
| `scripts/append_xgboost_v22_signed_week.py` | 使用完整新数据构建下一个待审核签名周包 |
| `test/test_xgboost_long_risk_gate_v22_weekly.py` | 周模型、历史一致性和fail-closed测试 |

v22复用了`scripts/xgboost_long_risk_gate_v21.py`中的基础状态结构。修改该文件时必须同时回归v21和v22。

## 4. 当前产物

根目录：

```text
results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/
```

主要文件：

```text
shadow_package/models/xgboost_long_risk_gate_v22_weekly.joblib
shadow_package/shadow_lock.json
application_bundle/summary.json
application_bundle/risk_states.csv.gz
application_bundle/risk_events.csv
application_bundle/risk_intervals.csv
xgboost-grid-long-risk-gate-v22-weekly-250d_plotly.html
```

当前冻结包包含BTC和ETH各36个周模型，共72个模型。当前哈希仅用于核验现有工作区；每次合法重建后都会变化：

```text
model_sha256    = 3a0eea362858830757db5bf3de3892b88923e5cbce52b5f3e72c8cdabbecc338
feature_sha256  = 1fdc99293e83bd00b68b18174f3dc4f854f5b972c294f8c64e13ad26e8498da1
strategy_sha256 = 5de7c72a35911bbaad43fe342a068cb80a276c2b145ded8bfce79bd5eb1f29cb
```

第36周的历史回测在2026-07-31截止，但模型签名有效期按完整自然周延长至：

```text
2026-08-02 15:00:00 UTC
```

超过该时间若没有第37周签名模型，必须fail-closed。

## 5. 完整历史重建

在项目根目录执行：

```powershell
& 'C:\Users\sunny\anaconda3\python.exe' -B scripts\freeze_xgboost_long_risk_gate_v22.py --workers 6 --xgb-threads 2
```

该命令会：

1. 从v21锁中读取BTC/ETH固定候选。
2. 使用v19的无前视训练和校准切分重新拟合所有历史周模型。
3. 保存每周训练截止点、树数、校准阈值、执行阈值、模型哈希和有效区间。
4. 与旧周度预测缓存核对概率和阈值。
5. 生成新的v22模型包和影子锁。

然后生成报告：

```powershell
& 'C:\Users\sunny\anaconda3\python.exe' -B scripts\build_xgboost_v22_weekly_report.py
```

报告构建必须满足：

```text
risk_state_mismatches = 0
transition_mismatches = 0
maximum_probability_absolute_error <= 1e-6
maximum_threshold_absolute_delta <= 1e-8
```

若状态或事件不一致，立即停止，不得更新主Plotly或声称复现成功。

### XGBoost浮点边界注意事项

XGBoost `hist`在多线程重新拟合时可能出现约`1e-8`的浮点差异。旧阈值又可能恰好等于离散概率值，导致`probability >= threshold`翻转。

历史冻结流程会：

- 保留原始`calibration_threshold`用于审计。
- 仅在确实发生边界翻转的周折，对`entry_threshold`做最小float32级调整。
- 最终以Risk-off状态和进入/退出事件零差异作为验收标准。

不要删除这层处理，也不要用最终全量模型替代周模型。

## 6. 构建未来一周模型

未来周必须使用从历史起点到本周截止点的完整、连续BTC/ETH 5分钟FDUSD数据。仅有45天实时缓存不足以重建扩展训练集。

先确认：

- BTC和ETH时间戳一一对齐。
- 每5分钟一根，无缺口、重复或非法OHLCV。
- 数据至少到训练截止点前最后一根完整5分钟K线。
- 新截止点必须严格等于当前签名清单的`effective_end`。

使用独立输出目录构建待审核包，禁止覆盖当前包：

```powershell
& 'C:\Users\sunny\anaconda3\python.exe' -B scripts\append_xgboost_v22_signed_week.py `
  --candle-dir '<完整FDUSD五分钟数据目录>' `
  --cutoff '2026-08-02T15:00:00Z' `
  --output-package 'results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/staged_week_37' `
  --xgb-threads 2
```

该脚本只生成`staged_for_review=true`的候选包，并固定：

```text
deployment_allowed=false
promotion_authorized=false
```

Agent不得自行复制候选包覆盖运行包，也不得修改锁文件绕过人工审核。

## 7. 影子信号演练

使用历史时间进行一次性演练：

```powershell
& 'C:\Users\sunny\anaconda3\python.exe' -B scripts\build_xgboost_v22_shadow_signal.py `
  --cache-dir 'results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/smoke/candles' `
  --seed-cache-dir 'results/backtests/eth_xgboost_long_risk_gate_v15_250d/extended_candles' `
  --output 'results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/smoke/signal.json' `
  --state 'results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/smoke/state.json' `
  --observed-at 1785509700
```

预期BTC和ETH均选择fold 36，且输出仍为影子契约：

```text
shadow_mode=true
deployment_allowed=false
promotion_authorized=false
buy_enabled=false
runtime_action=observe_only
```

`recommended_buy_enabled`是反事实建议；公共`buy_enabled`固定为false，避免误接入后开放BUY。

## 8. 测试

Windows环境应把pytest临时目录放在工作区，避免系统Temp ACL错误：

```powershell
& 'C:\Users\sunny\anaconda3\python.exe' -B -m pytest -q `
  --basetemp .pytest_tmp_v22_agent `
  test\test_xgboost_long_risk_gate_v22_weekly.py `
  test\test_xgboost_long_risk_gate_v21_shadow.py
```

当前预期：

```text
19 passed
```

还应执行：

```powershell
git diff --check
```

换行符警告可以记录，但任何真实空白错误、语法错误、哈希不一致或测试失败都必须修复。

## 9. Plotly验收

主报告必须来自冻结v22周模型包，不能读取旧CSV后单独画图冒充应用结果。

检查：

- BTC和ETH价格曲线存在。
- 概率曲线按周模型生成。
- 阈值曲线展示每周独立执行阈值。
- hover中包含fold编号和阈值。
- 进入/退出时间与`risk_events.csv`一致。
- BTC和ETH Risk-off阴影开关相互独立。
- 关闭阴影不能隐藏价格、概率和事件标记。
- 标题明确写明`weekly walk-forward`和`NO-GO`。

## 10. 禁止事项

其他Agent不得：

- 把v21最终重拟合模型或固定阈值写入v22。
- 在周边界重置状态机。
- 当周模型缺失时沿用上一周模型。
- 使用机制1作为故障回退。
- 添加短期插针通道。
- 让Risk-off触发SELL、Taker退出或撤销已有SELL。
- 把`recommended_buy_enabled`直接当作已授权交易指令。
- 将`deployment_allowed`或`promotion_authorized`改为true。
- 根据已查看的250天结果继续调参后仍称为样本外证据。

## 11. 修改后的交付清单

任何修改v22的Agent都应在交付中明确报告：

1. 修改了哪些代码和策略语义。
2. 新模型、特征和策略SHA-256。
3. 每对周模型数量及签名有效区间。
4. 与旧周度状态、事件的差异数量。
5. Grid收益、回撤、单对停止、组合停止和分对收益。
6. 测试命令和通过数量。
7. 是否存在未签名未来周。
8. `deployment_allowed=false`和`promotion_authorized=false`是否保持。

只要历史结论仍为`NO-GO`或未来周缺失，最终结论必须是“保持影子/研究状态，不接管Grid”。
