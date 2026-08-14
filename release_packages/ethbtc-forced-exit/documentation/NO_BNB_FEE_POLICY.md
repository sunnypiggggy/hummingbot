# 禁止使用 BNB 支付现货手续费

## 生产策略

- Binance 账户级现货 BNB 手续费抵扣必须保持关闭：`spotBNBBurn=false`。
- Grid Guard 与 DCA Guard 启动预检会读取该账户设置；若发现 BNB 抵扣开启，Guard 立即 Fail-Closed，不会以已武装状态运行。
- 本策略不允许自动重新开启 BNB 抵扣，也不会把 BNB 余额作为任何机器人资金归属的一部分。
- 关闭 BNB 抵扣后，实际手续费由 Binance 按成交交易对相关资产收取；所有成交仍以交易所返回的手续费资产和数量为最终审计依据。

## Grid 费用模型

- 普通 FDUSD `LIMIT_MAKER` 网格成交继续使用 Maker `0%` 模型。
- 强制退出、自动重入等 `MARKET` 成交使用 Taker `0.1%`。
- 若 Hummingbot 暂时无法把市场单手续费换算为报价币，运行时使用成交额的 `0.1%` 作为保守回退，并写入 `quote_fee_fallback_applied` 审计事件；不得回退到 Maker `0%`。
- 交易所稍后返回可换算的真实手续费时，以真实成交记录为准，避免重复计费。

## DCA 与紧急通道

- DCA Guard 和 Grid Guard 的独立 Binance 通道使用同一个账户级检查，禁止出现主交易进程关闭 BNB、紧急通道仍假定 BNB 扣费的分叉。
- Guard 的生产状态必须包含 `fee_policy.spot_bnb_burn_disabled=true`；缺失、读取失败或值为 `false` 均不通过启动预检。

## 2026-08-14 历史修正

- Grid `ETH-FDUSD` 市价重入成交 `exchange_trade_id=978446577` 曾实际支付 `0.00012237 BNB`。
- 按成交分钟 `BNB-FDUSD=612.16` 换算为 `0.0749100192 FDUSD`，数据库以微单位记录为 `74910`。
- 运行账本同步减少报价币、增加基础币成本及累计费用各 `0.0749100192 FDUSD`；幂等标记为 `978446577:no-bnb-fee-fix-v1`。
- 修正前数据库及运行状态均已在 OCI 原实例数据目录保留 `before_no_bnb_fee_fix` 备份。

## 上线验收

1. 两个 Guard 健康且重启次数为零。
2. Binance 账户查询返回 `spotBNBBurn=false`。
3. DCA Guard 状态包含 `spot_bnb_burn_disabled=true`。
4. Grid 普通挂单仍为 Maker；市场退出/重入的换算失败测试必须按 `0.1%` 记费。
5. 最近日志不得出现 BNB fee、手续费换算异常、订单拒绝或重复成交。

此配置是账户级安全约束。回滚代码或模型不得顺带重新开启 BNB 手续费抵扣。
