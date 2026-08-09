# ethbtc-forced-exit

> 生产状态（2026-08-09）：OCI 已从本发布族派生并批准 release
> `73f59bef...e9d60`，以 v22 live 模式服务 Grid/DCA。下面的 `NO-GO` 描述仍准确
> 指向本目录的原始冻结离线证据，不代表当前运行合同未获授权。线上权威说明见
> [documentation/ONLINE_MODELS.md](documentation/ONLINE_MODELS.md)。

这是 `v22-risk-off-forced-exit-v2` 的封存离线发布候选包，包含 BTC/ETH Grid 与 DCA
回放所需的源文件、冻结 v22 模型、冻结状态、四份实际使用的行情文件、证据产物和哈希清单。

状态：`NO-GO`、`offline_only=true`、`deployment_allowed=false`。本包可完整性校验和离线回放，
不能单独连接交易所或授予实盘权限；生产权来自派生 release、完整性预检和外部哈希绑定审批回执。

校验：`python tools/verify_package.py .`；关键行为自测：`python tools/smoke_test.py .`

严格上线门检查（当前应失败）：`python tools/verify_package.py . --require-deployable`

重新回放（跨平台）：`python run_replay.py`。Linux 也可执行 `bash run_replay.sh`；允许脚本执行的
Windows 环境可执行 `./run_replay.ps1`。输出写入
`reproduced/`，不会覆盖 `evidence/` 中的封存结果。
