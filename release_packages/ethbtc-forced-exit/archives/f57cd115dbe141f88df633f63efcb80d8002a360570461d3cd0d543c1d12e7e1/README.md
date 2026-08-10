# ethbtc-forced-exit OCI 完整源码归档

本目录是 OCI 当前 Grid、DCA、v22、Telegram 运行源码、配置、测试、Docker/Compose、发布族和模型依赖的内容寻址快照。`repository/` 保持 OCI 相对路径，可用于逐文件差异核验和重建。

运行状态、数据库、余额、日志、API/Telegram/交易所密钥和环境文件不归档。release 中的模型二进制仅用于依赖闭包和回溯，不会作为 Telegram 附件发送，也不授予交易权限。

包内完整性校验：

```bash
python tools/verify_ethbtc_oci_source_archive.py <archive-dir>
```

与 OCI 源目录逐文件校验：

```bash
python tools/verify_ethbtc_oci_source_archive.py <archive-dir> --source-root /home/ubuntu/extra_drive/hummingbot
```
