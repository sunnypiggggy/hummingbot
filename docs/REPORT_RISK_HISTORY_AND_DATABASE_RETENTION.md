# Report统一风控历史与数据库治理

## 职责与口径

Report是`risk_history.sqlite`唯一写入者。Guard、Grid执行、scheduler、FOMC和统一库存继续保留各自运行、恢复和审计证据；不改变风控控制循环。

归档从本次上线时开始，初次已有JSONL游标位于文件末尾，不追认旧历史。事件归档独立于Telegram发送开关、附件生成和outbox清理。Bot只读消费`risk_history.json`或`RiskHistoryReader`；当前权限仍来自新鲜运行合同。Cardputer接入及MQTT按用户最新选择暂不实施，不部署旧HTTPS上传服务。

快照每60秒采集，采用Guard真实`last_success_at`判断新鲜度。超过180秒的采样间隔断开，不能从概率、当前门或最后一次Risk-Off推算未知历史。图表不在归档起点之前补阴影。

## 保留规则

业务历史目标370天，执行必要证据例外。概率/价格、简单稽核、Bot操作审计、新风控历史适用370天；会话15分钟期限不变。

**Telegram已发送记录及容量压力清理完全保持原策略**，不改90天配置或紧急7天路径。统一风控数据库与outbox分离，消息清理不删除风控历史。

TradeFill、费用、资金分配、累计收益输入、Executors、订单、恢复/锁存、资产租约和经济幂等记录不盲目按年龄删除。DCA归属仍重算历史成交，相关成交长期保留。不通过修改期初或重置峰值压缩账本。

Report维护自身数据库。OCI每日北京时间03:10执行白名单维护，另有最多120秒随机延迟；SQLite只清理可证明无依赖的市场历史和操作日志，发现入向外键或未经审查的触发器则跳过。Hummingbot MarketData时间戳为`SqliteDecimal(6)`，是秒乘10^6，不是毫秒。库存金融事件、未投递事件及当前周期证据保留。

API PostgreSQL只清理`account_states`及关联`token_states`、`controller_performance_snapshots`的370天前历史。逐账户/连接器、机器人/控制器的最新时间快照及同时间记录全部保留，即使其已过370天。子记录与父记录在同一事务中删除；出现新外键或触发器时拒绝清理。核验依据是实际API仓库实现：这两类表保存展示快照，不是经济流水。

API的`orders`、`trades`、`executors`、`bot_runs`、`position_holds`、`position_snapshots`、资金费及Gateway经济证据不按年龄删除。没有建立安全依赖证明的表只盘点不删除。发布包、JSON审计、备份不受数据库期限自动删除。实例数据库只按当前运行文件名白名单维护，数据目录内的备份数据库不纳入。

SQLite每表每日最多删除5000行，API每类父快照最多1000条，并报告尚待处理数量。写锁或语句超时跳过，下一次维护重试，不长时间阻塞业务。释放的数据库页可被后续写入复用，但主文件未必立即缩小；不在运行库执行完整`VACUUM`或`VACUUM FULL`，也不把可复用空间称为已归还磁盘。

库存事件消费Guard已有的原始JSONL：Guard在确认投递前先持久化事件，该流程不依赖Telegram开关。Report不通过只读挂载打开生产WAL数据库，避免共享内存权限故障；库存任务、租约和经济证据继续由Guard保存。

## Stock报价存储

后续用户已改为授权废弃旧PAPER账本并暂停交易；本节旧报价迁移上线流程不再执行。以[Stock PAPER重设计与暂停规则](STOCK_PAPER_REDESIGN.md)为准，不自动继续旧迁移、旧run或建库。每日维护只包含非PAPER SQLite和API PostgreSQL，不调用Stock容器。以下报价设计作为后续重新设计的技术基础，不代表已经恢复模拟交易。

最新可信报价及同时间事件ID存入`paper_quote_latest`。较早报价和重复ID不再产生撮合流动性；成交引用原报价另存`paper_fill_quote_evidence`。这些都与现金/成交同事务更新，摘要不参与撮合。

报价摘要最近30天为分钟粒度，第31–370天为小时粒度，保存实际首尾报价、范围、数量及覆盖时间，不补造行情。迁移按固定物理页范围顺序读取，游标及摘要同事务提交，可中断重启；记录原表物理身份及页数，发生重写或扩展立即停止。迁移期间不得重写旧表，未完整转换或缺成交报价证据禁止回收旧表。

500 MiB是`hummingbot_stocks`逻辑数据库表和索引容量目标，不含API数据库、WAL和备份。400/450/500 MiB分别为治理/警告/严重告警，绝不限制订单、成交记录或保护性退出。必需证据导致超限时如实告警，不缩短370天来掩盖。

## 运维

当前只维护非PAPER数据库。先执行SQLite一致性备份和API PostgreSQL备份，并在隔离数据库验证恢复；保存维护脚本、调度配置及容器身份。Stock旧账本已经按用户授权备份后删除，任何恢复或新建PAPER run仍需新的明确授权，不能执行旧报价迁移上线命令。

维护前运行`python scripts/database_retention.py --root <OCI部署目录>`只读预览；明确证据和备份后加`--apply`。锁超时跳过重试，不放宽生产权限，不向Report挂交易密钥或Docker socket。

API预览使用`python scripts/api_database_retention.py`，确认后加`--apply`。目标固定为本机`hummingbot-api-postgres`中的`hummingbot_api`，不接受任意DSN或库名。OCI使用`ops/database-retention.service`与`ops/database-retention.timer`，任务低优先级运行；查看`systemctl list-timers database-retention.timer`及`journalctl -u database-retention.service`确认实际调度和结果。暂停维护只需停用timer，不重启交易容器。

测试入口：`python -m pytest test/test_report_risk_history.py test/test_database_retention.py test/test_api_database_retention.py test/test_management_probability_chart.py test/test_telegram_notifications.py -q`。

真实API PostgreSQL演练：`test/support/api_retention_check.py`只连接专用`hummingbot_api_retention_test`库、受限测试角色，覆盖预览、最新旧快照、同时间快照、依赖拒绝、中断回滚、批量清理及经济证据保护；测试库须提前创建，结束后按明确目标删除。Stock旧报价模拟测试仍只使用独立测试数据库，不需要启动生产Stock服务。

部署结果必须分别记录保留策略、实际容量、恢复验证及未完成事项；容器健康不能替代容量和账本验收。
