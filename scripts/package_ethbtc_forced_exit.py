#!/usr/bin/env python3
"""Build the self-contained, offline ethbtc-forced-exit release package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ID = "ethbtc-forced-exit"
DEFAULT_TARGET = ROOT / "release_packages" / PACKAGE_ID
FROZEN = ROOT / "results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d"
EVIDENCE = ROOT / "results/backtests/v22_grid_dca_forced_exit_v2"

SOURCES = (
    "backtest_dca_live_local.py",
    "backtest_dca_momentum_guard.py",
    "build_v22_grid_dca_forced_exit_v2.py",
    "build_v22_grid_dca_offline_audit.py",
    "cache_binance_klines.py",
    "compare_roc_sqzmom_plotly.py",
    "dca_live_common.py",
    "plot_v22_forced_exit_v2.py",
    "plot_v22_grid_dca_risk.py",
    "xgboost_long_risk_gate_v22.py",
    "xgboost_long_risk_gate_v22_features.py",
    "xgboost_v22_io.py",
)
GRID_CANDLES = ("binance_BTC-FDUSD_5m.csv", "binance_ETH-FDUSD_5m.csv")
DCA_CANDLES = ("BTCUSDT_5m.csv", "ETHUSDT_5m.csv")
REQUIREMENTS = (
    "joblib==1.4.2", "numpy==2.2.5", "pandas==2.3.1", "plotly==6.0.1",
    "requests==2.32.3", "scikit-learn==1.6.1", "xgboost==3.3.0",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def write_smoke_test(path: Path) -> None:
    write_text(path, SMOKE_TEST)


def build(target: Path, *, replace: bool = False) -> Path:
    target = target.resolve()
    staging = target.with_name(f".{target.name}.staging")
    backup = target.with_name(f".{target.name}.previous")
    if staging.exists() or backup.exists() or (target.exists() and not replace):
        raise FileExistsError(f"refusing to overwrite existing package/staging/backup path: {target}")
    staging.mkdir(parents=True)
    try:
        for name in SOURCES:
            copy_file(ROOT / "scripts" / name, staging / "sources" / name)
        copy_file(ROOT / "scripts/verify_ethbtc_forced_exit_package.py", staging / "tools/verify_package.py")
        write_smoke_test(staging / "tools/smoke_test.py")
        copy_file(ROOT / "test/test_v22_forced_exit_v2.py", staging / "tests/test_v22_forced_exit_v2.py")
        shutil.copytree(FROZEN / "application_bundle", staging / "inputs/frozen_v22/application_bundle")
        shutil.copytree(FROZEN / "shadow_package", staging / "inputs/frozen_v22/shadow_package")
        grid_root = ROOT / "results/backtests/eth_xgboost_long_risk_gate_v15_250d/extended_candles"
        dca_root = ROOT / "data/backtesting_candles"
        for name in GRID_CANDLES:
            copy_file(grid_root / name, staging / "inputs/candles/grid" / name)
        for name in DCA_CANDLES:
            copy_file(dca_root / name, staging / "inputs/candles/dca" / name)
        shutil.copytree(EVIDENCE, staging / "evidence")

        summary = json.loads((staging / "evidence/summary.json").read_text(encoding="utf-8"))
        lock = json.loads((staging / "inputs/frozen_v22/shadow_package/shadow_lock.json").read_text(encoding="utf-8"))
        if summary.get("package_id") != PACKAGE_ID:
            raise ValueError("evidence was not generated with the ethbtc-forced-exit package marker")
        effective_end = datetime.fromtimestamp(int(lock["effective_end"]), timezone.utc)
        now = datetime.now(timezone.utc)
        release = {
            "schema": "ethbtc-forced-exit-release-v1",
            "package_id": PACKAGE_ID,
            "execution_policy_version": summary["execution_policy_version"],
            "release_stage": "offline_release_candidate",
            "verdict": "NO-GO",
            "offline_only": True,
            "deployment_allowed": False,
            "promotion_authorized": False,
            "orders_submitted": False,
            "frozen_model_effective_end": effective_end.isoformat(),
            "frozen_model_expired_at_packaging": now > effective_end,
            "deployment_blockers": [
                "frozen v22 historical verdict is NO-GO and promotion is not authorized",
                "signed weekly coverage ended at 2026-08-02T15:00:00+00:00",
                "forced-exit-v2 is an offline execution overlay, not a production exchange adapter",
                "OCI credentials, balances, filters, ownership ledger, alerts, and rollback were not live-verified",
            ],
        "integrity_command": "python tools/verify_package.py .",
        "smoke_test_command": "python tools/smoke_test.py .",
            "strict_deployment_gate_command": "python tools/verify_package.py . --require-deployable",
        }
        write_text(staging / "release.json", json.dumps(release, ensure_ascii=False, indent=2) + "\n")
        write_text(staging / "requirements.txt", "\n".join(REQUIREMENTS) + "\n")
        installed = {}
        for item in REQUIREMENTS:
            name = item.split("==", 1)[0]
            try:
                installed[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                installed[name] = None
        environment = {"python": sys.version, "platform": sys.platform, "packages": installed}
        write_text(staging / "environment.json", json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
        write_text(staging / "README.md", README)
        write_text(staging / "DEPLOYMENT_RUNBOOK.md", RUNBOOK)
        write_text(staging / "run_replay.py", RUN_REPLAY_PY)
        write_text(staging / "run_replay.ps1", RUN_REPLAY_PS1)
        write_text(staging / "run_replay.sh", RUN_REPLAY_SH)

        files = sorted(path for path in staging.rglob("*") if path.is_file())
        lines = [f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}" for path in files]
        write_text(staging / "MANIFEST.sha256", "\n".join(lines) + "\n")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return target
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


README = """# ethbtc-forced-exit

这是 `v22-risk-off-forced-exit-v2` 的封存离线发布候选包，包含 BTC/ETH Grid 与 DCA
回放所需的源文件、冻结 v22 模型、冻结状态、四份实际使用的行情文件、证据产物和哈希清单。

状态：`NO-GO`、`offline_only=true`、`deployment_allowed=false`。本包可完整性校验和离线回放，
不能直接连接交易所或授予实盘权限。

校验：`python tools/verify_package.py .`；关键行为自测：`python tools/smoke_test.py .`

严格上线门检查（当前应失败）：`python tools/verify_package.py . --require-deployable`

重新回放（跨平台）：`python run_replay.py`。Linux 也可执行 `bash run_replay.sh`；允许脚本执行的
Windows 环境可执行 `./run_replay.ps1`。输出写入
`reproduced/`，不会覆盖 `evidence/` 中的封存结果。
"""

RUNBOOK = """# 上线准备流程

## 当前结论

本包仅达到“离线发布候选”阶段，尚不能上线。`--require-deployable` 必须保持失败，直到生成新的、
当前周有效且经人工批准的生产包，并完成真实交易执行适配。

## 阶段门

1. 完整性：校验 `MANIFEST.sha256`、模型、冻结状态和执行策略哈希。
2. 可复现性：在干净环境安装 `requirements.txt`，运行回放并核对目标退出/重入事件和消融指标。
3. 生产实现：把强制退出状态机接入现有 Grid/DCA Guard；按机器人账本卖出，动态读取交易所过滤器；
   保证幂等撤单、部分成交复核、独立通道接管、dust、告警和重启恢复。
4. 安全测试：覆盖 API 拒绝、撤单延迟、部分成交、网络中断、重复事件、余额越界、模型过期和哈希损坏。
5. OCI 观察模式：部署代码但禁止交易与自动重入，只记录拟执行动作；核对账户归属余额和交易所过滤器。
6. 小额退出演练：人工审批后，仅对隔离资金边界执行；确认不会卖出机器人外余额。
7. 原子授权：新模型合同健康、当前周签名有效、人工审批后再切换；任何失败均 Fail-Closed。
8. 回滚：撤销新单权限，取消机器人挂单，完成归属库存复核；回滚代码不能恢复旧模型自动放行。

## 上线前硬阻断

- v22 冻结包为 `NO-GO`，且周签名覆盖已结束。
- forced-exit-v2 当前是离线覆盖层，没有生产交易所执行适配。
- 尚未在 OCI 核验凭证、余额边界、动态过滤器、告警、持久化状态与恢复流程。
- 未获得人工实盘批准。
"""

RUN_REPLAY_PS1 = r"""$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $PackageRoot
try {
    python sources/build_v22_grid_dca_forced_exit_v2.py `
      --result-dir inputs/frozen_v22 `
      --grid-candles inputs/candles/grid `
      --dca-candles inputs/candles/dca `
      --output-dir reproduced
} finally { Pop-Location }
"""

RUN_REPLAY_PY = r'''#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


package = Path(__file__).resolve().parent
environment = os.environ.copy()
environment["PYTHONDONTWRITEBYTECODE"] = "1"
command = [
    sys.executable,
    str(package / "sources/build_v22_grid_dca_forced_exit_v2.py"),
    "--result-dir", str(package / "inputs/frozen_v22"),
    "--grid-candles", str(package / "inputs/candles/grid"),
    "--dca-candles", str(package / "inputs/candles/dca"),
    "--output-dir", str(package / "reproduced"),
]
raise SystemExit(subprocess.run(command, cwd=package, env=environment, check=False).returncode)
'''

RUN_REPLAY_SH = """#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
package_root="$(cd "$(dirname "$0")" && pwd)"
cd "$package_root"
python sources/build_v22_grid_dca_forced_exit_v2.py \\
  --result-dir inputs/frozen_v22 \\
  --grid-candles inputs/candles/grid \\
  --dca-candles inputs/candles/dca \\
  --output-dir reproduced
"""

SMOKE_TEST = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

import pandas as pd


def epoch(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def main() -> int:
    package = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    sys.path.insert(0, str(package / "sources"))
    from build_v22_grid_dca_forced_exit_v2 import PACKAGE_ID, POLICY, inventory_overlay

    assert PACKAGE_ID == "ethbtc-forced-exit" == POLICY["package_id"]
    frame = pd.DataFrame([
        {"timestamp": 1000, "open": 10_000.0, "close": 10_000.0},
        {"timestamp": 1300, "open": 9_900.0, "close": 9_900.0},
    ])
    inventory, actions = inventory_overlay(frame, pd.Series([False, False]), "BTC-USDT", True)
    assert inventory.phase.tolist() == ["EXITING", "COOLDOWN"]
    assert actions[["signal_ts", "execution_ts"]].iloc[0].tolist() == [1000, 1300]

    summary = json.loads((package / "evidence/summary.json").read_text(encoding="utf-8"))
    assert summary["package_id"] == PACKAGE_ID and summary["deployment_allowed"] is False
    evidence = pd.read_csv(package / "evidence/execution_actions.csv")
    targets = {
        ("grid", "BTC-FDUSD", epoch("2026-05-13T00:00:00Z")): epoch("2026-05-13T00:05:00Z"),
        ("grid", "ETH-FDUSD", epoch("2026-05-23T00:00:00Z")): epoch("2026-05-23T00:05:00Z"),
        ("dca", "BTC-USDT", epoch("2026-05-13T00:00:00Z")): epoch("2026-05-13T00:05:00Z"),
        ("dca", "ETH-USDT", epoch("2026-05-23T00:00:00Z")): epoch("2026-05-23T00:05:00Z"),
    }
    exits = evidence[evidence.action.eq("MARKET_EXIT")]
    for (strategy, pair, signal), execution in targets.items():
        row = exits[(exits.strategy == strategy) & (exits.pair == pair) & (exits.signal_ts == signal)]
        assert len(row) == 1 and int(row.iloc[0].execution_ts) == execution
    print(json.dumps({"package_id": PACKAGE_ID, "smoke_test": "PASS", "target_exits": len(targets)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--replace", action="store_true",
                        help="atomically replace the existing package after staging succeeds")
    args = parser.parse_args()
    print(build(args.target, replace=args.replace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
