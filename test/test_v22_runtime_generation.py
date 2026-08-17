import json
from pathlib import Path

from scripts.ethbtc_forced_exit_contract import atomic_json, sha256_file
from scripts.grid_v22_live_gate import V22LiveGateProducer, _runtime_deployment


def pair(signal_ts=100):
    return {
        "signal_ts": signal_ts, "risk_off_active": False, "model_week": 7,
        "buy_enabled": True, "force_exit": False,
        "week_model_sha256": "f" * 64,
    }


def test_prepared_generation_is_invisible_until_single_pointer_commit(tmp_path, monkeypatch):
    family = tmp_path / "family"
    release_sha = "a" * 64
    release = family / "releases" / release_sha
    (release / "shadow_package").mkdir(parents=True)
    (release / "shadow_package" / "shadow_lock.json").write_text("{}", encoding="utf-8")
    (release / "production_lock.json").write_text(json.dumps({
        "release_sha256": release_sha, "model_sha256": "b" * 64,
    }), encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "xgboost_risk_gate_v22_state.json").write_text(
        json.dumps({"seed": True}), encoding="utf-8",
    )
    live = state_dir / "ethbtc_forced_exit_observation.json"
    live.write_text(json.dumps({
        "source_healthy": True, "release_sha256": "c" * 64,
        "pairs": {"BTC-FDUSD": pair(), "ETH-FDUSD": pair()},
    }), encoding="utf-8")
    receipt = {
        "schema": "ethbtc-forced-exit-authorization-v1",
        "package_id": "ethbtc-forced-exit", "release_sha256": release_sha,
        "model_sha256": "b" * 64,
    }
    producer = V22LiveGateProducer(
        package_dir=family, cache_dir=tmp_path / "candles",
        seed_cache_dir=tmp_path / "candles", state_dir=state_dir,
        authorization_path=tmp_path / "auth.json", refresh_binance=False,
    )

    def fake_produce_for(**kwargs):
        kwargs["shadow_state"].write_text(json.dumps({"state": "prepared"}), encoding="utf-8")
        kwargs["shadow_output"].write_text(json.dumps({"shadow": True}), encoding="utf-8")
        return {
            "source_healthy": True,
            "pairs": {"BTC-FDUSD": pair(), "ETH-FDUSD": pair()},
        }

    monkeypatch.setattr(producer, "_produce_for", fake_produce_for)
    prepared = producer.prepare_generation(
        release=release, authorization=receipt,
        predecessor_release_sha256="c" * 64, fold_boundary=200,
        observed_at=100, live_contract_path=live,
    )
    assert not (producer.runtime_root / "current.json").exists()
    producer.commit_generation(prepared["pointer"])
    resolved = _runtime_deployment(family, producer.runtime_root)
    assert resolved is not None
    assert resolved[0] == release
    assert resolved[2]["runtime_generation"] == prepared["generation"]
    assert sha256_file(
        producer.runtime_root / "generations" / prepared["generation"] / "manifest.json"
    ) == prepared["generation"]


def test_prewarm_rejects_unhealthy_or_wrong_predecessor_without_pointer_change(
        tmp_path, monkeypatch):
    family = tmp_path / "family"
    release_sha = "a" * 64
    release = family / "releases" / release_sha
    (release / "shadow_package").mkdir(parents=True)
    (release / "shadow_package" / "shadow_lock.json").write_text("{}", encoding="utf-8")
    (release / "production_lock.json").write_text(json.dumps({
        "release_sha256": release_sha, "model_sha256": "b" * 64,
    }), encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    live = state_dir / "ethbtc_forced_exit_observation.json"
    live.write_text(json.dumps({
        "source_healthy": False, "release_sha256": "c" * 64,
        "pairs": {"BTC-FDUSD": pair(), "ETH-FDUSD": pair()},
    }), encoding="utf-8")
    producer = V22LiveGateProducer(
        package_dir=family, cache_dir=tmp_path / "candles",
        seed_cache_dir=tmp_path / "candles", state_dir=state_dir,
        authorization_path=tmp_path / "auth.json", refresh_binance=False,
    )

    def fake_produce_for(**kwargs):
        kwargs["shadow_state"].write_text("{}", encoding="utf-8")
        kwargs["shadow_output"].write_text("{}", encoding="utf-8")
        return {"source_healthy": True,
                "pairs": {"BTC-FDUSD": pair(), "ETH-FDUSD": pair()}}

    monkeypatch.setattr(producer, "_produce_for", fake_produce_for)
    receipt = {"release_sha256": release_sha, "model_sha256": "b" * 64}
    try:
        producer.prepare_generation(
            release=release, authorization=receipt,
            predecessor_release_sha256="c" * 64, fold_boundary=200,
            observed_at=100, live_contract_path=live,
        )
    except RuntimeError as exc:
        assert "healthy current live contract" in str(exc)
    else:
        raise AssertionError("unhealthy live contract must reject candidate prewarm")
    assert not (producer.runtime_root / "current.json").exists()


def test_pointer_cannot_reference_partial_generation(tmp_path):
    runtime = tmp_path / "runtime"
    atomic_json(runtime / "current.json", {
        "schema": "ethbtc-forced-exit-runtime-pointer-v1",
        "runtime_generation": "a" * 64,
        "generation_manifest_sha256": "a" * 64,
    })
    family = tmp_path / "family"
    try:
        _runtime_deployment(family, runtime)
    except ValueError as exc:
        assert "manifest hash" in str(exc)
    else:
        raise AssertionError("partial generation must be rejected")
