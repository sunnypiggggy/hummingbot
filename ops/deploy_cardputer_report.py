"""Build and update only OCI dca-live-report; retain the exact previous image.

Run with the deployment Python environment (Paramiko and Paho 2.1 installed).
The build context is an explicit source allowlist. Credentials are transferred
separately into a private, read-only Docker secret and never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import sys
import tempfile
import time

import paramiko


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "Dockerfile.dca-live-report-mqtt", "ops/compose.cardputer-report-mqtt.yml",
    "live_guard/dca_live_report.py", "live_guard/trading_status.py", "live_guard/risk_history.py",
    "management_bot/__init__.py", "management_bot/clients.py", "management_bot/risk_display.py",
    "cardputer_monitor/__init__.py", "cardputer_monitor/collector.py",
    "cardputer_monitor/history.py", "cardputer_monitor/mqtt_reporter.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def connect(alias):
    config = paramiko.SSHConfig()
    with (Path.home() / ".ssh/config").open() as handle:
        config.parse(handle)
    spec = config.lookup(alias)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(str(Path.home() / ".ssh/known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(spec["hostname"], port=int(spec.get("port", 22)),
                   username=spec.get("user", "ubuntu"),
                   key_filename=spec.get("identityfile"), timeout=15,
                   auth_timeout=15, banner_timeout=15)
    client.get_transport().set_keepalive(20)
    return client


def run(client, command, *, data=None, timeout=180):
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    try:
        if data is not None:
            stdin.write(data)
            stdin.flush()
            stdin.channel.shutdown_write()
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        status = stdout.channel.recv_exit_status()
    finally:
        stdin.close()
        stdout.close()
        stderr.close()
        stdout.channel.close()
    if status:
        # Never include a remote command, environment, config or raw log in an error.
        raise RuntimeError(f"Remote operation failed ({status}); {err[-400:]}")
    return out


def python(client, source):
    return json.loads(run(client, "python3 -", data=source))


def current(client, baseline):
    names = [item["name"] for item in baseline["containers"]]
    values = json.loads(run(client, "docker inspect " + " ".join(map(shlex.quote, names))))
    return [{"name": item["Name"].lstrip("/"), "id": item["Id"],
             "image_id": item["Image"], "image": item["Config"]["Image"],
             "started": item["State"]["StartedAt"],
             "health": item["State"].get("Health", {}).get("Status"),
             "restart_count": item["RestartCount"]} for item in values]


def unchanged(before, after, *, include_report=False):
    old = {v["name"]: v for v in before}
    for value in after:
        if value["name"] == "dca-live-report" and not include_report:
            continue
        for field in ("id", "image_id", "started", "restart_count"):
            if value[field] != old[value["name"]][field]:
                raise RuntimeError(f"Concurrent runtime change: {value['name']} {field}")


def source_checks(client, baseline):
    paths = {name: item["sha256"] for name, item in baseline["source_files"].items()}
    code = "import hashlib,json,pathlib\nroot=pathlib.Path(" + repr(baseline["root"]) + ")\n"
    code += "paths=" + repr(paths) + "\n"
    code += "print(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}))\n"
    actual = python(client, code)
    if actual != paths:
        raise RuntimeError("OCI source changed since baseline; audit before deployment")


def upload(client, local, remote, *, mode=0o600):
    # Use a single exec channel rather than repeated SFTP negotiation. Binary
    # data stays inside the authenticated encrypted stdin, including secrets.
    program = ("import os,pathlib,sys;os.umask(0o077);p=pathlib.Path(" + repr(remote)
               + ");p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.upload');"
               + "t.write_bytes(sys.stdin.buffer.read());os.chmod(t," + str(mode) + ");os.replace(t,p)")
    run(client, "python3 -c " + shlex.quote(program), data=Path(local).read_bytes())


def rollback_compose(baseline, manifest):
    previous = baseline.get("rollback_compose_command")
    if previous:
        # Restore the previous runtime configuration as well as its image. Pin
        # to the saved image tag so a rebuilt version tag cannot change rollback.
        if not previous.startswith("CARDPUTER_REPORT_IMAGE="):
            raise RuntimeError("Unsupported previous Compose launch command")
        _image_assignment, command = previous.split(" ", 1)
        return "CARDPUTER_REPORT_IMAGE=" + shlex.quote(manifest["base_tag"]) + " " + command
    return ("docker compose --project-directory " + shlex.quote(baseline["root"])
            + " -p hummingbot -f " + shlex.quote(baseline["compose_path"])
            + " -f " + shlex.quote(manifest["remote"] + "/compose.rollback.yml"))


def prepare(client, args, baseline):
    source_checks(client, baseline)
    unchanged(baseline["containers"], current(client, baseline), include_report=True)
    artifact = args.cloud_dir / "artifacts" / ("oci-report-mqtt-" + args.release)
    artifact.mkdir(parents=True, exist_ok=True)
    context = artifact / "context"
    context.mkdir(exist_ok=True)
    hashes = {}
    for name in SOURCE_FILES:
        src = ROOT / name
        dst = context / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        hashes[name] = digest(dst)
    # A pure Python universal wheel can be installed offline on OCI arm64.
    import subprocess
    wheels = context / "cardputer_monitor/wheels"
    wheels.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                    "--dest", str(wheels), "paho-mqtt==2.1.0"], check=True)
    remote = baseline["root"] + "/.cardputer-report-rollouts/" + args.release
    old = next(v for v in baseline["containers"] if v["name"] == "dca-live-report")
    base_tag = "hummingbot/dca-live-report:pre-cardputer-" + args.release
    image = "hummingbot/dca-live-report:cardputer-mqtt-" + args.release
    run(client, "umask 077; mkdir -p " + shlex.quote(remote) + "; chmod 700 " + shlex.quote(remote))
    run(client, "cp -p " + shlex.quote(baseline["compose_path"]) + " " + shlex.quote(remote + "/compose.before.yml")
        + " && cp -p " + shlex.quote(baseline["root"] + "/live_guard/dca_live_report.py") + " "
        + shlex.quote(remote + "/dca_live_report.before.py"))
    for name in SOURCE_FILES:
        upload(client, context / name, remote + "/context/" + name)
    for path in wheels.glob("*.whl"):
        upload(client, path, remote + "/context/cardputer_monitor/wheels/" + path.name)
    credentials = args.cloud_dir / "secrets/cardputer-report-mqtt.json"
    value = json.loads(credentials.read_text(encoding="utf-8"))
    if value.get("username") != "cardputer-report" or not value.get("password"):
        raise RuntimeError("Publisher credential file is invalid")
    credential_path = remote + "/cardputer-report-mqtt.json"
    ca_path = remote + "/isrg-root-x1.pem"
    upload(client, credentials, credential_path)
    upload(client, args.cloud_dir / "client-ca/isrg-root-x1.pem", ca_path, mode=0o644)
    overlay = remote + "/compose.cardputer-report-mqtt.yml"
    upload(client, ROOT / "ops/compose.cardputer-report-mqtt.yml", overlay)
    prefix = ("CARDPUTER_REPORT_IMAGE=" + shlex.quote(image)
              + " CARDPUTER_REPORT_CA_PATH=" + shlex.quote(ca_path)
              + " CARDPUTER_REPORT_CREDENTIALS_PATH=" + shlex.quote(credential_path) + " ")
    compose = ("docker compose --project-directory " + shlex.quote(baseline["root"])
               + " -p " + shlex.quote(baseline["project"]) + " -f " + shlex.quote(baseline["compose_path"])
               + " -f " + shlex.quote(overlay))
    rollback_overlay = remote + "/compose.rollback.yml"
    rollback_text = "services:\n  dca-live-report:\n    image: " + base_tag + "\n"
    release_info = {"base_tag": base_tag, "remote": remote}
    rollback_command = rollback_compose(baseline, release_info)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "rollback.yml"
        path.write_text(rollback_text, encoding="utf-8", newline="\n")
        upload(client, path, rollback_overlay)
        for filename, command in (
            ("deploy.sh", prefix + compose + " up -d --no-deps --force-recreate --no-build dca-live-report"),
            ("rollback.sh", rollback_command + " up -d --no-deps --force-recreate --no-build dca-live-report"),
        ):
            path = Path(temporary) / filename
            path.write_text("#!/bin/sh\nset -eu\n" + command + "\n", encoding="utf-8", newline="\n")
            upload(client, path, remote + "/" + filename, mode=0o700)
    manifest = {"release": args.release, "remote": remote, "image": image, "base_tag": base_tag,
                "old_image_id": old["image_id"], "source_sha256": hashes,
                "wheel_sha256": {path.name: digest(path) for path in wheels.glob("*.whl")},
                "rollback_compose_command": rollback_command,
                "compose_command": prefix + compose, "baseline": str(args.baseline)}
    (artifact / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": True, "release": args.release, "source_files": len(hashes)}))
    return manifest


def build(client, manifest):
    run(client, "docker image tag " + shlex.quote(manifest["old_image_id"]) + " " + shlex.quote(manifest["base_tag"]))
    output = run(client, "docker build --network=none --build-arg REPORT_BASE_IMAGE="
                 + shlex.quote(manifest["base_tag"]) + " -t " + shlex.quote(manifest["image"])
                 + " -f " + shlex.quote(manifest["remote"] + "/context/Dockerfile.dca-live-report-mqtt")
                 + " " + shlex.quote(manifest["remote"] + "/context"), timeout=300)
    # Import the real flat report and all namespaced telemetry modules without any
    # network, runtime data, credentials, or report cycle.
    program = ("import dca_live_report,cardputer_monitor.mqtt_reporter,cardputer_monitor.history;"
               "import paho.mqtt;assert paho.mqtt.__version__=='2.1.0';print('imports-ok')")
    check = run(client, "docker run --rm --network none --entrypoint python "
                + shlex.quote(manifest["image"]) + " -c " + shlex.quote(program))
    image_id = run(client, "docker image inspect --format '{{.Id}}' " + shlex.quote(manifest["image"])).strip()
    manifest["image_id"] = image_id
    print(json.dumps({"built": image_id, "import_check": check.strip()}))


def deploy(client, baseline, manifest):
    source_checks(client, baseline)
    unchanged(baseline["containers"], current(client, baseline), include_report=True)
    # Render privately; print only safe facts. Assert original mounts, command,
    # network, notification secret, and other service definitions are preserved.
    base = json.loads(run(client, "docker compose --project-directory " + shlex.quote(baseline["root"])
                          + " -p hummingbot -f " + shlex.quote(baseline["compose_path"]) + " --profile '*' config --format json"))
    merged = json.loads(run(client, manifest["compose_command"] + " --profile '*' config --format json"))
    for name, service in base["services"].items():
        if name != "dca-live-report" and service != merged["services"][name]:
            raise RuntimeError("Overlay unexpectedly changes " + name)
    old, new = base["services"]["dca-live-report"], merged["services"]["dca-live-report"]
    for field in ("command", "healthcheck", "networks", "depends_on", "restart", "container_name"):
        if old.get(field) != new.get(field):
            raise RuntimeError("Report runtime mismatch: " + field)
    for field in ("volumes", "secrets"):
        if any(v not in new.get(field, []) for v in old.get(field, [])):
            raise RuntimeError("Report original mounts/secrets missing")
    if new.get("image") != manifest["image"]:
        raise RuntimeError("Unverified image in Compose")
    try:
        run(client, "sh " + shlex.quote(manifest["remote"] + "/deploy.sh"))
        deadline = time.monotonic() + 100
        while time.monotonic() < deadline:
            after = current(client, baseline)
            unchanged(baseline["containers"], after)
            target = next(v for v in after if v["name"] == "dca-live-report")
            if target["health"] == "healthy" and target["image_id"] == manifest["image_id"]:
                manifest["containers_after"] = after
                manifest["deployed_at"] = time.time()
                print(json.dumps({"deployed": True, "container": target,
                                  "other_containers_unchanged": len(after) - 1}))
                return
            if target["health"] == "unhealthy":
                raise RuntimeError("Updated report failed healthcheck")
            time.sleep(5)
        raise RuntimeError("Report readiness timed out")
    except Exception:
        run(client, "sh " + shlex.quote(manifest["remote"] + "/rollback.sh"))
        raise


def verify_mqtt(args):
    import ssl
    import uuid
    import paho.mqtt.client as mqtt
    sys.path.insert(0, str(args.cloud_dir))
    from cardputer.telemetry import validate_trading
    from cardputer import trading_history
    credentials = json.loads((args.cloud_dir / "secrets/cardputer-mqtt.json").read_text())
    messages = queue.Queue()
    subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                             client_id="report-verification-" + uuid.uuid4().hex,
                             protocol=mqtt.MQTTv311)
    subscriber.username_pw_set(credentials["username"], credentials["password"])
    subscriber.tls_set_context(ssl.create_default_context(cafile=str(args.cloud_dir / "client-ca/isrg-root-x1.pem")))
    root = "cardputer/v1/trading/hummingbot-main/"
    def connected(client, _data, _flags, reason, _props):
        if reason.is_failure:
            messages.put(("error", b"authentication-failed", False))
        else:
            client.subscribe([(root + "snapshot", 1), (root + "availability", 1)])
    subscriber.on_connect = connected
    subscriber.on_message = lambda c, d, msg: messages.put((msg.topic, msg.payload, msg.retain))
    subscriber.connect("sh.sunnypiggy.top", 8883, 30)
    subscriber.loop_start()
    samples, availability = {}, []
    deadline = time.monotonic() + 150
    try:
        while time.monotonic() < deadline:
            try:
                topic, payload, retained = messages.get(timeout=min(10, max(0.1, deadline-time.monotonic())))
            except queue.Empty:
                continue
            if topic == "error":
                raise RuntimeError("MQTT subscription authentication failed")
            if topic.endswith("availability"):
                state = payload.decode("ascii")
                if state not in ("online", "offline"):
                    raise RuntimeError("Invalid publisher availability")
                availability.append(state)
            else:
                value = json.loads(payload)
                if len(payload) > 65536:
                    raise RuntimeError("Oversize report snapshot")
                clean = validate_trading(value)
                # Validate independently so unavailable fallback cannot hide a
                # malformed history. No production payload is written locally.
                for row in value["robots"]:
                    trading_history.validate(row["history"], value["collected_at"], time.time())
                samples[value["sample_id"]] = {
                    "sample_id": value["sample_id"], "collected_at": value["collected_at"],
                    "bytes": len(payload), "retained": retained,
                    "robots": [{k: row[k] for k in ("id", "quote_asset", "status", "status_data_state", "profit_data_state",
                                                   "status_observed_at", "profit_observed_at")}
                               | {"profit_points": len(row["history"]["profit_points"]),
                                  "profit_non_null": sum(p["value"] is not None for p in row["history"]["profit_points"]),
                                  "price_points": len(row["history"]["price_points"]),
                                  "price_non_null": sum(p["value"] is not None for p in row["history"]["price_points"]),
                                  "price_observed_at": row["history"]["price_observed_at"],
                                  "history_data_state": row["history"]["data_state"],
                                  "profit_windows_available": {k: v is not None for k, v in row["profit"].items()}}
                               for row in clean["robots"]],
                }
            if len(samples) >= 2 and availability and availability[-1] == "online":
                break
    finally:
        subscriber.disconnect()
        subscriber.loop_stop()
    if len(samples) < 2 or not availability or availability[-1] != "online":
        raise RuntimeError("Two consecutive MQTT snapshots/availability were not observed")
    records = sorted(samples.values(), key=lambda v: v["collected_at"])
    if records[-1]["collected_at"] <= records[0]["collected_at"]:
        raise RuntimeError("MQTT snapshots are not advancing")
    result = {"verified_at": time.time(), "samples": records, "availability": availability[-1]}
    encoded = json.dumps(result, indent=2) + "\n"
    (args.cloud_dir / "artifacts/oci-report-mqtt-verification.json").write_text(encoded)
    release_artifact = args.cloud_dir / "artifacts" / ("oci-report-mqtt-" + args.release)
    if release_artifact.is_dir():
        (release_artifact / "mqtt-verification.json").write_text(encoded)
    print(json.dumps(result, ensure_ascii=False))
    return result


def verify_runtime(client, baseline, manifest, artifact):
    after = current(client, baseline)
    unchanged(baseline["containers"], after)
    target = next(item for item in after if item["name"] == "dca-live-report")
    if target["image_id"] != manifest["image_id"] or target["health"] != "healthy":
        raise RuntimeError("Loaded report version/health is not the verified image")
    inspected = json.loads(run(client, "docker inspect dca-live-report"))[0]
    mounts = {item["Destination"]: item for item in inspected["Mounts"]}
    for item in baseline["target_runtime"]["mounts"]:
        actual = mounts.get(item["dst"])
        if not actual or actual["Source"] != item["src"] or actual["RW"] != item["rw"]:
            raise RuntimeError("Loaded report original mount changed")
    for destination in ("/run/secrets/cardputer_report_mqtt", "/etc/cardputer/isrg-root-x1.pem"):
        if destination not in mounts or mounts[destination]["RW"]:
            raise RuntimeError("Loaded report credential/CA is not read-only")
    if inspected["HostConfig"].get("PortBindings") or sorted(inspected["NetworkSettings"]["Networks"]) != sorted(baseline["target_runtime"]["network_names"]):
        raise RuntimeError("Loaded report ports/network changed")
    files = {name: sha for name, sha in manifest["source_sha256"].items()
             if name.endswith(".py")}
    names = {name: ("/app/dca_live_report.py" if name == "live_guard/dca_live_report.py" else "/app/" + name)
             for name in files}
    program = ("import hashlib,json,pathlib;paths=" + repr(names)
               + ";print(json.dumps({p:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() for p,f in paths.items()}))")
    loaded = json.loads(run(client, "docker exec dca-live-report python -c " + shlex.quote(program)))
    if loaded != files:
        raise RuntimeError("Loaded report modules do not match the tested source manifest")
    preserved = {name: item["sha256"] for name, item in baseline["source_files"].items()
                 if name in ("live_guard/model_probability_history.py", "live_guard/telegram_notifications.py",
                             "live_guard/risk_history.py")}
    program = ("import hashlib,json,pathlib;paths=" + repr(list(preserved))
               + ";print(json.dumps({p:hashlib.sha256((pathlib.Path('/app')/pathlib.Path(p).name).read_bytes()).hexdigest() for p in paths}))")
    actual = json.loads(run(client, "docker exec dca-live-report python -c " + shlex.quote(program)))
    if actual != preserved:
        raise RuntimeError("Report's existing risk/notification/probability code changed")
    logs = run(client, "docker logs --tail 30 dca-live-report")
    cycles = []
    for line in logs.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and "cardputer_mqtt" in value:
            cycles.append({"generated_at": value.get("generated_at"),
                           "cardputer_mqtt": value["cardputer_mqtt"],
                           "telegram": {k: value.get("telegram", {}).get(k)
                                        for k in ("pending", "retrying", "profit_report_error")}})
    if not cycles:
        raise RuntimeError("No completed report cycle with MQTT diagnostics")
    result = {"verified_at": time.time(), "container": target,
              "other_containers_unchanged": len(after) - 1, "containers": after,
              "credentials_and_ca_readonly": True, "original_mounts_preserved": True,
              "ports_and_network_preserved": True,
              "loaded_source_sha256": loaded, "preserved_source_sha256": actual,
              "recent_report_cycles": cycles[-2:]}
    (artifact / "runtime-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"runtime_verified": True, "container": target,
                      "other_containers_unchanged": len(after) - 1, "recent_report_cycles": cycles[-2:]}))


def repair_rollback(client, baseline, manifest):
    target = next(item for item in current(client, baseline) if item["name"] == "dca-live-report")
    if target["image_id"] != manifest["image_id"]:
        raise RuntimeError("Unexpected current report image; not editing rollback")
    pinned = run(client, "docker image inspect --format '{{.Id}}' " + shlex.quote(manifest["base_tag"])).strip()
    if pinned != manifest["old_image_id"]:
        raise RuntimeError("Rollback pinned image mismatch")
    command = rollback_compose(baseline, manifest)
    rendered = json.loads(run(client, command + " --profile '*' config --format json"))
    live_config = json.loads(run(client, manifest["compose_command"] + " --profile '*' config --format json"))
    for name in live_config["services"]:
        if name != "dca-live-report" and rendered["services"][name] != live_config["services"][name]:
            raise RuntimeError("Rollback changes another service")
    old = rendered["services"]["dca-live-report"]
    live = live_config["services"]["dca-live-report"]
    for field in ("command", "healthcheck", "networks", "restart", "container_name"):
        if old.get(field) != live.get(field):
            raise RuntimeError("Rollback runtime differs unexpectedly")
    if baseline.get("rollback_compose_command"):
        if str(old["environment"].get("CARDPUTER_MQTT_ENABLED")).lower() != "true":
            raise RuntimeError("Rollback unexpectedly disables prior MQTT publisher")
        for key in ("CARDPUTER_MQTT_HOST", "CARDPUTER_MQTT_PORT", "CARDPUTER_PRICE_PROVIDER"):
            if old["environment"].get(key) != live["environment"].get(key):
                raise RuntimeError("Rollback publisher configuration differs")
    run(client, "cp -p " + shlex.quote(manifest["remote"] + "/rollback.sh")
        + " " + shlex.quote(manifest["remote"] + "/rollback.before-config-preservation.sh"))
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "rollback.sh"
        path.write_text("#!/bin/sh\nset -eu\n" + command
                        + " up -d --no-deps --force-recreate --no-build dca-live-report\n",
                        encoding="utf-8", newline="\n")
        upload(client, path, manifest["remote"] + "/rollback.sh", mode=0o700)
    manifest["rollback_compose_command"] = command
    manifest["rollback_config_verified_at"] = time.time()
    print(json.dumps({"rollback_config_verified": True,
                      "preserves_prior_mqtt": bool(baseline.get("rollback_compose_command")),
                      "pinned_image_id": pinned, "current_container_id_unchanged": target["id"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare-build", "deploy", "verify", "verify-runtime", "repair-rollback"))
    parser.add_argument("--cloud-dir", type=Path, default=ROOT.parent / "serverdocker/cloud")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--release", default="20261005-mqtt-v1")
    parser.add_argument("--host", default="oci-ubuntu-152-69-205-54")
    args = parser.parse_args()
    args.baseline = args.baseline or args.cloud_dir / "artifacts/oci-report-before.json"
    if args.action == "verify":
        verify_mqtt(args)
        return
    baseline = json.loads(args.baseline.read_text())
    artifact = args.cloud_dir / "artifacts" / ("oci-report-mqtt-" + args.release)
    client = connect(args.host)
    try:
        if args.action == "prepare-build":
            manifest = prepare(client, args, baseline)
            build(client, manifest)
        else:
            manifest = json.loads((artifact / "manifest.json").read_text())
            if args.action == "verify-runtime":
                verify_runtime(client, baseline, manifest, artifact)
            elif args.action == "repair-rollback":
                repair_rollback(client, baseline, manifest)
            else:
                deploy(client, baseline, manifest)
        (artifact / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        client.close()


if __name__ == "__main__":
    main()
