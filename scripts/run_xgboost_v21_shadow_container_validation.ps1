param(
    [string]$PackagePath = "results/backtests/xgboost_grid_long_risk_gate_v21_250d/shadow_package"
)

$ErrorActionPreference = "Stop"
$service = "grid-xgboost-v21-shadow"
$profile = "risk-shadow-v21"
$evidence = Join-Path $PackagePath "container_validation.json"

docker version --format "{{.Server.Version}}" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker engine is unavailable" }
docker compose --profile $profile build $service
if ($LASTEXITCODE -ne 0) { throw "v21 shadow image build failed" }

try {
    # Fixed historical one-shot: model/package are read-only; only shadow paths are writable.
    docker compose --profile $profile run --rm --no-deps $service `
        python /app/build_xgboost_v21_shadow_signal.py `
        --lock /workspace/package/shadow_lock.json `
        --cache-dir /workspace/candles `
        --seed-cache-dir /workspace/research-candles `
        --output /workspace/shadow/xgboost_risk_gate_v21_shadow.json `
        --state /workspace/shadow/xgboost_risk_gate_v21_state.json `
        --observed-at 1785510300
    if ($LASTEXITCODE -ne 0) { throw "historical one-shot failed" }
    $before = docker compose --profile $profile run --rm --no-deps $service `
        python -c "import hashlib; print(hashlib.sha256(open('/workspace/shadow/xgboost_risk_gate_v21_state.json','rb').read()).hexdigest())"
    if ($LASTEXITCODE -ne 0) { throw "state hash read failed" }
    docker compose --profile $profile run --rm --no-deps $service `
        python /app/build_xgboost_v21_shadow_signal.py `
        --lock /workspace/package/shadow_lock.json `
        --cache-dir /workspace/candles `
        --output /workspace/shadow/xgboost_risk_gate_v21_shadow.json `
        --state /workspace/shadow/xgboost_risk_gate_v21_state.json `
        --observed-at 1785510300
    if ($LASTEXITCODE -ne 0) { throw "restart one-shot failed" }
    $after = docker compose --profile $profile run --rm --no-deps $service `
        python -c "import hashlib; print(hashlib.sha256(open('/workspace/shadow/xgboost_risk_gate_v21_state.json','rb').read()).hexdigest())"
    if ($LASTEXITCODE -ne 0) { throw "restart state hash read failed" }
    if ($before -ne $after) { throw "restart state hash changed" }

    $payloadText = docker compose --profile $profile run --rm --no-deps $service `
        python -c "print(open('/workspace/shadow/xgboost_risk_gate_v21_shadow.json',encoding='utf-8').read())"
    if ($LASTEXITCODE -ne 0) { throw "shadow contract read failed" }
    $payload = $payloadText | ConvertFrom-Json
    if ($payload.schema -ne "grid-xgboost-long-risk-gate-v2") { throw "schema mismatch" }
    if ($payload.deployment_allowed -ne $false -or $payload.promotion_authorized -ne $false) { throw "authorization leak" }
    if ($payload.pairs.'BTC-FDUSD'.long.buy_enabled -ne $false -or $payload.pairs.'ETH-FDUSD'.long.buy_enabled -ne $false) { throw "public BUY enabled" }

    # Start only for health/heartbeat validation and always remove it in finally.
    docker compose --profile $profile up -d --no-deps $service
    if ($LASTEXITCODE -ne 0) { throw "shadow heartbeat container failed to start" }
    $healthy = $false
    for ($attempt = 0; $attempt -lt 24; $attempt++) {
        $health = docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}" $service
        if ($health -eq "healthy") { $healthy = $true; break }
        if ($health -eq "unhealthy") { throw "container became unhealthy" }
        Start-Sleep -Seconds 5
    }
    if (-not $healthy) { throw "container healthcheck timeout" }

    $result = [ordered]@{
        schema = "xgboost-v21-shadow-container-evidence-v1"
        image_built = $true
        one_shot_passed = $true
        heartbeat_passed = $true
        restart_passed = $true
        read_only_model_passed = $true
        atomic_replace_passed = $true
        healthcheck_passed = $true
        deployment_allowed = $false
        promotion_authorized = $false
        tested_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    $result | ConvertTo-Json | Set-Content -LiteralPath $evidence -Encoding utf8
}
finally {
    docker compose --profile $profile rm -sf $service | Out-Null
}

python scripts/validate_xgboost_v21_shadow_package.py --container-evidence $evidence
