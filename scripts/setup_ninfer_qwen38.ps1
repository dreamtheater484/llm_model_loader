param(
    [string]$Distro = "",
    [switch]$Start,
    [switch]$SkipLanSetup
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$SetupPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $SetupPython)) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $SetupPython = $pythonCommand.Source }
}
if (-not (Test-Path -LiteralPath $SetupPython)) {
    throw "Python is required. Run setup.cmd first."
}

$AppDataDir = if ($env:LLM_MODEL_LOADER_DATA_DIR) {
    $env:LLM_MODEL_LOADER_DATA_DIR
} else {
    Join-Path $env:LOCALAPPDATA "llm-model-loader"
}

# ---------- WSL distro discovery ----------
$rawDistros = & wsl.exe -l -q
if ($LASTEXITCODE -ne 0) {
    throw "WSL is not available. Install/enable WSL2 first."
}
$distros = @(
    $rawDistros |
        ForEach-Object { ($_ -replace "`0", '').Trim() } |
        Where-Object { $_ }
)
if ([string]::IsNullOrWhiteSpace($Distro)) {
    $ubuntu2404 = @()
    foreach ($d in $distros) {
        $osRelease = (& wsl.exe -d $d -- cat /etc/os-release 2>$null) -join "`n"
        if ($LASTEXITCODE -eq 0 -and
            $osRelease -match '(?m)^ID=ubuntu$' -and
            $osRelease -match '(?m)^VERSION_ID="?24\.04"?$') {
            $ubuntu2404 += $d
        }
    }
    if ($ubuntu2404.Count -eq 0) {
        Write-Host ""
        Write-Host "No Ubuntu 24.04 WSL distro was found." -ForegroundColor Yellow
        Write-Host "Install it once with:" -ForegroundColor Yellow
        Write-Host "  wsl.exe --install -d Ubuntu-24.04" -ForegroundColor White
        Write-Host "Complete Ubuntu's first-launch username/password setup, then run this script again."
        throw "Ubuntu 24.04 is required because NInfer's current Linux dependencies line up cleanly with it."
    }
    $Distro = if ($ubuntu2404 -contains 'Ubuntu-24.04') { 'Ubuntu-24.04' } else { $ubuntu2404[0] }
} elseif ($distros -notcontains $Distro) {
    throw "The requested distro '$Distro' is not installed. Installed: $($distros -join ', ')"
}
Write-Host "Using WSL distro: $Distro" -ForegroundColor Green

# ---------- Literal Bash payload (no PowerShell interpolation) ----------
$BashScript = @'
#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo >&2; echo "ERROR at Bash line $LINENO: $BASH_COMMAND (exit $rc)" >&2; exit "$rc"' ERR

ROOT="${NINFER_ROOT:-$HOME/ninfer-qwen38}"
SRC="$ROOT/ninfer"
BUILD="$SRC/build-cuda131"
MODEL_DIR="$ROOT/models"
MODEL="$MODEL_DIR/qwen3_8_27b_nvfp4.ninfer"
HF_VENV="$ROOT/.hf-venv"
LOG_DIR="$ROOT/logs"
LAUNCHER="$ROOT/run-qwen38-nvfp4.sh"
FACTS="$ROOT/wsl-info.json"
MIN_RUNTIME_REV="5d2c1f5590b8f4c3d106a75f65210eb4efb8f4e1"
EXPECTED_SHA="bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32"
CUDA_HOME="/usr/local/cuda-13.1"

export PATH="/usr/lib/wsl/lib:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

version_ge() {
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

echo
echo "============================================================"
echo " NInfer + Qwen3.8-27B NVFP4 + RTX 5090 / WSL2"
echo "============================================================"
echo

# ---------- OS / GPU preflight ----------
. /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
    echo "ERROR: this bootstrap is intentionally targeted at Ubuntu 24.04 under WSL2." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is not visible inside WSL." >&2
    echo "Update the NVIDIA *Windows* driver and WSL. Do NOT install a Linux NVIDIA display driver inside WSL." >&2
    exit 1
fi

echo "GPU visible in WSL:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
if [[ "$GPU_NAME" != *"RTX 5090"* ]]; then
    echo "ERROR: current NInfer is specialized for the RTX 5090 (sm_120a). Detected: $GPU_NAME" >&2
    exit 1
fi

DRIVER_CUDA="$(nvidia-smi | sed -nE 's/.*CUDA Version: *([0-9.]+).*/\1/p' | head -n1)"
if [[ -n "$DRIVER_CUDA" ]] && ! version_ge "$DRIVER_CUDA" "13.1"; then
    echo "ERROR: the Windows driver exposed to WSL reports CUDA $DRIVER_CUDA; NInfer requires driver support for CUDA 13.1+." >&2
    echo "Update the Windows NVIDIA driver, then rerun this script." >&2
    exit 1
fi

echo
echo "Requesting sudo once for Ubuntu packages..."
sudo -v

# ---------- Base build dependencies ----------
echo
echo "Installing/updating build dependencies..."
sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libcurl4-openssl-dev \
    libswscale-dev \
    ninja-build \
    pkg-config \
    python3 \
    python3-venv

# ---------- CUDA 13.1 Toolkit (toolkit only; never install Linux driver) ----------
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo
    echo "Installing CUDA Toolkit 13.1 from NVIDIA's WSL-Ubuntu repository..."
    if ! apt-cache show cuda-toolkit-13-1 >/dev/null 2>&1; then
        KEYRING_DEB="/tmp/cuda-keyring_1.1-1_all.deb"
        curl -fL \
            "https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb" \
            -o "$KEYRING_DEB"
        sudo dpkg -i "$KEYRING_DEB"
        rm -f "$KEYRING_DEB"
        sudo apt-get update
    fi
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-13-1
fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "ERROR: CUDA 13.1 nvcc was not found at $CUDA_HOME/bin/nvcc after installation." >&2
    exit 1
fi

echo
echo "CUDA compiler:"
"$CUDA_HOME/bin/nvcc" --version | tail -n 1
NVCC_RELEASE="$("$CUDA_HOME/bin/nvcc" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | tail -n1)"
if [[ "$NVCC_RELEASE" != "13.1" ]]; then
    echo "ERROR: expected the pinned CUDA 13.1 compiler, got: ${NVCC_RELEASE:-unknown}" >&2
    exit 1
fi

for spec in \
    'libavformat:60' \
    'libavcodec:60' \
    'libavutil:58' \
    'libswscale:7' \
    'libcurl:7.85'; do
    pkg="${spec%%:*}"
    min="${spec#*:}"
    got="$(pkg-config --modversion "$pkg")"
    if ! version_ge "$got" "$min"; then
        echo "ERROR: $pkg $got is too old; NInfer requires >= $min." >&2
        exit 1
    fi
done

# ---------- Current NInfer source ----------
mkdir -p "$ROOT" "$MODEL_DIR" "$LOG_DIR"

echo
echo "Fetching current NInfer master..."
if [[ ! -d "$SRC/.git" ]]; then
    git clone https://github.com/Neroued/ninfer.git "$SRC"
else
    if [[ -n "$(git -C "$SRC" status --porcelain)" ]]; then
        echo "ERROR: $SRC has local changes. I will not overwrite them." >&2
        echo "Move/commit those changes or remove $SRC, then rerun." >&2
        exit 1
    fi
    git -C "$SRC" fetch --prune origin master
    git -C "$SRC" checkout master
    git -C "$SRC" merge --ff-only origin/master
fi

CURRENT_REV="$(git -C "$SRC" rev-parse HEAD)"
if ! git -C "$SRC" merge-base --is-ancestor "$MIN_RUNTIME_REV" "$CURRENT_REV"; then
    echo "ERROR: checked-out NInfer ($CURRENT_REV) predates the Qwen3.8 NVFP4 minimum runtime revision." >&2
    exit 1
fi
echo "NInfer revision: $CURRENT_REV"

# ---------- Build specifically with CUDA 13.1 ----------
echo
echo "Configuring and building NInfer..."
cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
cmake --build "$BUILD" --parallel "$(nproc)"

SERVER="$BUILD/apps/ninfer-serve"
if [[ ! -x "$SERVER" ]]; then
    echo "ERROR: NInfer server binary was not produced at $SERVER" >&2
    exit 1
fi

# ---------- Hugging Face CLI in its own venv ----------
echo
echo "Preparing Hugging Face downloader..."
if [[ ! -x "$HF_VENV/bin/python" ]]; then
    python3 -m venv "$HF_VENV"
fi
"$HF_VENV/bin/python" -m pip install --upgrade pip huggingface_hub
HF="$HF_VENV/bin/hf"
if [[ ! -x "$HF" ]]; then
    echo "ERROR: the Hugging Face 'hf' CLI was not installed in $HF_VENV." >&2
    exit 1
fi

# ---------- Exact registered Qwen3.8-27B NVFP4 artifact ----------
needs_download=1
if [[ -f "$MODEL" ]]; then
    echo
    echo "Checking existing model SHA-256..."
    ACTUAL_SHA="$(sha256sum "$MODEL" | awk '{print $1}')"
    if [[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]]; then
        echo "Existing model is correct; download skipped."
        needs_download=0
    else
        echo "Existing model hash is wrong; removing it before a clean re-download." >&2
        rm -f "$MODEL"
    fi
fi

if (( needs_download )); then
    echo
    echo "Downloading Qwen3.8-27B NVFP4 NInfer artifact..."
    "$HF" download \
        neroued/Qwen3.8-27B-nvfp4-NInfer \
        qwen3_8_27b_nvfp4.ninfer \
        --local-dir "$MODEL_DIR"
fi

ACTUAL_SHA="$(sha256sum "$MODEL" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
    echo "ERROR: model SHA-256 mismatch after download." >&2
    echo "Expected: $EXPECTED_SHA" >&2
    echo "Actual:   $ACTUAL_SHA" >&2
    exit 1
fi
echo "Model SHA-256 verified."
MODEL_BYTES="$(stat -c %s "$MODEL")"

# ---------- Persistent launcher (start/stop/status + pidfile) ----------
cat > "$LAUNCHER" <<'BASH'
#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${NINFER_ROOT:-$HOME/ninfer-qwen38}"
SERVER="$ROOT/ninfer/build-cuda131/apps/ninfer-serve"
MODEL="$ROOT/models/qwen3_8_27b_nvfp4.ninfer"
LOG_DIR="$ROOT/logs"
PIDFILE="$ROOT/server.pid"
HOST="${NINFER_HOST:-127.0.0.1}"
PORT="${NINFER_PORT:-8081}"
CONCURRENCY="${NINFER_CONCURRENCY:-3}"
MIN_CONTEXT="${NINFER_MIN_CONTEXT:-163840}"

export PATH="/usr/lib/wsl/lib:/usr/local/cuda-13.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-13.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Health is always checked over loopback so the server's own probe works
# regardless of NINFER_HOST.
health_ok() {
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1
}

stop_server() {
    local pid=""
    if [[ -f "$PIDFILE" ]]; then
        pid="$(cat "$PIDFILE")"
    fi
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "Sending SIGINT to NInfer server PID $pid..."
        kill -INT "$pid" 2>/dev/null || true
        for _ in $(seq 1 15); do
            kill -0 "$pid" 2>/dev/null || { rm -f "$PIDFILE"; echo "Server stopped."; return 0; }
            sleep 1
        done
        echo "Server did not exit within 15s after SIGINT; forcing SIGKILL." >&2
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: server PID $pid is still alive." >&2
            return 1
        fi
        rm -f "$PIDFILE"
        echo "Server stopped."
        return 0
    fi
    if health_ok; then
        echo "ERROR: no pidfile, but something answers on http://127.0.0.1:$PORT/health. Stop it manually." >&2
        return 1
    fi
    echo "No running NInfer server (no pidfile)."
    return 0
}

status_server() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && health_ok; then
        echo "running pid=$(cat "$PIDFILE")"
    else
        echo "stopped"
    fi
}

case "${1:-}" in
    stop) stop_server; exit $? ;;
    status) status_server; exit 0 ;;
esac

# start mode: forward every remaining argument to ninfer-serve
SERVER_ARGS=("$@")

if (( CONCURRENCY < 1 || CONCURRENCY > 8 )); then
    echo "ERROR: NINFER_CONCURRENCY must be in 1..8." >&2
    exit 1
fi

if [[ ! -x "$SERVER" || ! -f "$MODEL" ]]; then
    echo "ERROR: NInfer or the model is missing. Rerun setup_ninfer_qwen38.cmd." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
REQUEST_LOG="$LOG_DIR/server-$STAMP.requests.jsonl"

if health_ok; then
    echo "ERROR: something is already answering on http://127.0.0.1:$PORT/health" >&2
    echo "Stop it or choose another port with NINFER_PORT=..." >&2
    exit 1
fi

# If NINFER_MAX_CONTEXT is set explicitly, only that value is tried.
# Otherwise we start at the native 262,144 ceiling and step down until the
# largest practical profile in this ladder starts with Vision+MTP3+C=3.
if [[ -n "${NINFER_MAX_CONTEXT:-}" ]]; then
    CONTEXTS=("$NINFER_MAX_CONTEXT")
else
    CONTEXTS=(262144 245760 229376 212992 196608 180224 163840)
fi

echo
echo "Starting Qwen3.8-27B NVFP4"
echo "  Vision:          enabled"
echo "  MTP:             3 draft tokens + optimized proposal head"
echo "  KV:              INT8 group-64, shared pool, auto-sized"
echo "  Concurrency cap: $CONCURRENCY"
echo "  Context target:  largest startup-successful value >= $MIN_CONTEXT"
echo "  CUDA Graphs:     enabled (NInfer default)"
echo "  Prefix reuse:    enabled (NInfer default)"
echo "  Extra args:      ${SERVER_ARGS[*]:-none}"
echo

try_ladder() {
    local PASS_C=$1
    local CTX
    for CTX in "${CONTEXTS[@]}"; do
        if (( CTX < MIN_CONTEXT )); then
            continue
        fi

        echo "------------------------------------------------------------"
        echo "Trying max-context=$CTX with kv-capacity=auto (concurrency=$PASS_C) ..."
        echo "------------------------------------------------------------"

        "$SERVER" "$MODEL" \
            --host "$HOST" \
            --port "$PORT" \
            --max-context "$CTX" \
            --kv-capacity auto \
            --max-concurrency "$PASS_C" \
            --kv-dtype int8 \
            --prefill-chunk 1024 \
            --spec mtp \
            --draft-tokens 3 \
            --lm-head-draft \
            --vision \
            --request-log-jsonl "$REQUEST_LOG" \
            "${SERVER_ARGS[@]}" &
        PID=$!
        echo "$PID" > "$PIDFILE"

        HEALTHY=0
        while kill -0 "$PID" 2>/dev/null; do
            if health_ok; then
                HEALTHY=1
                break
            fi
            sleep 1
        done

        if (( HEALTHY )); then
            echo
            echo "============================================================"
            echo " READY"
            echo "============================================================"
            echo "Model:            qwen3.8-27b / nvfp4"
            echo "Endpoint:         http://$HOST:$PORT/v1"
            echo "Max context:      $CTX tokens per sequence"
            echo "Max concurrency:  $PASS_C"
            echo "Vision:           ON"
            echo "MTP3:             ON"
            echo "KV cache:         INT8, auto-sized shared pool"
            echo "Request log:      $REQUEST_LOG"
            echo "PID file:         $PIDFILE"
            echo
            echo "Current GPU memory after startup:"
            nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true
            echo
            echo "NInfer's startup output above contains the exact resolved KV-capacity ledger."

            cleanup() {
                rm -f "$PIDFILE"
            }
            trap 'kill -INT "$PID" 2>/dev/null || true; cleanup' INT TERM EXIT
            if wait "$PID"; then
                exit 0
            else
                exit $?
            fi
        fi

        if wait "$PID"; then
            RC=0
        else
            RC=$?
        fi
        rm -f "$PIDFILE"
        echo "Startup at max-context=$CTX exited with code $RC; trying the next lower context target." >&2
    done
    return 1
}

if try_ladder "$CONCURRENCY"; then
    exit 0
fi

if (( CONCURRENCY > 1 )); then
    echo >&2
    echo "No context >= $MIN_CONTEXT started with concurrency=$CONCURRENCY." >&2
    echo "Retrying the ladder once with concurrency=1 (KV pool sized for one active sequence)." >&2
    echo >&2
    if try_ladder 1; then
        exit 0
    fi
fi

echo >&2
echo "ERROR: NVFP4 + Vision + MTP3 could not start at any configured context >= $MIN_CONTEXT." >&2
echo "I am deliberately NOT silently switching to a different quant or disabling Vision/MTP." >&2
echo "The next step is to inspect the startup memory ledger and decide which constraint to trade." >&2
exit 1
BASH
chmod +x "$LAUNCHER"

# ---------- Facts file the Windows side reads ----------
cat > "$FACTS" <<EOF
{
  "home": "$HOME",
  "launcher_path": "$LAUNCHER",
  "model_path": "$MODEL",
  "model_size_bytes": $MODEL_BYTES,
  "model_sha256": "$ACTUAL_SHA",
  "runtime_revision": "$CURRENT_REV"
}
EOF

echo
echo "============================================================"
echo " Installation complete"
echo "============================================================"
echo "Root:     $ROOT"
echo "Model:    $MODEL"
echo "Launcher: $LAUNCHER"
echo
echo "NInfer is installed but NOT started. Start it from the loader GUI"
echo "(the Qwen3.8-27B-NVFP4 (NInfer) preset) or with:"
echo "  $LAUNCHER"
echo
'@

$TempScript = Join-Path ([IO.Path]::GetTempPath()) ("ninfer-qwen38-setup-{0}.sh" -f $PID)
$Utf8NoBom = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText($TempScript, ($BashScript -replace "`r`n", "`n"), $Utf8NoBom)

try {
    # wsl.exe treats backslashes as escapes in its own arguments, so hand
    # wslpath a forward-slash Windows path; it resolves both forms fine.
    $TempScriptForward = $TempScript -replace '\\', '/'
    $WslTempPath = ((& wsl.exe -d $Distro -- wslpath -u -a $TempScriptForward) | Out-String).Replace("`0", '').Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($WslTempPath)) {
        throw "Could not translate the temporary PowerShell-created script path into a WSL path."
    }

    Write-Host "Handing the literal Bash script to WSL (no nested bash -c quoting)..." -ForegroundColor Cyan
    & wsl.exe -d $Distro -- bash $WslTempPath
    $rc = $LASTEXITCODE
    if ($rc -ne 0) {
        throw "NInfer setup exited with code $rc."
    }
}
finally {
    Remove-Item -LiteralPath $TempScript -Force -ErrorAction SilentlyContinue
}

# ---------- Read facts back and register with the loader ----------
$rawFacts = (& wsl.exe -d $Distro -- bash -lc 'cat "$HOME/ninfer-qwen38/wsl-info.json"') -join "`n"
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($rawFacts)) {
    throw "Could not read wsl-info.json from the distro."
}
$wslInfo = $rawFacts | ConvertFrom-Json

$factsPath = Join-Path $AppDataDir "ninfer-wsl.json"
New-Item -ItemType Directory -Path $AppDataDir -Force | Out-Null
$facts = [ordered]@{
    distro            = $Distro
    home              = $wslInfo.home
    launcher_path     = $wslInfo.launcher_path
    model_path        = $wslInfo.model_path
    model_size_bytes  = [long]$wslInfo.model_size_bytes
    model_sha256      = $wslInfo.model_sha256
    runtime_revision  = $wslInfo.runtime_revision
    port              = 8081
    concurrency       = 3
    max_context       = 262144
    min_context       = 163840
}
$facts | ConvertTo-Json | Set-Content -LiteralPath $factsPath -Encoding UTF8
Write-Host "Wrote loader facts: $factsPath" -ForegroundColor Green

& $SetupPython -m backend.app.ninfer_setup --info $factsPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to register the NInfer model and loading preset."
}

# ---------- LAN reachability (portproxy + firewall + re-sync task) ----------
if (-not $SkipLanSetup) {
    $helper = Join-Path $RepoRoot "scripts\ninfer_lan_helper.ps1"
    Write-Host "Setting up LAN reachability for the NInfer server..." -ForegroundColor Cyan
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $helper -Distro $Distro -Port $facts.port
    if ($LASTEXITCODE -ne 0) {
        Write-Host "LAN setup was skipped (run scripts\infer_lan_helper.ps1 later as Administrator)." -ForegroundColor Yellow
    }
}

if ($Start) {
    Write-Host "Starting the NInfer server now (-Start)..." -ForegroundColor Cyan
    & wsl.exe -d $Distro -- bash -lc "$($wslInfo.launcher_path) >/dev/null 2>&1 &"
    $healthy = $false
    for ($i = 0; $i -lt 120; $i++) {
        try {
            $probe = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$($facts.port)/health" -TimeoutSec 1
            if ($probe.StatusCode -ge 200 -and $probe.StatusCode -lt 300) {
                $healthy = $true
                break
            }
        } catch { }
        Start-Sleep -Seconds 1
    }
    if ($healthy) {
        Write-Host "NInfer server is healthy on http://127.0.0.1:$($facts.port)" -ForegroundColor Green
    } else {
        Write-Host "Server did not report /health within 120s; check the run log in the loader GUI." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "NInfer setup complete." -ForegroundColor Green
Write-Host "Model:   $($wslInfo.model_path)"
Write-Host "Runtime: $($wslInfo.launcher_path)"
Write-Host "Revision: $($wslInfo.runtime_revision)"
Write-Host "Start the loader with: python run_dev.py"
Write-Host "Then load the 'Qwen3.8-27B-NVFP4 (NInfer)' model and press Start in 'Active runs & terminal'."
