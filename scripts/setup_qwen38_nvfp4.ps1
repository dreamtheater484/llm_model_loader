param(
    [string]$ModelDir
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$AiRoot = "D:\Documents\AI"
$SetupPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $SetupPython)) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $SetupPython = $pythonCommand.Source }
}
if ([string]::IsNullOrWhiteSpace($ModelDir)) {
    $ModelDir = Join-Path $AiRoot "models"
    $dbPath = Join-Path $env:LOCALAPPDATA "llm-model-loader\loader.sqlite3"
    if ((Test-Path -LiteralPath $dbPath) -and (Test-Path -LiteralPath $SetupPython)) {
        try {
            $configured = & $SetupPython -c "from backend.app.storage import store; print(store.setting('model_dir') or '')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $configured) { $ModelDir = $configured.Trim() }
        } catch { }
    }
}
$OfficialRoot = Join-Path $AiRoot "llama.cpp-official"
$ModelName = "Qwen3.8-27B-NVFP4-MTP.gguf"
$ModelPath = Join-Path $ModelDir $ModelName
$ModelUrl = "https://huggingface.co/felippeburk/Qwen3.8-27B-NVFP4-MTP-GGUF/resolve/main/qwen3.8-27b-text-nvfp4-mtp.gguf"
$ModelBytes = 19653896608
$ModelSha256 = "4fd7a135dabbe267f9d0c6916c642699786e67c047f05b6c002e52a8294a8c42"
$MmprojName = "mmproj-F16.gguf"
$MmprojPath = Join-Path $ModelDir $MmprojName
$MmprojUrl = "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-F16.gguf"
$MmprojBytes = 927607488
$MmprojSha256 = "cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Download-Verified([string]$Url, [string]$Target, [string]$ExpectedHash, [long]$ExpectedBytes = 0) {
    $targetFile = [System.IO.FileInfo]$Target
    New-Item -ItemType Directory -Path $targetFile.DirectoryName -Force | Out-Null
    if (Test-Path -LiteralPath $Target) {
        $hash = Get-Sha256 $Target
        if ($hash -eq $ExpectedHash -and (-not $ExpectedBytes -or (Get-Item -LiteralPath $Target).Length -eq $ExpectedBytes)) {
            return
        }
        throw "Existing file failed verification: $Target"
    }
    $part = "$Target.part"
    Write-Host "Downloading $([System.IO.Path]::GetFileName($Target)) (resumable)..."
    & curl.exe -L --fail --retry 5 --retry-delay 2 --retry-all-errors -C - --output $part $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Url"
    }
    if ($ExpectedBytes -and (Get-Item -LiteralPath $part).Length -ne $ExpectedBytes) {
        throw "Downloaded size mismatch for $Target"
    }
    if ((Get-Sha256 $part) -ne $ExpectedHash) {
        throw "SHA-256 verification failed for $Target"
    }
    Move-Item -LiteralPath $part -Destination $Target
}

function Find-Server([string]$Root) {
    if (-not (Test-Path -LiteralPath $Root)) { return $null }
    return Get-ChildItem -LiteralPath $Root -Filter "llama-server.exe" -File -Recurse | Select-Object -First 1
}

function Resolve-Cuda13Runtime {
    # Reuse an existing official CUDA 13 runtime; never install a new one.
    $candidates = @()
    if (Test-Path -LiteralPath $OfficialRoot) {
        $candidates = Get-ChildItem -LiteralPath $OfficialRoot -Directory |
            Where-Object { $_.Name -match "^b\d+-cuda13" } |
            Sort-Object { [int]($_.Name -replace '^b(\d+)-.*', '$1') } -Descending
    }
    foreach ($candidate in $candidates) {
        $server = Find-Server $candidate.FullName
        if ($server) {
            & $server.FullName --version
            if ($LASTEXITCODE -eq 0) { return $server.FullName }
        }
    }
    $configured = ""
    $dbPath = Join-Path $env:LOCALAPPDATA "llm-model-loader\loader.sqlite3"
    if ((Test-Path -LiteralPath $dbPath) -and (Test-Path -LiteralPath $SetupPython)) {
        try {
            $configured = & $SetupPython -c "from backend.app.storage import store; print(store.setting('llama_server_path') or '')" 2>$null
            if ($LASTEXITCODE -eq 0) { $configured = $configured.Trim() }
        } catch { }
    }
    if ($configured -and (Test-Path -LiteralPath $configured)) {
        & $configured --version
        if ($LASTEXITCODE -eq 0) { return $configured }
    }
    throw "No usable CUDA 13 llama-server found under $OfficialRoot (b*-cuda13*) or in the loader settings. This setup reuses an existing runtime and does not install one."
}

Set-Location $RepoRoot
New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
$RuntimeServer = Resolve-Cuda13Runtime
Write-Host "Reusing runtime: $RuntimeServer"

Download-Verified $ModelUrl $ModelPath $ModelSha256 $ModelBytes

if (Test-Path -LiteralPath $MmprojPath) {
    # Reuse the vision projection already on disk (validated against the model
    # by llama.cpp at load time). Only hash-verify when we fetch a fresh copy.
    Write-Host "Reusing existing mmproj: $MmprojPath"
} else {
    Download-Verified $MmprojUrl $MmprojPath $MmprojSha256 $MmprojBytes
}

if (-not (Test-Path -LiteralPath $SetupPython)) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw "Python is required. Run setup.cmd first." }
    $SetupPython = $pythonCommand.Source
}
& $SetupPython -m backend.app.qwen38_setup --model-path $ModelPath --runtime-path $RuntimeServer --mmproj-path $MmprojPath
if ($LASTEXITCODE -ne 0) { throw "Failed to register the Qwen3.8-27B NVFP4-MTP model and loading preset." }

Write-Host ""
Write-Host "Qwen3.8-27B NVFP4-MTP setup complete." -ForegroundColor Green
Write-Host "Model:   $ModelPath"
Write-Host "Vision:  $MmprojPath"
Write-Host "Runtime: $RuntimeServer"
Write-Host "Unload the currently loaded model, then start the loader with: python run_dev.py"
