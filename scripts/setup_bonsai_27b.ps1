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
$RuntimeRoot = Join-Path $AiRoot "runtimes\prism-b9596-9fcaed7-cuda12.4"
$RuntimeAsset = "llama-prism-b1-9fcaed7-bin-win-cuda-12.4-x64.zip"
$RuntimeSha256 = "6d109e2930c0eaf2f729c3a6fc58dd7809ce2ba7047bfb294547cc389af6de5d"
$ReleaseTag = "prism-b9596-9fcaed7"
$ReleaseBase = "https://github.com/PrismML-Eng/llama.cpp/releases/download/$ReleaseTag"
$ModelName = "Ternary-Bonsai-27B-Q2_0.gguf"
$ModelPath = Join-Path $ModelDir $ModelName
$ModelUrl = "https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/$ModelName"
$ModelBytes = 7165121600
$ModelSha256 = "868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757"
$CudaAsset = "cudart-llama-bin-win-cuda-12.4-x64.zip"
$CudaSha256 = "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d"

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

function Install-PrismRuntime {
    $stamp = Join-Path $RuntimeRoot ".llama_release"
    $existing = Find-Server $RuntimeRoot
    if ($existing -and (Test-Path -LiteralPath $stamp) -and ((Get-Content -Raw -LiteralPath $stamp).Trim() -eq $ReleaseTag)) {
        & $existing.FullName --version
        if ($LASTEXITCODE -eq 0) { return $existing.FullName }
    }
    if (Test-Path -LiteralPath $RuntimeRoot) {
        throw "Runtime directory exists but is not a valid $ReleaseTag install: $RuntimeRoot"
    }

    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "llm-model-loader-bonsai-$([guid]::NewGuid().ToString('N'))"
    $zip = Join-Path $tempRoot $RuntimeAsset
    $extract = Join-Path $tempRoot "extract"
    New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
    try {
        Download-Verified "$ReleaseBase/$RuntimeAsset" $zip $RuntimeSha256
        Expand-Archive -LiteralPath $zip -DestinationPath $extract
        $staged = Find-Server $extract
        if (-not $staged) { throw "PrismML archive did not contain llama-server.exe." }
        New-Item -ItemType Directory -Path (Split-Path -Parent $RuntimeRoot) -Force | Out-Null
        New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
        Get-ChildItem -LiteralPath $extract | Copy-Item -Destination $RuntimeRoot -Recurse -Force

        $stockRoot = Join-Path $AiRoot "llama.cpp"
        foreach ($dllName in @("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll")) {
            $destination = Join-Path $RuntimeRoot $dllName
            $source = Join-Path $stockRoot $dllName
            if (-not (Test-Path -LiteralPath $destination) -and (Test-Path -LiteralPath $source)) {
                Copy-Item -LiteralPath $source -Destination $destination
            }
        }
        $server = (Find-Server $RuntimeRoot).FullName
        try { & $server --version } catch { }
        if ($LASTEXITCODE -ne 0) {
            $cudaZip = Join-Path $tempRoot $CudaAsset
            Download-Verified "$ReleaseBase/$CudaAsset" $cudaZip $CudaSha256
            Expand-Archive -LiteralPath $cudaZip -DestinationPath $RuntimeRoot -Force
        }
        $server = (Find-Server $RuntimeRoot).FullName
        & $server --version
        if ($LASTEXITCODE -ne 0) { throw "PrismML llama-server failed validation." }
        Set-Content -LiteralPath (Join-Path $RuntimeRoot ".llama_release") -Value $ReleaseTag -NoNewline
        return $server
    } finally {
        if (Test-Path -LiteralPath $tempRoot) { Remove-Item -LiteralPath $tempRoot -Recurse -Force }
    }
}

Set-Location $RepoRoot
New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
$PrismServer = Install-PrismRuntime
Download-Verified $ModelUrl $ModelPath $ModelSha256 $ModelBytes

if (-not (Test-Path -LiteralPath $SetupPython)) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw "Python is required. Run setup.cmd first." }
    $SetupPython = $pythonCommand.Source
}
& $SetupPython -m backend.app.bonsai_setup --model-path $ModelPath --runtime-path $PrismServer
if ($LASTEXITCODE -ne 0) { throw "Failed to register the Bonsai model and loading preset." }

Write-Host ""
Write-Host "Bonsai setup complete." -ForegroundColor Green
Write-Host "Model:  $ModelPath"
Write-Host "Runtime: $PrismServer"
Write-Host "Start the loader with: python run_dev.py"
