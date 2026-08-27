$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LlamaBuild = "b10107"
$AiRoot = "D:\Documents\AI"
$LlamaDir = Join-Path $AiRoot "llama.cpp"
$LlamaServer = Join-Path $LlamaDir "llama-server.exe"
$Assets = @(
    @{
        Name = "llama-$LlamaBuild-bin-win-cuda-12.4-x64.zip"
        Sha256 = "1e43bbec9691cd0bc636603c366769148fa6265fd261c5f7c67050b450bbc237"
    },
    @{
        Name = "cudart-llama-bin-win-cuda-12.4-x64.zip"
        Sha256 = "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"
    }
)

function Refresh-Path {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $env:Path = "$userPath;$machinePath"
}

function Test-Command {
    param([string]$Name, [string[]]$Arguments)

    try {
        & $Name @Arguments *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Install-WingetPackage {
    param([string]$Id)

    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw "winget is required. Install App Installer from Microsoft, then run setup.cmd again."
    }

    & winget.exe install --exact --id $Id --source winget `
        --accept-package-agreements --accept-source-agreements --silent
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed to install $Id (exit code $LASTEXITCODE)."
    }
    Refresh-Path
}

function Install-LlamaCpp {
    if (Test-Path -LiteralPath $LlamaServer) {
        & $LlamaServer --version
        if ($LASTEXITCODE -eq 0) {
            Write-Host "llama.cpp is already installed at $LlamaDir"
            return
        }
        throw "An invalid llama.cpp installation already exists at $LlamaDir. Remove or repair it, then rerun setup.cmd."
    }

    if (Test-Path -LiteralPath $LlamaDir) {
        throw "The llama.cpp target directory already exists without llama-server.exe: $LlamaDir"
    }

    $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) "llm-model-loader-$([guid]::NewGuid().ToString('N'))"
    $extractDir = Join-Path $tempDir "llama.cpp"
    New-Item -ItemType Directory -Path $tempDir | Out-Null

    try {
        foreach ($asset in $Assets) {
            $archive = Join-Path $tempDir $asset.Name
            $url = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaBuild/$($asset.Name)"
            Write-Host "Downloading $($asset.Name)..."
            & curl.exe -L --fail --retry 5 --retry-delay 2 --retry-all-errors `
                --output $archive $url
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to download $($asset.Name)."
            }

            $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
            if ($actualHash -ne $asset.Sha256) {
                throw "Checksum verification failed for $($asset.Name)."
            }
        }

        New-Item -ItemType Directory -Path $extractDir | Out-Null
        foreach ($asset in $Assets) {
            Expand-Archive -LiteralPath (Join-Path $tempDir $asset.Name) -DestinationPath $extractDir
        }

        $stagedServer = Join-Path $extractDir "llama-server.exe"
        & $stagedServer --version
        if ($LASTEXITCODE -ne 0) {
            throw "The downloaded llama.cpp build failed validation."
        }

        $parentDir = Split-Path -Parent $LlamaDir
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
        Move-Item -LiteralPath $extractDir -Destination $LlamaDir
    }
    catch {
        throw
    }
    finally {
        if (Test-Path -LiteralPath $tempDir) {
            Remove-Item -LiteralPath $tempDir -Recurse -Force
        }
    }

    & $LlamaServer --version
    if ($LASTEXITCODE -ne 0) {
        throw "llama.cpp was extracted but failed validation."
    }
}

Set-Location $RepoRoot
Refresh-Path

if (-not (Test-Command "python" @("--version"))) {
    Write-Host "Installing Python 3.12..."
    Install-WingetPackage "Python.Python.3.12"
}

if (-not (Test-Command "node" @("--version")) -or -not (Test-Command "npm.cmd" @("--version"))) {
    Write-Host "Installing Node.js LTS..."
    Install-WingetPackage "OpenJS.NodeJS.LTS"
}

Write-Host "Installing Python dependencies..."
& python -m pip install -r (Join-Path $RepoRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Python dependency installation failed."
}

Write-Host "Installing frontend dependencies..."
& npm.cmd ci --prefix (Join-Path $RepoRoot "frontend")
if ($LASTEXITCODE -ne 0) {
    throw "Frontend dependency installation failed."
}

Write-Host "Building frontend..."
& npm.cmd run build --prefix (Join-Path $RepoRoot "frontend")
if ($LASTEXITCODE -ne 0) {
    throw "Frontend build failed."
}

Install-LlamaCpp

Write-Host ""
Write-Host "Setup complete."
Write-Host "Start the application yourself with:"
Write-Host "  python run_dev.py"
