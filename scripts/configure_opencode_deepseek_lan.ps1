[CmdletBinding()]
param(
    [string]$ServerHost = "192.168.50.23",
    [ValidateRange(1, 65535)]
    [int]$Port = 8080,
    [string]$ConfigPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProviderId = "localllama"
$ModelId = "DeepSeek-V4-Flash-0731-UD-Q8_K_XL-400k-Q8KV-no-draft"
$BaseUrl = "http://{0}:{1}/v1" -f $ServerHost, $Port

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $configCandidates = [System.Collections.Generic.List[string]]::new()
    $addCandidate = {
        param([string]$Path)
        if (-not [string]::IsNullOrWhiteSpace($Path) -and -not $configCandidates.Contains($Path)) {
            $configCandidates.Add($Path)
        }
    }

    & $addCandidate $env:OPENCODE_CONFIG
    if (-not [string]::IsNullOrWhiteSpace($env:OPENCODE_CONFIG_DIR)) {
        & $addCandidate ([System.IO.Path]::Combine($env:OPENCODE_CONFIG_DIR, "opencode.json"))
        & $addCandidate ([System.IO.Path]::Combine($env:OPENCODE_CONFIG_DIR, "opencode.jsonc"))
    }
    if (-not [string]::IsNullOrWhiteSpace($env:XDG_CONFIG_HOME)) {
        & $addCandidate ([System.IO.Path]::Combine($env:XDG_CONFIG_HOME, "opencode", "opencode.json"))
        & $addCandidate ([System.IO.Path]::Combine($env:XDG_CONFIG_HOME, "opencode", "opencode.jsonc"))
    }
    foreach ($root in @($env:USERPROFILE, $env:APPDATA, $env:LOCALAPPDATA)) {
        if (-not [string]::IsNullOrWhiteSpace($root)) {
            & $addCandidate ([System.IO.Path]::Combine($root, ".config", "opencode", "opencode.json"))
            & $addCandidate ([System.IO.Path]::Combine($root, ".config", "opencode", "opencode.jsonc"))
            & $addCandidate ([System.IO.Path]::Combine($root, "opencode", "opencode.json"))
            & $addCandidate ([System.IO.Path]::Combine($root, "opencode", "opencode.jsonc"))
        }
    }

    $projectPath = (Get-Location).Path
    while (-not [string]::IsNullOrWhiteSpace($projectPath)) {
        & $addCandidate ([System.IO.Path]::Combine($projectPath, "opencode.json"))
        & $addCandidate ([System.IO.Path]::Combine($projectPath, "opencode.jsonc"))
        $parentPath = Split-Path -Path $projectPath -Parent
        if ($parentPath -eq $projectPath) {
            break
        }
        $projectPath = $parentPath
    }

    $ConfigPath = $configCandidates | Where-Object {
        Test-Path -LiteralPath $_ -PathType Leaf
    } | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
        throw "Could not autodiscover OpenCode config. Start OpenCode once or pass -ConfigPath <path>. Checked: $($configCandidates -join ', ')"
    }
    Write-Host "Discovered OpenCode config: $ConfigPath"
}

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "OpenCode config was not found at $ConfigPath. Start OpenCode once, then rerun this script."
}

try {
    $config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
} catch {
    throw "Could not parse OpenCode config at $ConfigPath`: $($_.Exception.Message)"
}

$providerRootProperty = $config.PSObject.Properties["provider"]
if ($null -eq $providerRootProperty -or $null -eq $providerRootProperty.Value) {
    throw "OpenCode config has no provider section; refusing to rebuild it."
}

$providerProperty = $providerRootProperty.Value.PSObject.Properties[$ProviderId]
if ($null -eq $providerProperty -or $null -eq $providerProperty.Value) {
    throw "OpenCode config has no '$ProviderId' provider; refusing to create a parallel config."
}

$provider = $providerProperty.Value
$optionsProperty = $provider.PSObject.Properties["options"]
if ($null -eq $optionsProperty -or $null -eq $optionsProperty.Value) {
    $provider | Add-Member -MemberType NoteProperty -Name "options" -Value ([pscustomobject]@{})
}
$options = $provider.PSObject.Properties["options"].Value
$baseUrlProperty = $options.PSObject.Properties["baseURL"]
if ($null -eq $baseUrlProperty -or $baseUrlProperty.Value -ne $BaseUrl) {
    $options | Add-Member -MemberType NoteProperty -Name "baseURL" -Value $BaseUrl -Force
}

$modelsProperty = $provider.PSObject.Properties["models"]
if ($null -eq $modelsProperty -or $null -eq $modelsProperty.Value) {
    throw "OpenCode '$ProviderId' provider has no models section; refusing to rebuild it."
}
$models = $modelsProperty.Value

$model = [pscustomobject]@{
    id = $ModelId
    name = "llama.cpp DeepSeek V4 Flash Q8_K_XL (400k)"
    attachment = $true
    reasoning = $true
    interleaved = "reasoning_content"
    modalities = [pscustomobject]@{
        input = @("text")
        output = @("text")
    }
    limit = [pscustomobject]@{
        context = 409600
        output = 32768
    }
    variants = [pscustomobject]@{
        medium = [pscustomobject]@{
            reasoning_effort = "medium"
        }
        low = [pscustomobject]@{
            reasoning_effort = "low"
        }
        xhigh = [pscustomobject]@{
            reasoning_effort = "xhigh"
        }
        off = [pscustomobject]@{
            reasoning_effort = "none"
            chat_template_kwargs = [pscustomobject]@{
                enable_thinking = $false
            }
        }
    }
}

$models | Add-Member -MemberType NoteProperty -Name $ModelId -Value $model -Force

$backupPath = "$ConfigPath.bak"
Copy-Item -LiteralPath $ConfigPath -Destination $backupPath -Force
$json = $config | ConvertTo-Json -Depth 50
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($ConfigPath, $json, $utf8NoBom)

Write-Host "Added $ModelId to $ConfigPath"
Write-Host "OpenCode model: $ProviderId/$ModelId"
Write-Host "Endpoint: $BaseUrl"
Write-Host "Backup: $backupPath"

try {
    $models = Invoke-RestMethod -Uri ($BaseUrl.TrimEnd("/") + "/models") -TimeoutSec 5
    if (@($models.data | Where-Object { $_.id -eq $ModelId }).Count -gt 0) {
        Write-Host "Endpoint check: model is advertised by llama.cpp." -ForegroundColor Green
    } else {
        Write-Warning "Endpoint responded, but did not advertise $ModelId."
    }
} catch {
    Write-Warning "Endpoint check failed for $BaseUrl. Ensure llama.cpp is bound to 0.0.0.0 and port $Port is allowed through the host firewall."
}
