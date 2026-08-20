# One-time WSL2 NAT LAN wiring for the NInfer server.
#
#   scripts\ninfer_lan_helper.ps1 -Distro Ubuntu -Port 8081
#
# - Creates a SYSTEM scheduled task (onlogon) that keeps the portproxy WSL IP
#   in sync, so the port stays reachable across WSL restarts.
# - Installs a firewall allow rule for the listen port.
# - If the task already exists, re-sync runs elevated as SYSTEM with no UAC.
param(
    [string]$Distro,
    [int]$Port = 8081,
    [switch]$SyncOnly
)

$ErrorActionPreference = "Stop"
$TaskName = "LLM-Model-Loader-NInfer-LAN"
$RuleName = "LLM-Model-Loader-NInfer-$Port"

function Get-WslIp {
    if ([string]::IsNullOrWhiteSpace($Distro)) { throw "-Distro is required." }
    $ip = ((& wsl.exe -d $Distro -- hostname -I) | Out-String).Trim().Split(' ')[0]
    if ([string]::IsNullOrWhiteSpace($ip)) {
        throw "Could not resolve the WSL IP for distro '$Distro'. Is it running?"
    }
    return $ip
}

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-NinferHealthy {
    # Only forward the port while the NInfer server inside WSL actually
    # answers. This keeps the onlogon task from re-creating a stale portproxy
    # that would block other servers (e.g. llama.cpp) from binding the port.
    $ip = Get-WslIp
    try {
        $r = Invoke-WebRequest -Uri "http://$ip`:$Port/health" -TimeoutSec 3 -UseBasicParsing
        return $r.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Sync-PortProxy {
    if (-not (Test-NinferHealthy)) {
        Write-Host "NInfer is not healthy on the WSL host; removing any stale portproxy for port $Port."
        & netsh.exe interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$Port 2>$null | Out-Null
        return
    }
    $ip = Get-WslIp
    Write-Host "Syncing portproxy 0.0.0.0:$Port -> $ip`:$Port"
    & netsh.exe interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$Port 2>$null | Out-Null
    & netsh.exe interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$Port connectaddress=$ip connectport=$Port
    if ($LASTEXITCODE -ne 0) { throw "netsh portproxy add failed." }
    if (-not (Get-NetFirewallRule -Name $RuleName -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $RuleName -Name $RuleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port | Out-Null
        Write-Host "Created firewall rule: $RuleName"
    }
    & netsh.exe interface portproxy show v4tov4
}

function Test-Task {
    # cmd swallows stderr; PS 5.1 treats native stderr as a terminating
    # error under $ErrorActionPreference=Stop, which breaks the "not found" case.
    cmd /c "schtasks /query /tn `"$TaskName`" >nul 2>&1"
    return $LASTEXITCODE -eq 0
}

if (-not $SyncOnly) {
    if (Test-Task) {
        Write-Host "Task '$TaskName' already exists; re-syncing via the SYSTEM task (no UAC)."
        & schtasks.exe /run /tn $TaskName
        if ($LASTEXITCODE -ne 0) { throw "schtasks /run failed for '$TaskName'." }
        exit 0
    }
    if (-not (Test-Admin)) {
        Write-Host "Elevating once to create the LAN forwarder..." -ForegroundColor Cyan
        $args = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Distro `"$Distro`" -Port $Port"
        $p = Start-Process -FilePath powershell.exe -ArgumentList $args -Verb RunAs -Wait -PassThru
        exit $p.ExitCode
    }
    # Admin now: create the re-sync task, then sync immediately.
    $taskArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`" -SyncOnly -Distro `"$Distro`" -Port $Port"
    & schtasks.exe /create /tn $TaskName /tr "powershell.exe $taskArgs" /sc onlogon /rl highest /f
    if ($LASTEXITCODE -ne 0) { throw "schtasks /create failed for '$TaskName'." }
    Write-Host "Created scheduled task: $TaskName"
}

Sync-PortProxy
Write-Host "LAN forwarder for NInfer on port $Port is active." -ForegroundColor Green