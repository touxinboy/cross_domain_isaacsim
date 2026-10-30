#requires -Version 7.0
<##
Calls the already running own A100 service. Does not submit/start a GPU job.
An SSH tunnel started by this command is closed by this command on exit.
Keys stay in process memory and are never printed or saved.
##>
[CmdletBinding()]
param(
    [ValidateSet('models','text','image','decision','grounding')][string]$Action = 'models',
    [string]$ImagePath,
    [string]$ObservationPath,
    [string]$Target,
    [string]$OutputPath,
    [ValidateRange(1024,65535)][int]$LocalPort = 18000,
    [string]$Python = 'D:\python313\python.exe',
    [string]$SshConfig
)
$ErrorActionPreference = 'Stop'
$taskWorkspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../..'))
if (-not $SshConfig) {
    $SshConfig = Join-Path $taskWorkspace 'analysis/2026-10-28/qwen_connection/ssh_config'
}
if (-not (Test-Path -LiteralPath $SshConfig)) { throw 'SSH config not found.' }
if ($Action -in @('image','grounding') -and -not (Test-Path -LiteralPath $ImagePath -PathType Leaf)) {
    throw 'Provide -ImagePath with an existing observation image.'
}
if ($Action -eq 'decision' -and -not (Test-Path -LiteralPath $ObservationPath -PathType Leaf)) {
    throw 'Provide -ObservationPath with an existing observation JSON.'
}

function Test-OwnLocalPort([int]$Port) {
    $taskSocket = [System.Net.Sockets.TcpClient]::new()
    try {
        $taskWait = $taskSocket.ConnectAsync('127.0.0.1', $Port)
        return $taskWait.Wait(500) -and $taskSocket.Connected
    } catch { return $false } finally { $taskSocket.Dispose() }
}

$taskTunnel = $null
$taskKeyText = $null
$taskEnvNames = @('QWEN_BASE_URL','QWEN_MODEL','QWEN_API_KEY','PYTHONIOENCODING')
$taskSavedEnv = @{}
foreach ($taskName in $taskEnvNames) {
    $taskSavedEnv[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process')
}
try {
    if (-not (Test-OwnLocalPort $LocalPort)) {
        $taskStart = [System.Diagnostics.ProcessStartInfo]::new()
        $taskStart.FileName = (Get-Command ssh -CommandType Application).Source
        $taskStart.UseShellExecute = $false
        $taskStart.CreateNoWindow = $true
        $taskStart.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
        $taskStart.RedirectStandardError = $true
        foreach ($taskArg in @('-F',$SshConfig,'-N','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30',
                               '-L',"127.0.0.1:${LocalPort}:127.0.0.1:18080",'epic-a100')) {
            $taskStart.ArgumentList.Add($taskArg)
        }
        $taskTunnel = [System.Diagnostics.Process]::Start($taskStart)
        $taskReady = $false
        for ($taskTry = 0; $taskTry -lt 25; $taskTry++) {
            if ($taskTunnel.HasExited) { throw 'SSH tunnel failed; check the current network and SSH config.' }
            if (Test-OwnLocalPort $LocalPort) { $taskReady = $true; break }
            Start-Sleep -Milliseconds 200
        }
        if (-not $taskReady) { throw 'SSH tunnel did not become ready.' }
    }
    $taskKeyText = & ssh -T -F $SshConfig epic-a100 'cat /home/wangjiong/L17/qwen_service/credentials/api_key'
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the own service credential.' }
    $env:QWEN_API_KEY = ($taskKeyText -join '').Trim()
    if (-not $env:QWEN_API_KEY) { throw 'The own service credential is empty.' }
    $env:QWEN_BASE_URL = "http://127.0.0.1:${LocalPort}/v1"
    $env:QWEN_MODEL = 'l17-qwen3-vl-4b'
    $env:PYTHONIOENCODING = 'utf-8'
    switch ($Action) {
        'models' { & $Python -B (Join-Path $PSScriptRoot 'check_qwen_api.py') --models }
        'text' { & $Python -B (Join-Path $PSScriptRoot 'check_qwen_api.py') --text --timeout 120 }
        'image' { & $Python -B (Join-Path $PSScriptRoot 'check_qwen_api.py') --image $ImagePath --timeout 120 }
        'decision' { & $Python -B (Join-Path $PSScriptRoot 'run_navigation_interface.py') --observation $ObservationPath }
        'grounding' {
            if (-not $Target) { throw 'Provide -Target with the visible target description.' }
            $taskGroundArgs=@('-B',(Join-Path $PSScriptRoot 'ground_target_api.py'),'--image',$ImagePath,'--target',$Target)
            if ($OutputPath) { $taskGroundArgs+=@('--output',$OutputPath) }
            & $Python @taskGroundArgs
        }
    }
    if ($LASTEXITCODE -ne 0) { throw 'API check failed. Check that the own Slurm service is still running.' }
} finally {
    foreach ($taskName in $taskEnvNames) {
        [Environment]::SetEnvironmentVariable($taskName, $taskSavedEnv[$taskName], 'Process')
    }
    $taskKeyText = $null
    if ($taskTunnel) {
        if (-not $taskTunnel.HasExited) { $taskTunnel.Kill(); $taskTunnel.WaitForExit(3000) | Out-Null }
        $taskTunnel.Dispose()
    }
}
