param(
    [string]$Executable = ""
)

$ErrorActionPreference = "Stop"
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $Executable) {
    $Executable = Join-Path $taskRoot "src-tauri\resources\mellowd\mellowd.exe"
}
$Executable = [IO.Path]::GetFullPath($Executable)
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Packaged sidecar not found: $Executable"
}

$scratch = Join-Path ([IO.Path]::GetTempPath()) ("mellow-sidecar-check-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $scratch | Out-Null
$oldAppData = $env:APPDATA
$process = $null
try {
    # Console subsystem so --package-check is synchronous (app still hides it).
    $checkText = (& $Executable --package-check 2>&1 | Out-String).Trim()
    $checkExit = $LASTEXITCODE
    if ($checkExit -ne 0) {
        throw "Packaged dependency check failed with exit code ${checkExit}: $checkText"
    }
    $checkResult = $checkText | ConvertFrom-Json
    if (-not $checkResult.ok -or $checkResult.checks.torch) {
        throw "Packaged dependency check returned an invalid result: $checkText"
    }

    $probe = New-Object Net.Sockets.TcpClient
    try {
        $probe.Connect("127.0.0.1", 8765)
        throw "Port 8765 is already occupied. Stop the existing Mellow sidecar before verification."
    } catch [Net.Sockets.SocketException] {
        # Expected: port free so this check owns the server.
    } finally {
        $probe.Dispose()
    }

    # Isolated pet-only config with a fake key that must stay in-process.
    $env:APPDATA = $scratch
    $secret = "mellow-release-check-secret"
    & (Join-Path $taskRoot ".venv\Scripts\python.exe") -c "from mellowd import config; cfg=config.load(); cfg['ai_enabled']=False; cfg['llm']['api_key']='$secret'; config.save(cfg)"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not prepare the isolated verification config."
    }
    $process = Start-Process -FilePath $Executable -WorkingDirectory (Split-Path $Executable) -WindowStyle Hidden -PassThru

    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    $health = $null
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($process.HasExited) {
            throw "Packaged sidecar exited before becoming healthy (exit $($process.ExitCode))."
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 1
            break
        } catch {
            Start-Sleep -Milliseconds 100
        }
    }
    if ($null -eq $health) {
        throw "Packaged sidecar did not become healthy within 30 seconds."
    }

    $expectedVersion = (& (Join-Path $taskRoot ".venv\Scripts\python.exe") -c "from mellowd.version import VERSION; print(VERSION)")
    if (-not $health.ok -or $health.service -ne "mellowd" -or $health.protocol -ne 1 -or $health.version -ne $expectedVersion) {
        throw "Unexpected sidecar identity: $($health | ConvertTo-Json -Compress)"
    }

    # /models/available must exist (stale builds passed health without it).
    $available = Invoke-RestMethod -Uri "http://127.0.0.1:8765/models/available" -TimeoutSec 2
    if ($null -eq $available.tts -or $available.tts -isnot [bool]) {
        throw "Packaged /models/available response is invalid: $($available | ConvertTo-Json -Compress)"
    }

    $redacted = Invoke-RestMethod -Uri "http://127.0.0.1:8765/config" -TimeoutSec 2
    $redactedJson = $redacted | ConvertTo-Json -Depth 12 -Compress
    if ($redactedJson.Contains($secret) -or $redacted.settings.llm.api_key -ne "" -or -not $redacted.settings.llm.has_api_key) {
        throw "The packaged /config response exposed or lost the stored-key state."
    }

    $untrusted = Invoke-WebRequest `
        -UseBasicParsing `
        -Uri "http://127.0.0.1:8765/health" `
        -Headers @{ Origin = "https://not-mellow.invalid" } `
        -TimeoutSec 2
    if ($untrusted.Headers["Access-Control-Allow-Origin"]) {
        throw "The packaged service allowed an untrusted browser origin."
    }
    Write-Host "Packaged dependency check passed: $($checkResult | ConvertTo-Json -Compress)"
    Write-Host "Packaged health check passed: $($health | ConvertTo-Json -Compress)"
    Write-Host "Packaged boundary checks passed: API keys redacted; untrusted CORS origin denied"
} finally {
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit()
    }
    $env:APPDATA = $oldAppData
    if (Test-Path -LiteralPath $scratch) {
        Remove-Item -LiteralPath $scratch -Recurse -Force
    }
}
