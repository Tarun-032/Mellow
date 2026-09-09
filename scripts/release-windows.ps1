$ErrorActionPreference = "Stop"

$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$userProfile = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
$keyPath = Join-Path $userProfile ".tauri\mellow-updater.key"
$passwordPath = Join-Path $userProfile ".tauri\mellow-updater.password.txt"

if (-not (Test-Path -LiteralPath $keyPath -PathType Leaf)) {
    throw "Updater key not found: $keyPath"
}
if (-not (Test-Path -LiteralPath $passwordPath -PathType Leaf)) {
    throw "Updater key password not found: $passwordPath"
}

$tauri = Get-Content -LiteralPath (Join-Path $taskRoot "src-tauri\tauri.conf.json") -Raw | ConvertFrom-Json
$package = Get-Content -LiteralPath (Join-Path $taskRoot "package.json") -Raw | ConvertFrom-Json
$cargo = Get-Content -LiteralPath (Join-Path $taskRoot "src-tauri\Cargo.toml") -Raw
$sidecar = Get-Content -LiteralPath (Join-Path $taskRoot "mellowd\version.py") -Raw
$cargoVersion = [regex]::Match($cargo, '(?m)^version\s*=\s*"([^"]+)"').Groups[1].Value
$sidecarVersion = [regex]::Match($sidecar, '(?m)^VERSION\s*=\s*"([^"]+)"').Groups[1].Value
$versions = @($tauri.version, $package.version, $cargoVersion, $sidecarVersion) | Select-Object -Unique

if ($versions.Count -ne 1) {
    throw "Release versions do not match: $($versions -join ', ')"
}

$oldKeyPath = $env:TAURI_SIGNING_PRIVATE_KEY_PATH
$oldPassword = $env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD
$env:TAURI_SIGNING_PRIVATE_KEY_PATH = $keyPath
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = (Get-Content -LiteralPath $passwordPath -Raw).Trim()

try {
    & npm run sidecar:build
    if ($LASTEXITCODE -ne 0) { throw "Sidecar build failed." }

    & npm run sidecar:verify
    if ($LASTEXITCODE -ne 0) { throw "Sidecar verification failed." }

    & npm run tauri -- build --bundles nsis
    if ($LASTEXITCODE -ne 0) { throw "Tauri build failed." }

    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "publish-installer.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Publishing updater files failed." }
}
finally {
    $env:TAURI_SIGNING_PRIVATE_KEY_PATH = $oldKeyPath
    $env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = $oldPassword
}
