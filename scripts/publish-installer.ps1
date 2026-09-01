$ErrorActionPreference = "Stop"

$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$configPath = Join-Path $taskRoot "src-tauri\tauri.conf.json"
$version = (Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).version
$bundleDirectory = Join-Path $taskRoot "src-tauri\target\release\bundle\nsis"
$generatedInstaller = Join-Path $bundleDirectory "Mellow_${version}_x64-setup.exe"
$publishedInstaller = Join-Path $bundleDirectory "Mellow-Setup-${version}-x64.exe"

if (-not (Test-Path -LiteralPath $generatedInstaller -PathType Leaf)) {
    throw "Generated installer not found: $generatedInstaller"
}

# Distinct release filename so Explorer does not reuse a stale icon cache.
Move-Item -LiteralPath $generatedInstaller -Destination $publishedInstaller -Force
Write-Output "Published installer: $publishedInstaller"
