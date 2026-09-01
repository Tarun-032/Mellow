param(
    [switch]$NoClean
)

$ErrorActionPreference = "Stop"
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $taskRoot ".venv\Scripts\python.exe"
$stageRoot = [IO.Path]::GetFullPath((Join-Path $taskRoot "src-tauri\resources"))
$sidecar = [IO.Path]::GetFullPath((Join-Path $stageRoot "mellowd"))
$work = [IO.Path]::GetFullPath((Join-Path $taskRoot "build\pyinstaller"))

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Create .venv and install mellowd/requirements.txt before packaging."
}

& $python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Install build requirements first: .\.venv\Scripts\python.exe -m pip install -r mellowd\requirements-build.txt"
}

# One version must describe the UI, native shell, and service handshake.
$packageVersion = (Get-Content -Raw (Join-Path $taskRoot "package.json") | ConvertFrom-Json).version
$tauriVersion = (Get-Content -Raw (Join-Path $taskRoot "src-tauri\tauri.conf.json") | ConvertFrom-Json).version
$cargoText = Get-Content -Raw (Join-Path $taskRoot "src-tauri\Cargo.toml")
$cargoMatch = [regex]::Match($cargoText, '(?m)^version\s*=\s*"([^\"]+)"')
if (-not $cargoMatch.Success) {
    throw "Could not read the package version from src-tauri/Cargo.toml"
}
$cargoVersion = $cargoMatch.Groups[1].Value
$serviceVersion = (& $python -c "from mellowd.version import VERSION; print(VERSION)")
$versions = @($packageVersion, $tauriVersion, $cargoVersion, $serviceVersion)
if (($versions | Select-Object -Unique).Count -ne 1) {
    throw "Version mismatch: package=$packageVersion tauri=$tauriVersion cargo=$cargoVersion service=$serviceVersion"
}

if (-not $stageRoot.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside the repository: $stageRoot"
}
if (-not $work.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside the repository: $work"
}

New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
New-Item -ItemType Directory -Force -Path $work | Out-Null
if (Test-Path -LiteralPath $sidecar) {
    Remove-Item -LiteralPath $sidecar -Recurse -Force
}

$arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--distpath", $stageRoot,
    "--workpath", $work
)
if (-not $NoClean) {
    $arguments += "--clean"
}
$arguments += (Join-Path $taskRoot "mellowd.spec")

Push-Location $taskRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$exe = Join-Path $sidecar "mellowd.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "PyInstaller did not produce $exe"
}

$bytes = (Get-ChildItem -LiteralPath $sidecar -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("Packaged mellowd {0} ({1:N1} MB)" -f $packageVersion, ($bytes / 1MB))
