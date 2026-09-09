# Renames the installer, writes latest.json, checks it, and uploads both to a
# draft GitHub release. The updater reads latest.json; a release without it can
# never offer an update.
param(
    [string]$BundleDirectory = "",
    # Overwrite assets on an already-published release.
    [switch]$Force,
    # Build the files without touching GitHub.
    [switch]$SkipUpload
)

$ErrorActionPreference = "Stop"

$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$configPath = Join-Path $taskRoot "src-tauri\tauri.conf.json"
$version = (Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).version
if (-not $BundleDirectory) {
    $BundleDirectory = Join-Path $taskRoot "src-tauri\target\release\bundle\nsis"
}
$bundleDirectory = [IO.Path]::GetFullPath($BundleDirectory)
$generatedInstaller = Join-Path $bundleDirectory "Mellow_${version}_x64-setup.exe"
$publishedInstaller = Join-Path $bundleDirectory "Mellow-Setup-${version}-x64.exe"
$generatedSignature = "$generatedInstaller.sig"
$publishedSignature = "$publishedInstaller.sig"
$metadataPath = Join-Path $bundleDirectory "latest.json"

# Distinct filename so Explorer does not reuse a stale icon cache. Accept either
# name so a re-run after a failed validation still works.
if (Test-Path -LiteralPath $generatedInstaller -PathType Leaf) {
    Move-Item -LiteralPath $generatedInstaller -Destination $publishedInstaller -Force
} elseif (-not (Test-Path -LiteralPath $publishedInstaller -PathType Leaf)) {
    throw "Installer not found. Looked for $generatedInstaller and $publishedInstaller."
}
if (Test-Path -LiteralPath $generatedSignature -PathType Leaf) {
    Move-Item -LiteralPath $generatedSignature -Destination $publishedSignature -Force
} elseif (-not (Test-Path -LiteralPath $publishedSignature -PathType Leaf)) {
    throw "Updater signature not found: $generatedSignature. Build through scripts/release-windows.ps1 so the signing key is set."
}

$assetName = [IO.Path]::GetFileName($publishedInstaller)
$tag = "v$version"
$downloadUrl = "https://github.com/Tarun-032/Mellow/releases/download/$tag/$assetName"
$signature = (Get-Content -LiteralPath $publishedSignature -Raw)
if ([string]::IsNullOrWhiteSpace($signature)) {
    throw "Updater signature is empty: $publishedSignature. The build was not signed, so the update could never install."
}
$signature = $signature.Trim()
$metadata = [ordered]@{
    version = $version
    notes = "See the GitHub release for what is new in Mellow $version."
    pub_date = [DateTime]::UtcNow.ToString("o")
    platforms = [ordered]@{
        "windows-x86_64" = [ordered]@{
            signature = $signature
            url = $downloadUrl
        }
    }
}
$json = $metadata | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText($metadataPath, $json, [Text.UTF8Encoding]::new($false))

# A malformed manifest fails silently in the app.
& node (Join-Path $taskRoot "scripts\updater.check.ts") $metadataPath $version $assetName
if ($LASTEXITCODE -ne 0) { throw "Updater manifest failed validation; nothing was uploaded." }

Write-Output "Published installer: $publishedInstaller"
Write-Output "Published signature: $publishedSignature"
Write-Output "Published updater metadata: $metadataPath"

if ($SkipUpload) {
    Write-Output ""
    Write-Output "Skipped upload. Attach BOTH of these to the $tag release:"
    Write-Output "  $publishedInstaller"
    Write-Output "  $metadataPath"
    return
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "The GitHub CLI (gh) is not installed, so $assetName and latest.json were not uploaded. Install gh, or re-run with -SkipUpload and attach both files by hand."
}
& gh auth status *> $null
if ($LASTEXITCODE -ne 0) {
    throw "gh is not logged in. Run 'gh auth login', or re-run with -SkipUpload and attach both files by hand."
}

# Exits non-zero when the release does not exist yet.
$existing = & gh release view $tag --json isDraft,url 2>$null
if ($LASTEXITCODE -eq 0) {
    $release = $existing | ConvertFrom-Json
    if (-not $release.isDraft -and -not $Force) {
        throw "Release $tag is already published at $($release.url). Re-run with -Force to overwrite assets people may already be downloading."
    }
    Write-Output "Reusing $(if ($release.isDraft) { 'draft' } else { 'PUBLISHED' }) release $tag"
} else {
    & gh release create $tag --draft --title "Mellow $tag" --notes "Draft for Mellow $version. Assets uploaded by scripts/publish-installer.ps1."
    if ($LASTEXITCODE -ne 0) { throw "Could not create the draft release $tag." }
    Write-Output "Created draft release $tag"
}

# --clobber so a rebuild replaces the assets instead of failing.
& gh release upload $tag $publishedInstaller $metadataPath --clobber
if ($LASTEXITCODE -ne 0) { throw "Could not upload assets to $tag." }

$url = (& gh release view $tag --json url | ConvertFrom-Json).url
Write-Output ""
Write-Output "Uploaded $assetName and latest.json to $tag"
Write-Output "Review and publish the draft: $url"
