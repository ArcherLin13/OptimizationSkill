# Pull visual export images from phone to ./results/
param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/local/tmp/seamless_clone_bench/out",
    [string]$LocalDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($LocalDir)) {
    $LocalDir = Join-Path $Root "results"
}

$cfg = & (Join-Path $PSScriptRoot "load_config.ps1") @{ OhosNative = $OhosNative }
$OhosNative = if ($cfg.OhosNative) { $cfg.OhosNative } else {
    "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"
if (-not (Test-Path $Hdc)) {
    throw "hdc not found: $Hdc"
}

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
Write-Host "Pulling $RemoteDir -> $LocalDir"
& $Hdc file recv $RemoteDir $LocalDir
Write-Host "Done. Open:"
Write-Host "  $LocalDir\grid_results.bmp"
Write-Host "  $LocalDir\grid_diff_x8.bmp"
