param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "",
    [string]$LocalDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($LocalDir)) { $LocalDir = Join-Path $Root "results" }

$cfg = & (Join-Path $PSScriptRoot "load_config.ps1") @{ OhosNative = $OhosNative; RemoteDir = $RemoteDir }
$OhosNative = if ($cfg.OhosNative) { $cfg.OhosNative } else {
    "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
if ($cfg.RemoteDir) {
    $RemoteDir = if ($cfg.RemoteDir -match "/out$") { $cfg.RemoteDir } else { "$($cfg.RemoteDir)/out" }
} else {
    $RemoteDir = "/data/vendor/camera/out"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"
New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
& $Hdc file recv $RemoteDir $LocalDir
Write-Host "Pulled to $LocalDir"
