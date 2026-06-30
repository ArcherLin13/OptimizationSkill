# Deploy __BENCH_NAME__ to device and run once.
param(
    [string]$OhosNative = "",
    [string]$ExePath = "",
    [string]$RemoteDir = "/data/vendor/camera",
    [string]$DeviceLibDir = "/chip_prod/lib64"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BenchExe = "__BENCH_NAME__"

$cfg = & (Join-Path $PSScriptRoot "load_config.ps1") @{
    OhosNative = $OhosNative
    DeviceLibDir = $DeviceLibDir
    RemoteDir = $RemoteDir
}
$OhosNative = if ($cfg.OhosNative) { $cfg.OhosNative } else {
    "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
if ($cfg.DeviceLibDir) { $DeviceLibDir = $cfg.DeviceLibDir }
if ($cfg.RemoteDir) { $RemoteDir = $cfg.RemoteDir }

$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"
if ([string]::IsNullOrWhiteSpace($ExePath)) {
    $ExePath = Join-Path $Root "build\ohos-arm64\$BenchExe"
}
if (-not (Test-Path $ExePath)) { throw "Build first: .\scripts\build.ps1" }
if (-not (Test-Path $Hdc)) { throw "hdc not found: $Hdc" }

& $Hdc shell "mkdir -p $RemoteDir/out"
& $Hdc file send $ExePath "$RemoteDir/$BenchExe"
& $Hdc shell "cd $RemoteDir && chmod +x $BenchExe && export LD_LIBRARY_PATH=$DeviceLibDir:`$LD_LIBRARY_PATH && ./$BenchExe"
Write-Host "Pull results: .\scripts\pull_results.ps1"
