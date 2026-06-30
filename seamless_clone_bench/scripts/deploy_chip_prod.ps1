# Deploy benchmark using OpenCV already on device (/chip_prod/lib64).
param(
    [string]$OhosNative = "",
    [string]$ExePath = "",
    [string]$RemoteDir = "/data/local/tmp/seamless_clone_bench",
    [string]$DeviceLibDir = "/chip_prod/lib64"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$cfg = & (Join-Path $PSScriptRoot "load_config.ps1") @{
    OhosNative = $OhosNative
    DeviceLibDir = $DeviceLibDir
}
$OhosNative = if ($cfg.OhosNative) { $cfg.OhosNative } else {
    "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
if ($cfg.DeviceLibDir) { $DeviceLibDir = $cfg.DeviceLibDir }

$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

if ([string]::IsNullOrWhiteSpace($ExePath)) {
    $candidates = @(
        (Join-Path $Root "bin\arm64-v8a\seamless_clone_bench"),
        (Join-Path $Root "build\ohos-arm64\seamless_clone_bench")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $ExePath = $c; break }
    }
}

if (-not (Test-Path $ExePath)) {
    Write-Host "Executable not found. Build first or pass -ExePath"
    exit 1
}
if (-not (Test-Path $Hdc)) {
    throw "hdc not found: $Hdc"
}

Write-Host "Checking NEEDED libs ..."
& $Hdc shell "readelf -d $RemoteDir/seamless_clone_bench 2>/dev/null | grep NEEDED || true"
$readelf = Join-Path $OhosNative "llvm\bin\llvm-readelf.exe"
if (Test-Path $readelf) {
    & $readelf -d $ExePath | Select-String "NEEDED"
}

Write-Host "Device OpenCV libs in $DeviceLibDir :"
& $Hdc shell "ls -la $DeviceLibDir/libopencv_*.so* 2>/dev/null || echo '(cannot list)'"

& $Hdc shell "mkdir -p $RemoteDir/out"
& $Hdc file send $ExePath "$RemoteDir/seamless_clone_bench"

Write-Host "Running with LD_LIBRARY_PATH=$DeviceLibDir ..."
& $Hdc shell "cd $RemoteDir && chmod +x seamless_clone_bench && export LD_LIBRARY_PATH=$DeviceLibDir:`$LD_LIBRARY_PATH && ./seamless_clone_bench --export $RemoteDir/out"
Write-Host ""
Write-Host "Images on device: $RemoteDir/out/"
Write-Host "Pull to PC:       .\scripts\pull_results.ps1"
