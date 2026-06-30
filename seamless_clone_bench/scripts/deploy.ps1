# Push benchmark + runtime libs to Huawei phone and run via hdc.
param(
    [string]$OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native",
    [string]$OpenCvOhosDir = "",
    [string]$RemoteDir = "/data/vendor/camera"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\ohos-arm64"

$cfg = & (Join-Path $PSScriptRoot "load_config.ps1") @{
    OhosNative = $OhosNative
    RemoteDir = $RemoteDir
}
$OhosNative = if ($cfg.OhosNative) { $cfg.OhosNative } else {
    "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
if ($cfg.RemoteDir) { $RemoteDir = $cfg.RemoteDir }
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

if ([string]::IsNullOrWhiteSpace($OpenCvOhosDir)) {
    $OpenCvOhosDir = Join-Path $Root "third_party\opencv-ohos-arm64"
}

$Exe = Join-Path $BuildDir "seamless_clone_bench"
if (-not (Test-Path $Exe)) {
    Write-Host "Executable not found. Run: .\scripts\build.ps1"
    exit 1
}
if (-not (Test-Path $Hdc)) {
    throw "hdc not found: $Hdc"
}

& $Hdc shell "mkdir -p $RemoteDir/out"
& $Hdc file send $Exe "$RemoteDir/seamless_clone_bench"

$LibDir = Join-Path $OpenCvOhosDir "lib"
Get-ChildItem $LibDir -Filter "libopencv_*.so" | ForEach-Object {
    & $Hdc file send $_.FullName "$RemoteDir/$($_.Name)"
}

$CppShared = Join-Path $OhosNative "llvm\lib\aarch64-linux-ohos\c++\libc++_shared.so"
if (Test-Path $CppShared) {
    & $Hdc file send $CppShared "$RemoteDir/libc++_shared.so"
}

Write-Host "Running on device (images -> $RemoteDir/out) ..."
& $Hdc shell "cd $RemoteDir && chmod +x seamless_clone_bench && export LD_LIBRARY_PATH=$RemoteDir:`$LD_LIBRARY_PATH && ./seamless_clone_bench && ls -la out"
Write-Host ""
Write-Host "Images on device: $RemoteDir/out/"
Write-Host "Pull to PC:       .\scripts\pull_results.ps1"
