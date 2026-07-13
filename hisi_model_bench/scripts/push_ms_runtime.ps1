# Push MindSpore Lite + C++ runtime libs next to hdc-pushed bench binaries.
param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/vendor/camera"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"
if (-not (Test-Path $Hdc)) {
    throw "hdc not found: $Hdc"
}

$MsLib = Join-Path $OhosNative "sysroot\usr\lib\aarch64-linux-ohos\libmindspore_lite_ndk.so"
$CppShared = Join-Path $OhosNative "llvm\lib\aarch64-linux-ohos\c++\libc++_shared.so"

foreach ($lib in @($MsLib, $CppShared)) {
    if (-not (Test-Path $lib)) {
        throw "Missing runtime lib: $lib"
    }
    $name = Split-Path $lib -Leaf
    & $Hdc file send $lib "$RemoteDir/$name"
    Write-Host "Pushed $name"
}

return "export LD_LIBRARY_PATH=$RemoteDir:`$LD_LIBRARY_PATH"
