param(
    [string]$OhosNative = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\ohos-arm64"

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}

Write-Host "Note: OHOS NDK may not ship OpenCL headers/libs."
Write-Host "If this fails, compile ocl_test_2d.cpp on your device toolchain with -lOpenCL"
Write-Host ""

$Toolchain = Join-Path $OhosNative "build\cmake\ohos.toolchain.cmake"
$Cmake = Join-Path $OhosNative "build-tools\cmake\bin\cmake.exe"
$Ninja = Join-Path $OhosNative "build-tools\cmake\bin\ninja.exe"

$cmakeArgs = @(
    "-G", "Ninja",
    "-S", (Join-Path $Root "device_test"),
    "-B", $BuildDir,
    "-DCMAKE_MAKE_PROGRAM=$Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
    "-DOHOS_ARCH=arm64-v8a",
    "-DOHOS_PLATFORM=OHOS",
    "-DOHOS_STL=c++_shared",
    "-DCMAKE_BUILD_TYPE=Release"
)

& $Cmake @cmakeArgs
& $Cmake --build $BuildDir --target ocl_test_2d
Write-Host "Built: $BuildDir\ocl_test_2d"
