param(
    [string]$OhosNative = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\ohos-ocl-test"

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}

$Headers = Join-Path $Root "..\ocr_softmax_bench\third_party\OpenCL-Headers\CL\cl.h"
if (-not (Test-Path $Headers)) {
    Write-Host "Missing OpenCL headers: $Headers"
    exit 1
}

$Toolchain = Join-Path $OhosNative "build\cmake\ohos.toolchain.cmake"
$Cmake = Join-Path $OhosNative "build-tools\cmake\bin\cmake.exe"
$Ninja = Join-Path $OhosNative "build-tools\cmake\bin\ninja.exe"

if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}

$OpenClHeaders = (Resolve-Path (Join-Path $Root "..\ocr_softmax_bench\third_party\OpenCL-Headers")).Path -replace '\\','/'

$cmakeArgs = @(
    "-G", "Ninja",
    "-S", $Root,
    "-B", $BuildDir,
    "-DCMAKE_MAKE_PROGRAM=$Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
    "-DOHOS_ARCH=arm64-v8a",
    "-DOHOS_PLATFORM=OHOS",
    "-DOHOS_STL=c++_shared",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DOPENCL_HEADERS_DIR=$OpenClHeaders"
)

& $Cmake @cmakeArgs
& $Cmake --build $BuildDir --target ocl_test_planar test_roi_crop
Write-Host "Built: $BuildDir\ocl_test_planar"
Write-Host "Built: $BuildDir\test_roi_crop"
