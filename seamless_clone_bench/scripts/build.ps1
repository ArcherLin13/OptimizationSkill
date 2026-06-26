# Build seamless_clone_bench for HarmonyOS arm64-v8a.
param(
    [string]$OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native",
    [string]$OpenCvOhosDir = "",
    [int]$Jobs = 8
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\ohos-arm64"

if ([string]::IsNullOrWhiteSpace($OpenCvOhosDir)) {
    $OpenCvOhosDir = Join-Path $Root "third_party\opencv-ohos-arm64"
}

$Toolchain = Join-Path $OhosNative "build\cmake\ohos.toolchain.cmake"
$Cmake = Join-Path $OhosNative "build-tools\cmake\bin\cmake.exe"
$Ninja = Join-Path $OhosNative "build-tools\cmake\bin\ninja.exe"
$OpenCvConfig = Join-Path $OpenCvOhosDir "lib\cmake\opencv4\OpenCVConfig.cmake"

if (-not (Test-Path $OpenCvConfig)) {
    Write-Host "OpenCV for OHOS not found at: $OpenCvOhosDir"
    Write-Host "Run: .\scripts\build_opencv.ps1"
    exit 1
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$env:OPENCV_OHOS_DIR = $OpenCvOhosDir

Write-Host "Configuring benchmark ..."
& $Cmake -G Ninja `
    -S $Root `
    -B $BuildDir `
    -DCMAKE_MAKE_PROGRAM="$Ninja" `
    -DCMAKE_TOOLCHAIN_FILE="$Toolchain" `
    -DOHOS_ARCH=arm64-v8a `
    -DOHOS_PLATFORM=OHOS `
    -DOHOS_STL=c++_shared `
    -DCMAKE_BUILD_TYPE=Release `
    -DOpenCV_DIR="$OpenCvOhosDir\lib\cmake\opencv4"

Write-Host "Building benchmark ..."
& $Cmake --build $BuildDir --parallel $Jobs

$Exe = Join-Path $BuildDir "seamless_clone_bench"
Write-Host "Built: $Exe"
