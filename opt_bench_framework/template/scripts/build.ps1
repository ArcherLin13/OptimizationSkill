# Build __BENCH_NAME__ for HarmonyOS arm64-v8a.
param(
    [string]$OhosNative = "",
    [string]$OpenCvRoot = "",
    [int]$Jobs = 8
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\ohos-arm64"
$BenchExe = "__BENCH_NAME__"

$cfg = & (Join-Path $PSScriptRoot "load_config.ps1") @{
    OhosNative = $OhosNative
    OpenCvRoot = $OpenCvRoot
}
$OhosNative = if ($cfg.OhosNative) { $cfg.OhosNative } else {
    "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$OpenCvRoot = if ($cfg.OpenCvRoot) { $cfg.OpenCvRoot } else { Join-Path $Root "opencv" }

$Toolchain = Join-Path $OhosNative "build\cmake\ohos.toolchain.cmake"
$Cmake = Join-Path $OhosNative "build-tools\cmake\bin\cmake.exe"
$Ninja = Join-Path $OhosNative "build-tools\cmake\bin\ninja.exe"

$cmakeArgs = @(
    "-G", "Ninja",
    "-S", $Root,
    "-B", $BuildDir,
    "-DCMAKE_MAKE_PROGRAM=$Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
    "-DOHOS_ARCH=arm64-v8a",
    "-DOHOS_PLATFORM=OHOS",
    "-DOHOS_STL=c++_shared",
    "-DCMAKE_BUILD_TYPE=Release"
)

$coreLib = Get-ChildItem $OpenCvRoot -Filter "libopencv_core.so*" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $coreLib) {
    $coreLib = Get-ChildItem (Join-Path $OpenCvRoot "lib") -Filter "libopencv_core.so*" -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (-not $coreLib) {
    Write-Host "ERROR: put device OpenCV under $OpenCvRoot (libopencv_core/imgproc.so*)"
    exit 1
}
$cmakeArgs += "-DUSE_DEVICE_OPENCV=ON"
$cmakeArgs += "-DOPENCV_DEVICE_DIR=$OpenCvRoot"

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Write-Host "Configuring $BenchExe ..."
& $Cmake @cmakeArgs
Write-Host "Building ..."
& $Cmake --build $BuildDir --parallel $Jobs
Write-Host "Built: $(Join-Path $BuildDir $BenchExe)"
