# Build seamless_clone_bench for HarmonyOS arm64-v8a.
param(
    [string]$OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native",
    [string]$OpenCvRoot = "",
    [switch]$UseBundledOpenCV,
    [string]$OpenCvOhosDir = "",
    [int]$Jobs = 8
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\ohos-arm64"

if ([string]::IsNullOrWhiteSpace($OpenCvRoot)) {
    $OpenCvRoot = Join-Path $Root "opencv"
}

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

if ($UseBundledOpenCV) {
    if ([string]::IsNullOrWhiteSpace($OpenCvOhosDir)) {
        $OpenCvOhosDir = Join-Path $Root "third_party\opencv-ohos-arm64"
    }
    $OpenCvConfig = Join-Path $OpenCvOhosDir "lib\cmake\opencv4\OpenCVConfig.cmake"
    if (-not (Test-Path $OpenCvConfig)) {
        Write-Host "OpenCV for OHOS not found at: $OpenCvOhosDir"
        Write-Host "Run: .\scripts\build_opencv.ps1"
        exit 1
    }
    $cmakeArgs += "-DUSE_DEVICE_OPENCV=OFF"
    $cmakeArgs += "-DOpenCV_DIR=$OpenCvOhosDir\lib\cmake\opencv4"
} else {
    $coreLib = Get-ChildItem $OpenCvRoot -Filter "libopencv_core.so*" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $coreLib) {
        $coreLib = Get-ChildItem (Join-Path $OpenCvRoot "lib") -Filter "libopencv_core.so*" -ErrorAction SilentlyContinue |
            Select-Object -First 1
    }
    if (-not $coreLib) {
        Write-Host "ERROR: put device OpenCV libs under: $OpenCvRoot"
        Write-Host "Expected: libopencv_core/imgcodecs/imgproc/photo.so*"
        Write-Host "And headers under: $OpenCvRoot\include (opencv4/)"
        exit 1
    }
    $cmakeArgs += "-DUSE_DEVICE_OPENCV=ON"
    $cmakeArgs += "-DOPENCV_ROOT=$OpenCvRoot"
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

Write-Host "Configuring benchmark ..."
& $Cmake @cmakeArgs

Write-Host "Building benchmark ..."
& $Cmake --build $BuildDir --parallel $Jobs

$Exe = Join-Path $BuildDir "seamless_clone_bench"
Write-Host "Built: $Exe"

$readelf = Join-Path $OhosNative "llvm\bin\llvm-readelf.exe"
if (Test-Path $readelf) {
    Write-Host "NEEDED libraries:"
    & $readelf -d $Exe | Select-String "NEEDED"
}
