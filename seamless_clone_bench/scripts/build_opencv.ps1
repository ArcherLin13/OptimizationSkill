# Cross-compile minimal OpenCV for HarmonyOS arm64 (core + imgproc + photo).
param(
    [string]$OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native",
    [string]$OpenCvVersion = "4.9.0",
    [int]$Jobs = 8
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ThirdParty = Join-Path $Root "third_party"
$OpenCvSrc = Join-Path $ThirdParty "opencv-$OpenCvVersion"
$InstallDir = Join-Path $ThirdParty "opencv-ohos-arm64"
$BuildDir = Join-Path $Root "build\opencv-ohos-arm64"

$Toolchain = Join-Path $OhosNative "build\cmake\ohos.toolchain.cmake"
$Cmake = Join-Path $OhosNative "build-tools\cmake\bin\cmake.exe"
$Ninja = Join-Path $OhosNative "build-tools\cmake\bin\ninja.exe"

if (-not (Test-Path $Toolchain)) {
    throw "ohos.toolchain.cmake not found: $Toolchain"
}
if (-not (Test-Path $Cmake)) {
    throw "cmake not found: $Cmake"
}

New-Item -ItemType Directory -Force -Path $ThirdParty | Out-Null

if (-not (Test-Path $OpenCvSrc)) {
    Write-Host "Downloading OpenCV $OpenCvVersion ..."
    $ZipPath = Join-Path $ThirdParty "opencv-$OpenCvVersion.zip"
    if (-not (Test-Path $ZipPath)) {
        $Url = "https://github.com/opencv/opencv/archive/refs/tags/$OpenCvVersion.zip"
        Invoke-WebRequest -Uri $Url -OutFile $ZipPath
    }
    Expand-Archive -Path $ZipPath -DestinationPath $ThirdParty -Force
    $Extracted = Join-Path $ThirdParty "opencv-$OpenCvVersion"
    if (-not (Test-Path $Extracted)) {
        $Alt = Join-Path $ThirdParty "opencv-opencv-$OpenCvVersion"
        if (Test-Path $Alt) {
            Rename-Item $Alt $Extracted
        } else {
            $Alt2 = Get-ChildItem $ThirdParty -Directory | Where-Object { $_.Name -like "opencv*$OpenCvVersion*" } | Select-Object -First 1
            if ($Alt2) { Rename-Item $Alt2.FullName $Extracted }
        }
    }
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

Write-Host "Configuring OpenCV for OHOS arm64-v8a ..."
& $Cmake -G Ninja `
    -S $OpenCvSrc `
    -B $BuildDir `
    -DCMAKE_MAKE_PROGRAM="$Ninja" `
    -DCMAKE_TOOLCHAIN_FILE="$Toolchain" `
    -DOHOS_ARCH=arm64-v8a `
    -DOHOS_PLATFORM=OHOS `
    -DOHOS_STL=c++_shared `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX="$InstallDir" `
    -DBUILD_SHARED_LIBS=ON `
    -DBUILD_TESTS=OFF `
    -DBUILD_PERF_TESTS=OFF `
    -DBUILD_EXAMPLES=OFF `
    -DBUILD_opencv_apps=OFF `
    -DBUILD_opencv_java=OFF `
    -DBUILD_opencv_python=OFF `
    "-DBUILD_LIST=core,imgproc,photo" `
    -DWITH_IPP=OFF `
    -DWITH_OPENCL=OFF `
    -DWITH_CUDA=OFF `
    -DWITH_FFMPEG=OFF `
    -DWITH_GTK=OFF `
    -DWITH_QT=OFF `
    -DWITH_WEBP=OFF `
    -DWITH_JPEG=ON `
    -DWITH_PNG=ON `
    -DWITH_TIFF=OFF `
    -DWITH_OPENEXR=OFF `
    -DCPU_BASELINE=DETECT `
    -DCPU_DISPATCH=NEON

Write-Host "Building OpenCV ..."
& $Cmake --build $BuildDir --parallel $Jobs
& $Cmake --install $BuildDir

Write-Host "OpenCV installed to: $InstallDir"
Write-Host "Set OPENCV_OHOS_DIR=$InstallDir before building benchmark."
