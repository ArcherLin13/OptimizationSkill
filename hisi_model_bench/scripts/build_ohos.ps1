param(
    [string]$OhosNative = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\ohos-arm64"

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}

$Toolchain = Join-Path $OhosNative "build\cmake\ohos.toolchain.cmake"
$Cmake = Join-Path $OhosNative "build-tools\cmake\bin\cmake.exe"
$Ninja = Join-Path $OhosNative "build-tools\cmake\bin\ninja.exe"

if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}

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
    "-DOHOS_NATIVE=$OhosNative"
)

& $Cmake @cmakeArgs
& $Cmake --build $BuildDir --target ms_bench
Write-Host "Built: $BuildDir\ms_bench"
