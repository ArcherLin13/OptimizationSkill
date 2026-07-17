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
if (-not (Test-Path -LiteralPath $Headers)) {
    Write-Host "Missing OpenCL headers: $Headers"
    exit 1
}

$Toolchain = Join-Path $OhosNative "build\cmake\ohos.toolchain.cmake"
$Cmake = Join-Path $OhosNative "build-tools\cmake\bin\cmake.exe"
$Ninja = Join-Path $OhosNative "build-tools\cmake\bin\ninja.exe"

foreach ($p in @($Toolchain, $Cmake, $Ninja)) {
    if (-not (Test-Path -LiteralPath $p)) {
        Write-Host "Missing: $p"
        Write-Host "Pass -OhosNative to your DevEco native SDK"
        exit 1
    }
}

if (Test-Path -LiteralPath $BuildDir) {
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$OpenClHeaders = (Resolve-Path (Join-Path $Root "..\ocr_softmax_bench\third_party\OpenCL-Headers")).Path
$ToolchainUnix = $Toolchain -replace '\\', '/'
$NinjaUnix = $Ninja -replace '\\', '/'
$OpenClHeadersUnix = $OpenClHeaders -replace '\\', '/'
$RootUnix = $Root -replace '\\', '/'
$BuildDirUnix = $BuildDir -replace '\\', '/'

Write-Host "CMake:  $Cmake"
Write-Host "Ninja:  $Ninja"
Write-Host "Build:  $BuildDir"

$cmakeArgs = @(
    "-G", "Ninja",
    "-S", $RootUnix,
    "-B", $BuildDirUnix,
    "-DCMAKE_MAKE_PROGRAM=$NinjaUnix",
    "-DCMAKE_TOOLCHAIN_FILE=$ToolchainUnix",
    "-DOHOS_ARCH=arm64-v8a",
    "-DOHOS_PLATFORM=OHOS",
    "-DOHOS_STL=c++_shared",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DOPENCL_HEADERS_DIR=$OpenClHeadersUnix"
)

& $Cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configure FAILED (exit $LASTEXITCODE). build.ninja was not generated."
    exit $LASTEXITCODE
}

$NinjaFile = Join-Path $BuildDir "build.ninja"
if (-not (Test-Path -LiteralPath $NinjaFile)) {
    Write-Host "CMake finished but build.ninja missing: $NinjaFile"
    exit 1
}

Push-Location $BuildDir
try {
    & $Ninja test_roi_crop
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Ninja build FAILED for test_roi_crop (exit $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
    Write-Host "Built: $BuildDir\test_roi_crop"

    & $Ninja ocl_test_planar
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: ocl_test_planar failed - ROI test binary is still ready."
        Write-Host "Next: .\scripts\run_roi_crop.ps1"
        exit 0
    }
    Write-Host "Built: $BuildDir\ocl_test_planar"
} finally {
    Pop-Location
}

Write-Host "Next: .\scripts\run_roi_crop.ps1"
