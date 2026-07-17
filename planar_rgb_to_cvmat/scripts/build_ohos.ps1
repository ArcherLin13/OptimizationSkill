param(
    [string]$OhosNative = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root "build\ohos-ocl-test"

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}

$Toolchain = Join-Path $OhosNative "build\cmake\ohos.toolchain.cmake"
$Cmake = Join-Path $OhosNative "build-tools\cmake\bin\cmake.exe"
$Ninja = Join-Path $OhosNative "build-tools\cmake\bin\ninja.exe"

foreach ($p in @($Toolchain, $Cmake, $Ninja)) {
    if (-not (Test-Path -LiteralPath $p)) {
        Write-Host "Missing: $p"
        exit 1
    }
}

if (Test-Path -LiteralPath $BuildDir) {
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$ToolchainUnix = $Toolchain -replace '\\', '/'
$NinjaUnix = $Ninja -replace '\\', '/'
$RootUnix = $Root -replace '\\', '/'
$BuildDirUnix = $BuildDir -replace '\\', '/'

Write-Host "Build CPU ROI bench only (no OpenCL / dma_buf)"
Write-Host "Build: $BuildDir"

& $Cmake @(
    "-G", "Ninja",
    "-S", $RootUnix,
    "-B", $BuildDirUnix,
    "-DCMAKE_MAKE_PROGRAM=$NinjaUnix",
    "-DCMAKE_TOOLCHAIN_FILE=$ToolchainUnix",
    "-DOHOS_ARCH=arm64-v8a",
    "-DOHOS_PLATFORM=OHOS",
    "-DOHOS_STL=c++_shared",
    "-DCMAKE_BUILD_TYPE=Release"
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path -LiteralPath (Join-Path $BuildDir "build.ninja"))) {
    Write-Host "build.ninja missing"
    exit 1
}

Push-Location $BuildDir
try {
    & $Ninja test_roi_crop
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}

Write-Host "Built: $BuildDir\test_roi_crop"
Write-Host "Next:  .\scripts\run_roi_crop.ps1"
