# Pull device OpenCV libs from phone for relinking benchmark.
param(
    [string]$OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native",
    [string]$DeviceLibDir = "/chip_prod/lib64",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $Root "device_libs\chip_prod_lib64"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Listing device libs ..."
& $Hdc shell "ls -la $DeviceLibDir/libopencv_*.so*"

foreach ($mod in @("core", "imgproc", "photo")) {
    $pattern = "libopencv_$mod.so"
    $remote = & $Hdc shell "ls $DeviceLibDir/${pattern}* 2>/dev/null | head -1"
    $remote = ($remote -replace "`r", "").Trim()
    if ([string]::IsNullOrWhiteSpace($remote)) {
        Write-Warning "Not found on device: $pattern*"
        continue
    }
    $name = Split-Path $remote -Leaf
    Write-Host "Pull $remote -> $OutDir\$name"
    & $Hdc file recv $remote (Join-Path $OutDir $name)
}

Write-Host "Pulled libs to: $OutDir"
Write-Host "Rebuild with:"
Write-Host "  .\scripts\build.ps1 -LinkDeviceOpenCV -OpenCvDeviceLibDir `"$OutDir`" -OpenCvDeviceIncludeDir `"<headers matching device opencv>`""
