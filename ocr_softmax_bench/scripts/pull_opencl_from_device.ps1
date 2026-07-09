param(
    [string]$OhosNative = "",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $Root "third_party\ohos"
}
if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$targets = & $Hdc list targets 2>&1 | Out-String
if ($targets -match "Empty") {
    Write-Host "Connect HarmonyOS device first (hdc list targets)."
    exit 1
}

$paths = @(
    "/vendor/lib64/libOpenCL.so",
    "/system/lib64/libOpenCL.so",
    "/vendor/lib/libOpenCL.so",
    "/system/lib/libOpenCL.so"
)

$found = $null
foreach ($p in $paths) {
    $check = & $Hdc shell "test -f $p && echo ok" 2>&1
    if ($check -match "ok") {
        $found = $p
        break
    }
}

if (-not $found) {
    Write-Host "libOpenCL.so not found on device. Search manually:"
    Write-Host "  hdc shell find /vendor /system -name libOpenCL.so 2>/dev/null"
    exit 1
}

$out = Join-Path $OutDir "libOpenCL.so"
Write-Host "Pulling $found -> $out"
& $Hdc file recv $found $out
Write-Host "Done. Build with:"
Write-Host "  .\scripts\build_ohos_ocl_test.ps1 -OpenCLLibrary $out"
