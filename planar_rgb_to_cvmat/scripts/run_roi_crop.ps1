param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/local/tmp/planar_rgb",
    [int]$Width = 4096,
    [int]$Height = 3072,
    [int]$Boxes = 16,
    [int]$BoxW = 1000,
    [int]$BoxH = 150,
    [int]$Runs = 30
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-ocl-test\test_roi_crop"

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

if (-not (Test-Path $Exe)) {
    Write-Host "Missing: $Exe"
    Write-Host "Build: .\scripts\build_ohos.ps1"
    exit 1
}

$targets = & $Hdc list targets 2>&1 | Out-String
if ($targets -match "Empty") {
    Write-Host "No hdc device connected."
    exit 1
}

& $Hdc shell "mkdir -p $RemoteDir"
& $Hdc file send $Exe "$RemoteDir/test_roi_crop"

$cmd = "cd $RemoteDir && chmod +x test_roi_crop && ./test_roi_crop --width $Width --height $Height --boxes $Boxes --box-w $BoxW --box-h $BoxH --runs $Runs"
Write-Host $cmd
& $Hdc shell $cmd
