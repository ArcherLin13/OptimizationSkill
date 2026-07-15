param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/local/tmp/planar_rgb",
    [int]$Width = 3840,
    [int]$Height = 2160,
    [int]$Runs = 20
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-ocl-test\ocl_test_planar"
$Kernel = Join-Path $Root "planar_rgb_to_cv32fc3.cl"

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

foreach ($f in @($Exe, $Kernel)) {
    if (-not (Test-Path $f)) {
        Write-Host "Missing: $f"
        Write-Host "Build: .\scripts\build_ohos.ps1"
        exit 1
    }
}

$targets = & $Hdc list targets 2>&1 | Out-String
if ($targets -match "Empty") {
    Write-Host "No hdc device connected."
    exit 1
}

& $Hdc shell "mkdir -p $RemoteDir"
& $Hdc file send $Exe "$RemoteDir/ocl_test_planar"
& $Hdc file send $Kernel "$RemoteDir/planar_rgb_to_cv32fc3.cl"

$cmd = "cd $RemoteDir && chmod +x ocl_test_planar && ./ocl_test_planar --kernel planar_rgb_to_cv32fc3.cl --width $Width --height $Height --runs $Runs"
Write-Host $cmd
& $Hdc shell $cmd
