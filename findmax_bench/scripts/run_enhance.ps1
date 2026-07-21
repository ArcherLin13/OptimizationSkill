param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/local/tmp/findmax",
    [int]$Width = 5760,
    [int]$Height = 4320,
    [int]$Runs = 30,
    [int]$LwsX = 16,
    [int]$LwsY = 16,
    [int]$LwsOpt = 256,
    [int]$Nwg = 256
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-ocl-test\ocl_test_enhance"
$Kernels = @(
    (Join-Path $Root "findmax_orig_2d.cl"),
    (Join-Path $Root "enhance_brightness.cl"),
    (Join-Path $Root "findmax_enhance_fused.cl")
)

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

foreach ($p in @($Exe) + $Kernels) {
    if (-not (Test-Path -LiteralPath $p)) {
        Write-Host "Missing: $p"
        Write-Host "Build: .\scripts\build_ohos.ps1"
        exit 1
    }
}

$targets = & $Hdc list targets 2>&1 | Out-String
if ($targets -match "Empty") {
    Write-Host "No hdc device connected."
    exit 1
}

Write-Host "============================================================"
Write-Host " RUN: ocl_test_enhance  (findMax + enhanceBrightness pipeline)"
Write-Host " NOT ocl_test_findmax   (that one is findmax-only)"
Write-Host "============================================================"

& $Hdc shell "mkdir -p $RemoteDir"
& $Hdc file send $Exe "$RemoteDir/ocl_test_enhance"
foreach ($k in $Kernels) {
    & $Hdc file send $k "$RemoteDir/$(Split-Path $k -Leaf)"
}

$cmd = "cd $RemoteDir && chmod +x ocl_test_enhance && ./ocl_test_enhance --findmax findmax_orig_2d.cl --enhance enhance_brightness.cl --fused findmax_enhance_fused.cl --width $Width --height $Height --runs $Runs --lwsx $LwsX --lwsy $LwsY --lws-opt $LwsOpt --nwg $Nwg"
Write-Host $cmd
& $Hdc shell $cmd
