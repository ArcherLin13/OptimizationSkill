param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/local/tmp/findmax",
    [int]$Width = 5760,
    [int]$Height = 4320,
    [int]$Runs = 30,
    [int]$LwsX = 16,
    [int]$LwsY = 16,
    [int]$LwsOpt = 256,
    [int]$Nwg = 256,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-ocl-test\ocl_test_enhance"
$Kernels = @(
    (Join-Path $Root "findmax_orig_2d.cl"),
    (Join-Path $Root "enhance_brightness.cl"),
    (Join-Path $Root "findmax_enhance_fused.cl"),
    (Join-Path $Root "findmax_opt.cl"),
    (Join-Path $Root "enhance_brightness_opt.cl")
)

if (-not $SkipBuild) {
    Write-Host "Building..."
    & (Join-Path $PSScriptRoot "build_ohos.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

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

$exeTime = (Get-Item -LiteralPath $Exe).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
Write-Host "============================================================"
Write-Host " RUN: ocl_test_enhance  - findMax + enhanceBrightness"
Write-Host " Expect FIRST: SIZE SWEEP for opt path C, then A/B/C bench"
Write-Host " A=baseline 2k  B=fused 1WG  C=opt 2k"
Write-Host "============================================================"
Write-Host "Local exe: $Exe"
Write-Host "Exe mtime: $exeTime"

& $Hdc shell "mkdir -p $RemoteDir"
& $Hdc file send $Exe "$RemoteDir/ocl_test_enhance"
foreach ($k in $Kernels) {
    & $Hdc file send $k "$RemoteDir/$(Split-Path $k -Leaf)"
}

$cmd = "cd $RemoteDir && chmod +x ocl_test_enhance && ./ocl_test_enhance --findmax findmax_orig_2d.cl --enhance enhance_brightness.cl --fused findmax_enhance_fused.cl --findmax-opt findmax_opt.cl --enhance-opt enhance_brightness_opt.cl --width $Width --height $Height --runs $Runs --lwsx $LwsX --lwsy $LwsY --lws-opt $LwsOpt --nwg $Nwg"
Write-Host $cmd
& $Hdc shell $cmd
