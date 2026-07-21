param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/local/tmp/findmax",
    [int]$Width = 5760,
    [int]$Height = 4320,
    [int]$Runs = 30,
    [int]$LwsX = 16,
    [int]$LwsY = 16,
    [int]$Lws1d = 256,
    [int]$LwsOpt = 256,
    [int]$Nwg = 256,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-ocl-test\ocl_test_findmax"
$Kernels = @(
    (Join-Path $Root "findmax_orig_2d.cl"),
    (Join-Path $Root "findmax_baseline.cl"),
    (Join-Path $Root "findmax_opt.cl")
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
Write-Host " RUN: ocl_test_findmax  - findMax ONLY, no enhance"
Write-Host " Expect FIRST: SIZE SWEEP banner, then timed bench"
Write-Host " For pipeline: .\scripts\run_enhance.ps1"
Write-Host "============================================================"
Write-Host "Local exe: $Exe"
Write-Host "Exe mtime: $exeTime"

& $Hdc shell "mkdir -p $RemoteDir"
& $Hdc file send $Exe "$RemoteDir/ocl_test_findmax"
foreach ($k in $Kernels) {
    & $Hdc file send $k "$RemoteDir/$(Split-Path $k -Leaf)"
}

$cmd = "cd $RemoteDir && chmod +x ocl_test_findmax && ./ocl_test_findmax --orig findmax_orig_2d.cl --mine findmax_baseline.cl --opt findmax_opt.cl --width $Width --height $Height --runs $Runs --lwsx $LwsX --lwsy $LwsY --lws1d $Lws1d --lws-opt $LwsOpt --nwg $Nwg"
Write-Host $cmd
& $Hdc shell $cmd
