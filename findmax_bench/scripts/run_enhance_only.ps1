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
    [double]$Max = 0.25,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-ocl-test\ocl_test_enhance_only"
$Kernels = @(
    (Join-Path $Root "enhance_brightness.cl"),
    (Join-Path $Root "enhance_brightness_2d_v4.cl"),
    (Join-Path $Root "enhance_brightness_opt.cl"),
    (Join-Path $Root "enhance_brightness_opt_h8.cl"),
    (Join-Path $Root "enhance_brightness_opt_h16.cl")
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
Write-Host " RUN: enhance ONLY"
Write-Host " 2d_1px | 2d_v4 | 1d_h4 | 1d_h8 | 1d_h16"
Write-Host "============================================================"
Write-Host "Local exe: $Exe"
Write-Host "Exe mtime: $exeTime"

& $Hdc shell "mkdir -p $RemoteDir"
& $Hdc file send $Exe "$RemoteDir/ocl_test_enhance_only"
foreach ($k in $Kernels) {
    & $Hdc file send $k "$RemoteDir/$(Split-Path $k -Leaf)"
}

$cmd = "cd $RemoteDir && chmod +x ocl_test_enhance_only && ./ocl_test_enhance_only --2d enhance_brightness.cl --2dv4 enhance_brightness_2d_v4.cl --h4 enhance_brightness_opt.cl --h8 enhance_brightness_opt_h8.cl --h16 enhance_brightness_opt_h16.cl --width $Width --height $Height --runs $Runs --lwsx $LwsX --lwsy $LwsY --lws-opt $LwsOpt --nwg $Nwg --max $Max"
Write-Host $cmd
& $Hdc shell $cmd
