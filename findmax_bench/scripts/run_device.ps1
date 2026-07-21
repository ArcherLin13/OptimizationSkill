param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/local/tmp/findmax",
    [int]$Width = 5760,
    [int]$Height = 4320,
    [int]$Runs = 30,
    [int]$Wg = 256,
    [int]$Lws = 256
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-ocl-test\ocl_test_findmax"
$Kernel = Join-Path $Root "findmax_baseline.cl"

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

foreach ($p in @($Exe, $Kernel)) {
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

& $Hdc shell "mkdir -p $RemoteDir"
& $Hdc file send $Exe "$RemoteDir/ocl_test_findmax"
& $Hdc file send $Kernel "$RemoteDir/findmax_baseline.cl"

$cmd = "cd $RemoteDir && chmod +x ocl_test_findmax && ./ocl_test_findmax --kernel findmax_baseline.cl --width $Width --height $Height --runs $Runs --wg $Wg --lws $Lws"
Write-Host $cmd
& $Hdc shell $cmd
