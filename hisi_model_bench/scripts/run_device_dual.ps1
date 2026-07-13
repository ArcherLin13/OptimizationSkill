param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/vendor/camera",
    [string]$ModelA = "",
    [string]$ModelB = "",
    [string]$Device = "nnrt",
    [int]$Runs = 20,
    [switch]$SkipRuntimeLibs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-arm64\ms_dual_bench"
if ([string]::IsNullOrWhiteSpace($ModelA)) {
    $ModelA = Join-Path $Root "testdata\mobilenetv2.ms"
}
if ([string]::IsNullOrWhiteSpace($ModelB)) {
    $ModelB = $ModelA
}

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

if (-not (Test-Path $Exe)) {
    Write-Host "Missing: $Exe"
    Write-Host "Build: .\scripts\build_ohos.ps1"
    exit 1
}
if (-not (Test-Path $ModelA)) {
    Write-Host "Missing model A: $ModelA"
    Write-Host "Download: .\scripts\download_model.ps1"
    exit 1
}
if ($ModelB -ne $ModelA -and -not (Test-Path $ModelB)) {
    Write-Host "Missing model B: $ModelB"
    exit 1
}

$targets = & $Hdc list targets 2>&1 | Out-String
if ($targets -match "Empty") {
    Write-Host "No hdc device connected."
    exit 1
}

& $Hdc shell "mkdir -p $RemoteDir/testdata"
& $Hdc file send $Exe "$RemoteDir/ms_dual_bench"
& $Hdc file send $ModelA "$RemoteDir/testdata/model_a.ms"
if ($ModelB -eq $ModelA) {
    $remoteB = "testdata/model_a.ms"
} else {
    & $Hdc file send $ModelB "$RemoteDir/testdata/model_b.ms"
    $remoteB = "testdata/model_b.ms"
}

$ldPath = ""
if (-not $SkipRuntimeLibs) {
    $ldPath = & (Join-Path $PSScriptRoot "push_ms_runtime.ps1") -OhosNative $OhosNative -RemoteDir $RemoteDir
}

$cmd = "cd $RemoteDir && chmod +x ms_dual_bench && $ldPath ./ms_dual_bench --model-a testdata/model_a.ms --model-b $remoteB --device $Device --runs $Runs"
Write-Host $cmd
& $Hdc shell $cmd
