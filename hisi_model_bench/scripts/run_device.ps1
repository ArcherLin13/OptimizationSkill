param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/vendor/camera",
    [string]$ModelPath = "",
    [string]$Device = "nnrt"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-arm64\ms_bench"
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $Root "testdata\tiny.ms"
}

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

foreach ($f in @($Exe, $ModelPath)) {
    if (-not (Test-Path $f)) {
        Write-Host "Missing: $f"
        if (-not (Test-Path $ModelPath)) {
            Write-Host "Download model: .\scripts\download_model.ps1"
            Write-Host "Or copy your .ms model to testdata/tiny.ms"
        }
        Write-Host "Build: .\scripts\build_ohos.ps1"
        exit 1
    }
}

$targets = & $Hdc list targets 2>&1 | Out-String
if ($targets -match "Empty") {
    Write-Host "No hdc device connected."
    exit 1
}

$RemoteModel = "$RemoteDir/testdata/tiny.ms"
& $Hdc shell "mkdir -p $RemoteDir/testdata"
& $Hdc file send $Exe "$RemoteDir/ms_bench"
& $Hdc file send $ModelPath $RemoteModel

$cmd = "cd $RemoteDir && chmod +x ms_bench && ./ms_bench --model $RemoteModel --device $Device --runs 10"
Write-Host $cmd
& $Hdc shell $cmd
