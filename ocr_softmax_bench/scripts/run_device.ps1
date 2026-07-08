param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/vendor/camera"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-arm64\ocr_softmax_bench"

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

if (-not (Test-Path $Exe)) {
    Write-Host "Build first: .\scripts\build_ohos.ps1"
    exit 1
}

& $Hdc file send $Exe "$RemoteDir/ocr_softmax_bench"
& $Hdc shell "cd $RemoteDir && chmod +x ocr_softmax_bench && ./ocr_softmax_bench"
