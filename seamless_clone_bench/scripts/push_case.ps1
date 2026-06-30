# Push custom src/dst/mask PNGs to phone for visual export.
param(
    [string]$CaseDir = "",
    [string]$OhosNative = "",
    [string]$RemoteCaseDir = "/data/local/tmp/seamless_clone_bench/case"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($CaseDir)) {
    $CaseDir = Join-Path $Root "samples\app_case"
}

$cfg = & (Join-Path $PSScriptRoot "load_config.ps1") @{ OhosNative = $OhosNative }
$OhosNative = if ($cfg.OhosNative) { $cfg.OhosNative } else {
    "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

foreach ($name in @("src.bmp", "dst.bmp", "mask.bmp")) {
    $local = Join-Path $CaseDir $name
    if (-not (Test-Path $local)) {
        throw "Missing $local (put src.bmp dst.bmp mask.bmp in case folder)"
    }
}

& $Hdc shell "mkdir -p $RemoteCaseDir"
& $Hdc file send (Join-Path $CaseDir "src.bmp") "$RemoteCaseDir/src.bmp"
& $Hdc file send (Join-Path $CaseDir "dst.bmp") "$RemoteCaseDir/dst.bmp"
& $Hdc file send (Join-Path $CaseDir "mask.bmp") "$RemoteCaseDir/mask.bmp"
if (Test-Path (Join-Path $CaseDir "center.txt")) {
    & $Hdc file send (Join-Path $CaseDir "center.txt") "$RemoteCaseDir/center.txt"
}

Write-Host "Pushed case to $RemoteCaseDir"
Write-Host "Run on device:"
Write-Host "  ./seamless_clone_bench --images $RemoteCaseDir --export /data/local/tmp/seamless_clone_bench/out"
