param(
    [string]$OhosNative = "",
    [string]$RemoteDir = "/data/vendor/camera",
    [int]$LocalChar = 512,
    [int]$Runs = 20,
    [int]$Warmup = 3
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "build\ohos-ocl-test\ocl_test_ctc"
$DataDir = Join-Path $Root "testdata"

if ([string]::IsNullOrWhiteSpace($OhosNative)) {
    $OhosNative = "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native"
}
$Hdc = Join-Path (Split-Path $OhosNative -Parent) "toolchains\hdc.exe"

$KernelFiles = @(
    "softmax_ocr_opt_2d.cl",
    "softmax_ocr_fused_ctc_2d.cl"
)

foreach ($f in @($Exe, "$DataDir\logits.bin") + ($KernelFiles | ForEach-Object { Join-Path $Root $_ })) {
    if (-not (Test-Path $f)) {
        Write-Host "Missing: $f"
        Write-Host "Build: .\scripts\build_ohos_ocl_test.ps1"
        Write-Host "Data:  node generate_testdata.js"
        exit 1
    }
}

$targets = & $Hdc list targets 2>&1 | Out-String
if ($targets -match "Empty") {
    Write-Host "No hdc device connected."
    exit 1
}

& $Hdc shell "mkdir -p $RemoteDir/testdata"
& $Hdc file send $Exe "$RemoteDir/ocl_test_ctc"
foreach ($kf in $KernelFiles) {
    & $Hdc file send (Join-Path $Root $kf) "$RemoteDir/$kf"
}
& $Hdc file send "$DataDir\logits.bin" "$RemoteDir/testdata/logits.bin"

$cmd = "cd $RemoteDir && chmod +x ocl_test_ctc && ./ocl_test_ctc --data testdata --local-char $LocalChar --runs $Runs --warmup $Warmup"
Write-Host $cmd
& $Hdc shell $cmd
