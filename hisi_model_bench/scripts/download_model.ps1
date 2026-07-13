# Download official MindSpore Lite models for HarmonyOS device bench.
# mobilenetv2.ms is the same model used in HarmonyOS quick_start samples (~14 MB).

param(
    [ValidateSet("mobilenetv2", "add", "mnist", "all")]
    [string]$Model = "mobilenetv2"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$TestDir = Join-Path $Root "testdata"
New-Item -ItemType Directory -Force -Path $TestDir | Out-Null

$models = @{
  mobilenetv2 = @{
    Name = "mobilenetv2 (14 MB, recommended for HarmonyOS)"
    Url  = "https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms"
    File = Join-Path $TestDir "mobilenetv2.ms"
  }
  add = @{
    Name = "add (1 KB, may fail on some devices)"
    Url  = "https://download.mindspore.cn/model_zoo/official/lite/quick_start/add.ms"
    File = Join-Path $TestDir "tiny.ms"
  }
  mnist = @{
    Name = "mnist (13 KB)"
    Url  = "https://download.mindspore.cn/model_zoo/official/lite/mnist_lite/mnist.ms"
    File = Join-Path $TestDir "mnist.ms"
  }
}

function Download-One($key) {
    $m = $models[$key]
    Write-Host "Downloading $($m.Name) -> $($m.File)"
    Invoke-WebRequest -Uri $m.Url -OutFile $m.File -UseBasicParsing
    $kb = [math]::Round((Get-Item $m.File).Length / 1024, 1)
    Write-Host "Done: $(Split-Path $m.File -Leaf) ($kb KB)"
}

if ($Model -eq "all") {
    foreach ($k in @("mobilenetv2", "add", "mnist")) { Download-One $k }
} else {
    Download-One $Model
}

Write-Host ""
Write-Host "Device bench default: testdata\mobilenetv2.ms"
Write-Host "If NNRT still fails, try: .\scripts\run_device_dual.ps1 -Device cpu"
