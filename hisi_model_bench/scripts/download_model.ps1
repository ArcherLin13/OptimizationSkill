# Download official MindSpore Lite tiny model (MNIST, ~13 KB)
# Source: https://download.mindspore.cn/model_zoo/official/lite/mnist_lite/mnist.ms

param(
    [string]$Out = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Out)) {
    $Out = Join-Path $Root "testdata\tiny.ms"
}

New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null

$models = @(
    @{
        Name = "add (1 KB, recommended for bench)"
        Url  = "https://download.mindspore.cn/model_zoo/official/lite/quick_start/add.ms"
        File = $Out
    },
    @{
        Name = "mnist (13 KB)"
        Url  = "https://download.mindspore.cn/model_zoo/official/lite/mnist_lite/mnist.ms"
        File = Join-Path $Root "testdata\mnist.ms"
    },
    @{
        Name = "mobilenetv2 (13 MB, slower load test)"
        Url  = "https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms"
        File = Join-Path $Root "testdata\mobilenetv2.ms"
    }
)

Write-Host "Downloading add tiny model -> $Out"
Invoke-WebRequest -Uri $models[0].Url -OutFile $models[0].File -UseBasicParsing
$kb = [math]::Round((Get-Item $models[0].File).Length / 1024, 1)
Write-Host "Done: tiny.ms ($kb KB)"
Write-Host ""
Write-Host "Optional larger model for load-stress test:"
Write-Host "  $($models[1].Url)"
Write-Host "  -> testdata\mobilenetv2.ms (run: Invoke-WebRequest manually)"
