# Scaffold a new optimization bench from template/.
param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [string]$ParentDir = ""
)

$ErrorActionPreference = "Stop"
$FrameworkRoot = Split-Path -Parent $PSScriptRoot
$TemplateDir = Join-Path $FrameworkRoot "template"

if ([string]::IsNullOrWhiteSpace($ParentDir)) {
    $ParentDir = Split-Path -Parent $FrameworkRoot
}
$DestDir = Join-Path $ParentDir $Name

if (Test-Path $DestDir) {
    throw "Already exists: $DestDir"
}

$libName = ($Name -replace '_bench$', '')
if ([string]::IsNullOrWhiteSpace($libName)) { $libName = $Name }

Write-Host "Creating $DestDir (lib: $libName) ..."
Copy-Item -Recurse $TemplateDir $DestDir

$replace = @{
    "__BENCH_NAME__" = $Name
    "your_optimized" = $libName
}

Get-ChildItem $DestDir -Recurse -File | ForEach-Object {
    $text = [IO.File]::ReadAllText($_.FullName)
    $changed = $false
    foreach ($k in $replace.Keys) {
        if ($text.Contains($k)) {
            $text = $text.Replace($k, $replace[$k])
            $changed = $true
        }
    }
    if ($changed) { [IO.File]::WriteAllText($_.FullName, $text) }
}

$oldLib = Join-Path $DestDir "lib\your_optimized"
$newLib = Join-Path $DestDir "lib\$libName"
if (Test-Path $oldLib) {
    Rename-Item $oldLib $libName
}
if (Test-Path $newLib) {
    $pairs = @(
        @("your_optimized.cpp", "$libName.cpp"),
        @("your_optimized.h", "$libName.h")
    )
    foreach ($p in $pairs) {
        $src = Join-Path $newLib $p[0]
        $dst = Join-Path $newLib $p[1]
        if (Test-Path $src) {
            Move-Item -LiteralPath $src -Destination $dst -Force
        }
    }
}

Write-Host ""
Write-Host "Created: $DestDir"
Write-Host "Next:"
Write-Host "  cd $DestDir"
Write-Host "  copy scripts\config.local.ps1.example scripts\config.local.ps1"
Write-Host "  Edit lib\$libName\ and src\benchmark.cpp"
Write-Host "  .\scripts\build.ps1"
Write-Host "  .\scripts\deploy.ps1"
