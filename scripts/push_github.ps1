# Push OptimizationSkill to GitHub (user: conica)
param(
    [string]$User = "conica",
    [string]$Repo = "OptimizationSkill",
    [string]$Visibility = "public"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$TokenFile = Join-Path $Root ".secrets\github_token"

$env:Path = "C:\Program Files\Git\bin;C:\Program Files\GitHub CLI;" + $env:Path

$token = $env:GH_TOKEN
if ([string]::IsNullOrWhiteSpace($token) -and (Test-Path $TokenFile)) {
    $token = (Get-Content $TokenFile -Raw).Trim()
}

if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Host "No token found."
    Write-Host "Create file: $TokenFile"
    Write-Host "Put your GitHub PAT (ghp_...) on a single line, then re-run this script."
    exit 1
}

$env:GH_TOKEN = $token
Set-Location $Root

Write-Host "Checking GitHub user ..."
$login = gh api user --jq .login 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Token invalid or network error: $login"
    exit 1
}
Write-Host "Authenticated as: $login"

if (git remote get-url origin 2>$null) {
    Write-Host "Remote origin already exists."
} else {
    $flag = if ($Visibility -eq "private") { "--private" } else { "--public" }
    Write-Host "Creating repo $User/$Repo ..."
    gh repo create "$User/$Repo" $flag --source=. --remote=origin
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Repo may already exist, adding remote manually ..."
        git remote add origin "https://github.com/$User/$Repo.git"
    }
}

Write-Host "Pushing ..."
git push -u origin master
if ($LASTEXITCODE -ne 0) {
    git branch -M main 2>$null
    git push -u origin master:main
}

Write-Host "Done: https://github.com/$User/$Repo"
