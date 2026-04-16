# ---------------------------------------------------------------------------
# Bootstrap this project as a fresh Git repository and push to GitHub (Windows).
#
# Usage (PowerShell):
#   # 1) Create an EMPTY repo on GitHub first (no README, no .gitignore, no LICENSE).
#   # 2) From the repo root:
#   .\scripts\bootstrap_git.ps1 -RemoteUrl "git@github.com:<user>/traffic-sign-recognition.git"
#
# Or with HTTPS:
#   .\scripts\bootstrap_git.ps1 -RemoteUrl "https://github.com/<user>/traffic-sign-recognition.git"
# ---------------------------------------------------------------------------
param(
    [Parameter(Mandatory = $true)][string]$RemoteUrl
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "pyproject.toml") -or -not (Test-Path "src/traffic_signs")) {
    Write-Error "Run this from the repository root (pyproject.toml not found)."
}

# Keep empty tracked directories alive.
foreach ($p in @("data/raw/.gitkeep", "data/interim/.gitkeep", "data/processed/.gitkeep", "checkpoints/.gitkeep")) {
    if (-not (Test-Path $p)) { New-Item -ItemType File -Path $p -Force | Out-Null }
}

if (-not (Test-Path ".git")) {
    git init -b main
    Write-Host "Initialised new repository on branch 'main'."
} else {
    Write-Host "Existing .git directory found — reusing it."
}

git config core.autocrlf true | Out-Null

git add .
git commit -m "chore: initial public import of traffic-signs project" -m "Refactored from a single Jupyter notebook into a library, CLI and test suite. See CHANGELOG.md."

$remotes = git remote
if ($remotes -match "^origin$") {
    git remote set-url origin $RemoteUrl
} else {
    git remote add origin $RemoteUrl
}

git branch -M main

Write-Host ""
Write-Host "About to push to: $RemoteUrl"
Write-Host "If this is wrong, Ctrl-C now. Otherwise press Enter."
Read-Host | Out-Null
git push -u origin main
Write-Host "`n✓ Pushed."
