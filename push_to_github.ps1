# Push this project to the "sharkTank" GitHub repo:
#   https://github.com/avapalmieri/sharkTank
#
# Safe to re-run: it only inits git / creates the remote if they don't
# already exist, and it never force-pushes.
#
# Usage (PowerShell, from inside the shark-tank-bot folder):
#   .\push_to_github.ps1
#   $env:COMMIT_MSG = "fix: typo"; .\push_to_github.ps1
#
# If PowerShell refuses to run this ("running scripts is disabled on this
# system"), run it as:
#   powershell -ExecutionPolicy Bypass -File .\push_to_github.ps1
#
# If your GitHub account uses SSH keys instead of HTTPS auth, change
# $RemoteUrl below to: git@github.com:avapalmieri/sharkTank.git

$RemoteUrl  = "https://github.com/avapalmieri/sharkTank"
$RemoteName = "origin"
$Branch     = "main"
$CommitMsg  = if ($env:COMMIT_MSG) { $env:COMMIT_MSG } else { "Update The Tank" }

Set-Location $PSScriptRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git is not installed. Get it from https://git-scm.com/download/win"
    exit 1
}

# --- .gitignore: keep secrets and junk out of the repo -------------------
if (-not (Test-Path ".gitignore")) {
    @"
__pycache__/
*.pyc
.venv/
venv/
.env
.DS_Store
"@ | Set-Content -Path ".gitignore" -Encoding utf8
    Write-Host "Created .gitignore"
}

# --- init repo if needed --------------------------------------------------
if (-not (Test-Path ".git")) {
    git init | Out-Null
    git branch -M $Branch
    Write-Host "Initialized git repo on branch '$Branch'"
}

# --- wire up the remote ---------------------------------------------------
$existingRemote = (git remote) -contains $RemoteName
if ($existingRemote) {
    git remote set-url $RemoteName $RemoteUrl
} else {
    git remote add $RemoteName $RemoteUrl
}
Write-Host "Remote '$RemoteName' -> $RemoteUrl"

# --- stage, commit ---------------------------------------------------------
git add -A

$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "Nothing to commit -- working tree matches the last commit."
} else {
    git commit -m $CommitMsg
    if ($LASTEXITCODE -ne 0) {
        Write-Error "git commit failed."
        exit 1
    }
}

# --- pull --rebase first if the remote branch already has history --------
git ls-remote --exit-code --heads $RemoteUrl $Branch | Out-Null
if ($LASTEXITCODE -eq 0) {
    git pull --rebase $RemoteName $Branch
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Rebase pull failed -- resolve conflicts, then run: git push $RemoteName $Branch"
        exit 1
    }
}

# --- push --------------------------------------------------------------
git push -u $RemoteName $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Error "git push failed -- check your GitHub credentials/permissions."
    exit 1
}

Write-Host "Pushed to $RemoteUrl ($Branch)"
