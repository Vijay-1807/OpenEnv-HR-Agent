# Push a fresh single-commit snapshot of this repo to a Hugging Face Space.
# HF rejects Space git pushes with files > ~10 MB and some binaries (use HF Xet or omit).
# Prereq: hf auth login; remote Space must exist (e.g. Vijay-1807/sentinelhire-hr).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$SpaceRepo = if ($env:HF_SPACE_REPO) { $env:HF_SPACE_REPO } else { "https://huggingface.co/spaces/Vijay-1807/sentinelhire-hr" }
$Branch = "_hf_space_push"

Set-Location $Root
git checkout main
git branch -D $Branch 2>$null
git checkout --orphan $Branch
git add -A
# HF Space git rejects some binaries (e.g. PNG) without Xet — keep curve on HF model repo only
git rm -f --cached reward_curve.png 2>$null | Out-Null
$staged = git diff --cached --name-only
if (-not $staged) { Write-Host "Nothing staged — aborting push."; git checkout -f main; git branch -D $Branch 2>$null; exit 0 }
git commit -m "Space snapshot: SentinelHire ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))"
git push $SpaceRepo "${Branch}:main" --force
git checkout -f main
git branch -D $Branch
Write-Host "Done. Space: $SpaceRepo"
