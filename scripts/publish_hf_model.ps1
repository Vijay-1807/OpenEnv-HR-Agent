# Publishes inference LoRA files to Hugging Face model repo Vijay-1807/OpenEnv-HR-Agent
# Requires: pip install huggingface_hub[cli]  AND  hf auth login  OR  env HF_TOKEN
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Src = Join-Path $Root "qwen-hr-agent-trained"
$Bundle = Join-Path $Root "hf_model_bundle"
$RepoId = "Vijay-1807/OpenEnv-HR-Agent"

$files = @(
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja"
)
foreach ($f in $files) {
    $p = Join-Path $Src $f
    if (-not (Test-Path $p)) { throw "Missing $p" }
    Copy-Item -Force $p (Join-Path $Bundle $f)
}

# GRPO / RL training reward curve (repo root)
$curve = Join-Path $Root "reward_curve.png"
if (Test-Path $curve) {
    Copy-Item -Force $curve (Join-Path $Bundle "reward_curve.png")
    Write-Host "Included reward_curve.png in bundle."
} else {
    Write-Warning "reward_curve.png not found at repo root; skipping."
}

$readme = Join-Path $Bundle "README.md"
if (-not (Test-Path $readme)) { throw "Missing model card README at $readme" }

if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    Write-Host "Install CLI: pip install -U huggingface_hub[cli]"
    exit 1
}
# Credentials: `hf auth login` (cached) or env HF_TOKEN

Write-Host "Uploading $Bundle to $RepoId ..."
hf upload $RepoId $Bundle . --repo-type model
