---
license: apache-2.0
language:
  - en
tags:
  - peft
  - lora
  - qwen
  - hr
  - agent
  - openenv
base_model: unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit
---

# OpenEnv-HR-Agent (LoRA adapter)

PEFT LoRA adapter for **SentinelHire** / OpenEnv HR hiring agent demos.  
Base checkpoint: `unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit` (see `adapter_config.json`).

## Use in the app

Set:

```text
SENTINEL_ADAPTER_REPO=Vijay-1807/OpenEnv-HR-Agent
```

or download this repo snapshot and point `SENTINEL_ADAPTER_PATH` at the folder containing `adapter_model.safetensors`.

## Training

Trained with GRPO / Unsloth in this project (`train_qwen_grpo.py`).  
Code: [github.com/Vijay-1807/OpenEnv-HR-Agent](https://github.com/Vijay-1807/OpenEnv-HR-Agent)

Reward curve during RL training: see `reward_curve.png` in this repo.

## Limitations

Inference requires a **GPU** with enough VRAM for the 4-bit base plus adapter, or a configured CPU/offload path. See the GitHub README for Streamlit setup.
