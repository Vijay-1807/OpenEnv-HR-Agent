"""
SentinelHire agent: optional LLM (PEFT) + reliable protocol-following fallback.

The LoRA adapter must be a valid safetensors file. OneDrive / interrupted writes
often corrupt `adapter_model.safetensors` (safetensors header error). If load
fails, we fall back to a deterministic heuristic so Streamlit and demos still work.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.env import HRHiringEnv


def _adapter_safetensors_ok(model_path: str) -> bool:
    path = os.path.join(model_path, "adapter_model.safetensors")
    if not os.path.isfile(path):
        return False
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
        return len(keys) > 0
    except Exception:
        return False


def _read_adapter_base_model(model_path: str) -> Optional[str]:
    cfg = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(cfg):
        return None
    try:
        with open(cfg, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("base_model_name_or_path")
    except Exception:
        return None


def _resolve_adapter_directory(model_path: str) -> tuple[str, Optional[str]]:
    """
    Prefer a directory whose adapter_model.safetensors passes safetensors parse.

    Order:
      1. SENTINEL_ADAPTER_PATH (absolute or relative folder with LoRA files)
      2. model_path argument (default ./qwen-hr-agent-trained)
      3. SENTINEL_ADAPTER_REPO — huggingface.co repo id; snapshot_download to cache
    """
    paths_to_try: list[str] = []
    env_override = os.environ.get("SENTINEL_ADAPTER_PATH")
    if env_override:
        paths_to_try.append(os.path.abspath(env_override))
    paths_to_try.append(os.path.abspath(model_path))

    for p in paths_to_try:
        if _adapter_safetensors_ok(p):
            return p, None

    hub_err: Optional[str] = None
    repo = os.environ.get("SENTINEL_ADAPTER_REPO")
    if repo:
        try:
            from huggingface_hub import snapshot_download

            cached = snapshot_download(repo_id=repo)
            ap = os.path.abspath(cached)
            if _adapter_safetensors_ok(ap):
                return ap, None
            hub_err = (
                f"SENTINEL_ADAPTER_REPO={repo!r} downloaded but "
                "adapter_model.safetensors is missing or invalid."
            )
        except Exception as e:
            hub_err = f"SENTINEL_ADAPTER_REPO={repo!r} download failed: {e}"

    return paths_to_try[-1], hub_err


class SentinelAgent:
    """
    LLM agent when weights load; otherwise `backend == \"heuristic\"`.
    Call `reset_episode()` whenever the environment starts a new episode.
    """

    def __init__(
        self,
        model_path: str = "./qwen-hr-agent-trained",
        base_model: Optional[str] = None,
    ):
        self.backend = "heuristic"
        self.load_error: Optional[str] = None
        self.model = None
        self.tokenizer = None

        resolved, hub_err = _resolve_adapter_directory(model_path)
        self.model_path = resolved
        self.base_model = base_model or _read_adapter_base_model(
            self.model_path
        ) or "Qwen/Qwen2.5-3B-Instruct"

        # Quieter transformers on Windows
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

        if hub_err:
            print(f"[WARN] {hub_err}")

        if not _adapter_safetensors_ok(self.model_path):
            parts = [
                "LoRA adapter_model.safetensors is missing or corrupted.",
                "Fix: move project out of OneDrive, re-copy qwen-hr-agent-trained,",
                "re-run train_qwen_grpo.py export, or push a good adapter to the Hub",
                "and set SENTINEL_ADAPTER_REPO=<your>/<repo>.",
            ]
            if hub_err:
                parts.append(hub_err)
            self.load_error = " ".join(parts)
            print(f"[WARN] {self.load_error}")
            return

        self._try_load_llm()

    def reset_episode(self) -> None:
        """Call when HRHiringEnv.reset() runs so heuristics stay in sync."""
        pass  # heuristic uses env._decision_log each step; no internal state

    def _try_load_llm(self) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import PeftModel

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        except Exception:
            # e.g. HF Space git push limit: omit tokenizer.json and load from base Hub id
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model, trust_remote_code=True
                )
            except Exception as e:
                self.load_error = f"Tokenizer load failed: {e}"
                print(f"[WARN] {self.load_error}")
                return

        try:
            kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            base_id = (self.base_model or "").lower()
            # Hub repos like unsloth/...-bnb-4bit ship with quantization_config already.
            # Do not pass BitsAndBytesConfig or torch_dtype for those: transformers may warn
            # ("equivalent parameters") and some Windows + bnb builds crash after download.
            already_quantized = any(
                s in base_id
                for s in (
                    "bnb-4bit",
                    "bnb_4bit",
                    "bnb4bit",
                    "bitsandbytes",
                    "-awq",
                    "-gptq",
                )
            )

            force_cpu = os.environ.get("SENTINEL_FORCE_CPU", "").lower() in (
                "1",
                "true",
                "yes",
            )
            use_cuda = torch.cuda.is_available() and not force_cpu
            if force_cpu and torch.cuda.is_available():
                print("[LOAD] SENTINEL_FORCE_CPU=1 — using CPU path (CUDA ignored for this run).")

            load_attempts: List[Dict[str, Any]] = []
            if use_cuda:
                if already_quantized:
                    # 1) Prefer all weights on GPU (no extra BnB kwargs).
                    load_attempts.append({**kwargs, "device_map": "auto"})
                    # 2) If VRAM is tight, accelerate may place layers on CPU/disk; for 4-bit
                    # loads transformers then requires explicit fp32 CPU offload support.
                    offload_dir = os.environ.get(
                        "SENTINEL_OFFLOAD_FOLDER",
                        os.path.join(tempfile.gettempdir(), "sentinelhire_hf_offload"),
                    )
                    os.makedirs(offload_dir, exist_ok=True)
                    load_attempts.append(
                        {
                            **kwargs,
                            "device_map": "auto",
                            "quantization_config": BitsAndBytesConfig(
                                load_in_4bit=True,
                                llm_int8_enable_fp32_cpu_offload=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                            ),
                            "offload_folder": offload_dir,
                        }
                    )
                else:
                    kw = {
                        **kwargs,
                        "device_map": "auto",
                        "torch_dtype": torch.float16,
                    }
                    try:
                        kw["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                    except Exception:
                        pass
                    load_attempts.append(kw)
            else:
                load_attempts.append(
                    {
                        **kwargs,
                        "device_map": {"": "cpu"},
                        "torch_dtype": torch.float32,
                    }
                )

            if os.environ.get("SENTINEL_BNB_CPU_OFFLOAD", "").lower() in (
                "1",
                "true",
                "yes",
            ) and len(load_attempts) > 1:
                load_attempts = [load_attempts[1], load_attempts[0]]

            last_err: Optional[BaseException] = None
            self.model = None
            for i, attempt_kw in enumerate(load_attempts, start=1):
                try:
                    print(
                        f"[LOAD] Loading base model ({i}/{len(load_attempts)}): {self.base_model}",
                        flush=True,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.base_model,
                        **attempt_kw,
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(f"[WARN] Base load attempt {i} failed: {e}", flush=True)
                    if i == len(load_attempts):
                        raise
            if self.model is None:
                raise RuntimeError("Model failed to load") from last_err

            print("[LOAD] Attaching LoRA adapters...")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            self.model.eval()
            self.backend = "llm"
            dev = next(self.model.parameters()).device
            print(f"[OK] SentinelHire LLM online (device={dev})")
        except Exception as e:
            self.model = None
            self.backend = "heuristic"
            self.load_error = f"LLM load failed: {e}"
            print(f"[WARN] {self.load_error} -- using heuristic agent.")

    def get_action(
        self,
        observation_text: str,
        env: Optional["HRHiringEnv"] = None,
    ) -> Dict[str, Any]:
        if self.backend == "llm" and self.model is not None and self.tokenizer:
            return self._llm_action(observation_text)
        if env is not None:
            return self._heuristic_action(env)
        return {
            "action_type": "read_inbox",
            "reasoning": "No env reference; defaulting to read_inbox.",
            "target_candidate": None,
            "scheduled_time": None,
            "memory_scratchpad": observation_text[:500],
        }

    def _llm_action(self, observation_text: str) -> Dict[str, Any]:
        import torch

        system_prompt = (
            "You are SentinelHire AI, an autonomous HR agent. "
            "Always reason step-by-step inside <scratchpad>...</scratchpad> "
            "and then emit exactly one tool call inside <action>...</action>. "
            "Valid tools: read_inbox, query_crm_database, check_calendar, "
            "evaluate_candidate, send_email, declare_done. "
            "For send_email include lines: target_candidate: <name> and "
            "scheduled_time: <slot from calendar>."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation_text},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )
        response = self.tokenizer.decode(
            outputs[0][len(input_ids[0]) :],
            skip_special_tokens=True,
        )
        parsed = self._parse_response(response)
        parsed["memory_scratchpad"] = parsed.get("reasoning", "")[:4000]
        return parsed

    def _parse_response(self, text: str) -> Dict[str, Any]:
        reasoning_match = re.search(
            r"<scratchpad>(.*?)</scratchpad>", text, re.DOTALL | re.IGNORECASE
        )
        reasoning = (
            reasoning_match.group(1).strip()
            if reasoning_match
            else "No reasoning provided."
        )

        action_match = re.search(
            r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE
        )
        action_type = (
            action_match.group(1).strip().lower()
            if action_match
            else "declare_done"
        )

        target = None
        for pattern in (
            r"target_candidate:\s*(.+?)(?:\n|$)",
            r"recipient:\s*(.+?)(?:\n|$)",
            r"candidate:\s*(.+?)(?:\n|$)",
        ):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                target = match.group(1).strip()
                break

        scheduled_time = None
        match = re.search(
            r"scheduled_time:\s*(.+?)(?:\n|$)", text, re.IGNORECASE
        )
        if match:
            scheduled_time = match.group(1).strip()

        return {
            "action_type": action_type,
            "reasoning": reasoning,
            "target_candidate": target,
            "scheduled_time": scheduled_time,
        }

    def _heuristic_action(self, env: "HRHiringEnv") -> Dict[str, Any]:
        """Same protocol as train_hr_agent.run_smart_agent_evaluation (local)."""
        scenario = env._scenario
        log = " ".join(env._decision_log)
        memory_parts: list[str] = []

        if "Read inbox" not in log:
            memory_parts.append("Step: read inbox for applicants.")
            return {
                "action_type": "read_inbox",
                "reasoning": "Gather all applications before CRM or decisions.",
                "target_candidate": None,
                "scheduled_time": None,
                "memory_scratchpad": " ".join(memory_parts),
            }

        for cand in scenario.candidate_names:
            if f"CRM check for {cand}" not in log:
                memory_parts.append(f"Must CRM-check {cand}.")
                return {
                    "action_type": "query_crm_database",
                    "reasoning": f"Compliance: verify {cand} in CRM before proceeding.",
                    "target_candidate": cand,
                    "scheduled_time": None,
                    "memory_scratchpad": " ".join(memory_parts),
                }

        crm_flags: Dict[str, str] = {}
        for cand in scenario.candidate_names:
            rec = scenario.get_crm_record(cand)
            if rec is None:
                crm_flags[cand] = "UNKNOWN"
                continue
            status_u = str(rec.get("status", "")).upper()
            flag_u = str(rec.get("flag", "")).upper()
            reason_u = str(rec.get("reason", "")).upper()
            if (
                "BLACKLIST" in status_u
                or "DO NOT REHIRE" in flag_u
                or "BLACKLIST" in reason_u
            ):
                crm_flags[cand] = "REJECT"
            elif any(
                x in reason_u or x in flag_u
                for x in (
                    "NON-COMPETE",
                    "LITIGATION",
                    "DATA BREACH",
                    "NON COMPETE",
                )
            ):
                crm_flags[cand] = "REJECT"
            else:
                crm_flags[cand] = "CLEAR"

        def _eligible(c: str) -> bool:
            if scenario.is_candidate_trap(c):
                return False
            return crm_flags.get(c) == "CLEAR"

        eligible = [c for c in scenario.candidate_names if _eligible(c)]

        for cand in eligible:
            if f"Evaluated {cand}" not in log:
                memory_parts.append(f"Evaluate {cand} (CRM clear).")
                return {
                    "action_type": "evaluate_candidate",
                    "reasoning": f"Skill match check for {cand} after CRM clearance.",
                    "target_candidate": cand,
                    "scheduled_time": None,
                    "memory_scratchpad": " ".join(memory_parts),
                }

        if "Checked calendar" not in log:
            memory_parts.append("Check hiring manager calendar for slots.")
            return {
                "action_type": "check_calendar",
                "reasoning": "Schedule only in an available slot.",
                "target_candidate": None,
                "scheduled_time": None,
                "memory_scratchpad": " ".join(memory_parts),
            }

        best = scenario.correct_hire
        if eligible:
            best = max(
                eligible,
                key=lambda c: scenario.get_candidate_score(c),
            )

        slot = scenario.calendar_slots[0] if scenario.calendar_slots else None

        if "Sent interview email" not in log and best and slot:
            memory_parts.append(f"Invite {best} for {slot}.")
            return {
                "action_type": "send_email",
                "reasoning": f"Send interview invite to {best} at {slot}.",
                "target_candidate": best,
                "scheduled_time": slot,
                "memory_scratchpad": " ".join(memory_parts),
            }

        memory_parts.append("Workflow complete.")
        return {
            "action_type": "declare_done",
            "reasoning": "All protocol steps done; finalize episode.",
            "target_candidate": None,
            "scheduled_time": None,
            "memory_scratchpad": " ".join(memory_parts),
        }
