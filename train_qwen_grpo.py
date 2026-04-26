import torch
import sys
import re
import traceback
import matplotlib.pyplot as plt
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

from src.env import HRHiringEnv
from src.models import HRAction

# Patch Unsloth for fast Reinforcement Learning
PatchFastRL("GRPO", FastLanguageModel)

# ──────────────────────────────────────────────
# Constants & Protocol Definition
# ──────────────────────────────────────────────
VALID_ACTIONS = [
    "read_inbox",
    "query_crm_database",
    "check_calendar",
    "evaluate_candidate",
    "send_email",
    "declare_done",
]

# Optimal actions per prompt key (Used for reward shaping/alignment)
SCENARIO_OPTIMAL_FIRST_ACTIONS = {
    "start_hiring":          "read_inbox",
    "candidate_applied":     "query_crm_database",
    "verify_violations":     "query_crm_database",
    "check_manager_calendar":"check_calendar",
    "candidate_passed":      "declare_done",
    "extract_skills":        "query_crm_database",
    "hiring_decision":       "evaluate_candidate",
    "blacklisted_candidate": "query_crm_database",
    "interview_slots":       "check_calendar",
    "begin_onboarding":      "evaluate_candidate",
    "send_rejection":        "send_email",
    "send_offer":            "send_email",
    "re_evaluate":           "evaluate_candidate",
    "crm_lookup":            "query_crm_database",
    "final_step":            "declare_done",
}

# The Training Scenario Bank (15 Unique States)
PROMPT_BANK = [
    ("senior_python_dev",   "Start the hiring process for the Senior Python Dev role. What is your first action?",              "start_hiring"),
    ("senior_python_dev",   "You read the inbox. Arjun Kumar applied for the Python Dev role. What is your next action?",       "candidate_applied"),
    ("senior_python_dev",   "You need to verify if Arjun Kumar has past policy violations. Which tool do you use?",             "verify_violations"),
    ("senior_python_dev",   "The hiring manager is John Smith. When is he free for an interview?",                              "check_manager_calendar"),
    ("senior_python_dev",   "Arjun has passed all background checks and evaluation. How do you complete the process?",          "candidate_passed"),
    ("senior_python_dev",   "What is the best way to extract Arjun's previous work history from the database?",                 "extract_skills"),
    ("senior_python_dev",   "You are deciding whether to hire or reject Arjun. You need to review their evaluation score.",     "hiring_decision"),
    ("senior_python_dev",   "A candidate was previously blacklisted in the CRM. What do you do before proceeding?",            "blacklisted_candidate"),
    ("fullstack_engineer", "Check the team's calendar for available interview slots for the Full-Stack Engineer position.",    "interview_slots"),
    ("fullstack_engineer", "Begin the onboarding evaluation for the selected Full-Stack Engineer candidate.",                  "begin_onboarding"),
    ("fullstack_engineer", "The Full-Stack Engineer candidate did not pass. Send them a polite rejection email.",              "send_rejection"),
    ("fullstack_engineer", "The Full-Stack Engineer candidate passed. Send them an offer letter.",                             "send_offer"),
    ("fullstack_engineer", "The evaluation score for the Full-Stack Engineer seems low. Re-evaluate before deciding.",         "re_evaluate"),
    ("fullstack_engineer", "You need to look up whether this candidate applied before. Check the CRM.",                        "crm_lookup"),
    ("fullstack_engineer", "All steps are complete for the Full-Stack Engineer hire. What is your final action?",             "final_step"),
]

SYSTEM_PROMPT = (
    "You are SentinelHire AI, an autonomous HR agent. "
    "Always reason step-by-step inside <scratchpad>...</scratchpad> "
    "and then emit exactly one tool call inside <action>...</action>. "
    f"Valid tools: {', '.join(VALID_ACTIONS)}."
)

# ──────────────────────────────────────────────
# Reward Function (The Brain)
# ──────────────────────────────────────────────
def openenv_reward_function(completions, prompts, **kwargs):
    """
    Definitive Reward Function v5 (Final Polish).
    Implements full 15-key State Injection and robust protocol grading.
    """
    rewards = []

    for completion, prompt in zip(completions, prompts):
        text = completion[0]["content"] if isinstance(completion, list) else completion

        # 1. XML Parsing
        action_match     = re.search(r"<action>(.*?)</action>",         text, re.IGNORECASE)
        scratchpad_match = re.search(r"<scratchpad>(.*?)</scratchpad>", text, re.IGNORECASE | re.DOTALL)

        if not action_match:
            rewards.append(-2.0)
            continue

        action_type = action_match.group(1).strip().lower()
        scratchpad  = scratchpad_match.group(1).strip() if scratchpad_match else ""

        # 2. Schema Check
        if action_type not in VALID_ACTIONS:
            rewards.append(-1.5)
            continue

        # 3. Scenario Recovery
        user_message = ""
        prompt_key   = "start_hiring"
        scenario_id  = "senior_python_dev"

        for turn in reversed(prompt):
            if turn.get("role") == "user":
                user_message = turn["content"]
                break

        for (sid, msg, pkey) in PROMPT_BANK:
            if msg.strip() == user_message.strip():
                scenario_id = sid
                prompt_key  = pkey
                break

        try:
            # 4. Environment Simulation with COMPLETE State Injection
            env = HRHiringEnv(scenario_id=scenario_id)
            env.reset()
            
            # Injection Logic: Pre-stepping the env to match the prompt's reality
            # Note: 'start_hiring' intentionally starts from Step 0 (no pre-steps needed).
            
            if prompt_key in ["candidate_applied", "verify_violations", "extract_skills", "blacklisted_candidate", "crm_lookup"]:
                env.step(HRAction(action_type="read_inbox", memory_scratchpad="System: Initializing candidate context from inbox."))
            
            elif prompt_key in ["check_manager_calendar", "interview_slots"]:
                env.step(HRAction(action_type="read_inbox", memory_scratchpad="System: Context read."))
                env.step(HRAction(action_type="query_crm_database", memory_scratchpad="System: Verified history in CRM."))
                
            elif prompt_key in ["candidate_passed", "hiring_decision"]:
                env.step(HRAction(action_type="read_inbox", memory_scratchpad="System: Context read."))
                env.step(HRAction(action_type="query_crm_database", memory_scratchpad="System: Verified history in CRM."))
                env.step(HRAction(action_type="evaluate_candidate", memory_scratchpad="System: Candidate profile evaluated."))
            
            elif prompt_key in ["begin_onboarding", "send_rejection", "send_offer", "re_evaluate"]:
                env.step(HRAction(action_type="read_inbox", memory_scratchpad="System: Context read."))
                env.step(HRAction(action_type="query_crm_database", memory_scratchpad="System: History verified."))
                env.step(HRAction(action_type="evaluate_candidate", memory_scratchpad="System: Profile evaluated."))
                env.step(HRAction(action_type="check_calendar", memory_scratchpad="System: Calendar slots confirmed."))
            
            elif prompt_key == "final_step":
                env.step(HRAction(action_type="read_inbox", memory_scratchpad="System: Context read."))
                env.step(HRAction(action_type="query_crm_database", memory_scratchpad="System: History verified."))
                env.step(HRAction(action_type="evaluate_candidate", memory_scratchpad="System: Profile evaluated."))
                env.step(HRAction(action_type="check_calendar", memory_scratchpad="System: Calendar slots confirmed."))
                env.step(HRAction(action_type="send_email", memory_scratchpad="System: Notification sent to candidate."))

            # AI's Action Execution
            action = HRAction(action_type=action_type, memory_scratchpad=scratchpad)
            obs    = env.step(action)

            reward_score = float(obs.reward) if obs.reward is not None else 0.0

            # 5. Reasoning Bonus (Keyword Scan)
            reasoning_keywords = ["blacklist", "crm", "calendar", "evaluate", "policy", "verify", "violation", "candidate"]
            scratchpad_lower    = scratchpad.lower()
            keyword_hits        = sum(1 for kw in reasoning_keywords if kw in scratchpad_lower)
            reward_score       += min(keyword_hits * 0.07, 0.35)  # Modest boost, stays balanced with alignment

            # 6. Alignment Bonus
            optimal = SCENARIO_OPTIMAL_FIRST_ACTIONS.get(prompt_key)
            if optimal and action_type == optimal:
                reward_score += 0.30

            # 7. Quality Gates
            # Length quality gate (encourages detail but prevents padding)
            word_count = len(text.split())
            if word_count < 20:
                reward_score -= 0.4
            elif 40 <= word_count <= 120:
                reward_score += 0.2

            # Scratchpad minimum (forces meaningful reasoning)
            if len(scratchpad) < 20:
                reward_score -= 0.4

            rewards.append(reward_score)

        except Exception as e:
            print(f"⚠️ Env Error in {prompt_key}: {e}")
            traceback.print_exc()
            rewards.append(-2.0)

    return rewards

# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
def plot_reward_curve(log_history: list, out_path: str = "reward_curve.png"):
    """Professional Grade Plotting."""
    candidate_keys = ["reward", "rewards/mean", "reward_mean", "train/reward"]
    steps, reward_vals = [], []
    
    for log in log_history:
        found = False
        for key in candidate_keys:
            if key in log:
                steps.append(log.get("step", len(steps)))
                reward_vals.append(log[key])
                found = True
                break
        if not found:
            for key, val in log.items():
                if "reward" in key.lower() and isinstance(val, (int, float)):
                    steps.append(log.get("step", len(steps)))
                    reward_vals.append(val)
                    break

    if not reward_vals:
        print("⚠️ No log data found.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, reward_vals, marker="o", color="#2563eb", label="SentinelHire Agent", linewidth=2)
    plt.axhline(y=-2.0, color="#ef4444", linestyle="--", label="Penalty Baseline")
    plt.axhline(y=0.0,  color="#6b7280", linestyle=":", label="Neutral Baseline")
    plt.title("SentinelHire AI: RL Training Progress", fontsize=14, fontweight="bold")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward Score")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"✅ Success! Chart saved to {out_path}")

# ──────────────────────────────────────────────
# Main Trainer
# ──────────────────────────────────────────────
def main():
    print("🚀 Initializing SentinelHire AI Final Gold Master Run...")

    # Setup
    max_seq_length = 2048
    lora_rank      = 16


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit = True,
        fast_inference = False,
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.6,
        dtype = torch.float16, # Reverting to standard FP16 to match the environment's current constraints
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # Dataset
    prompts = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}] for (_, msg, _) in PROMPT_BANK]
    dataset = Dataset.from_dict({"prompt": prompts * 14}) # ~210 samples

    # Config
    training_args = GRPOConfig(
        output_dir = "outputs/qwen-hr-agent",
        learning_rate = 2e-5,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        max_prompt_length = 512,
        max_completion_length = 256,
        num_generations = 4,
        max_steps = 200, 
        logging_steps = 5,
        save_steps = 50,
        bf16 = False,
        fp16 = True,
        temperature = 0.9, # Added temperature to prevent reward collapse/identical completions
    )

    trainer = GRPOTrainer(
        model = model,
        reward_funcs = [openenv_reward_function],
        args = training_args,
        train_dataset = dataset,
    )

    # Train
    print("🔥 Gold Master Training in progress...")
    trainer.train()

    # Finish
    plot_reward_curve(trainer.state.log_history)
    
    print("💾 Saving model...")
    model.save_pretrained("qwen-hr-agent-trained")
    tokenizer.save_pretrained("qwen-hr-agent-trained")
    
    # Run Final Inference Test (to Terminal)
    run_inference_test(model, tokenizer)

    # Save Inference Output to file for Hackathon Submission
    with open("inference_test_output.txt", "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        try:
            sys.stdout = f
            run_inference_test(model, tokenizer)
        finally:
            sys.stdout = original_stdout
    print("📝 Inference result saved to inference_test_output.txt")
    
    print("✅ PROJECT COMPLETE: Final Gold Master model is ready.")

def run_inference_test(model, tokenizer):
    """Prove the model actually learned — run it on a held-out prompt."""
    print("\n🧠 RUNNING FINAL INFERENCE TEST...")
    FastLanguageModel.for_inference(model)
    
    test_prompt = "You've just received an application from Arjun Mehta for the Python Dev role. What is your first step to ensure company policy compliance?"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")
    
    outputs = model.generate(
        input_ids = inputs,
        max_new_tokens = 256,
        temperature = 0.5,
        top_p = 0.9,
    )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print(f"PROMPT: {test_prompt}")
    print("-" * 50)
    print(f"AGENT RESPONSE:\n{decoded}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
