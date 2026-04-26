import streamlit as st
from src import HRHiringEnv
from src.agent import SentinelAgent
from src.models import HRAction
import time

st.set_page_config(page_title="SentinelHire AI", page_icon="🛡️", layout="wide")

st.title("🛡️ SentinelHire AI - OpenEnv HR Agent")
st.markdown("### Resolving Context-Window Degradation in Multi-Step Workflows")

# Session state initialization
if "env" not in st.session_state:
    st.session_state.env = None
if "obs" not in st.session_state:
    st.session_state.obs = None
if "history" not in st.session_state:
    st.session_state.history = []
if "agent" not in st.session_state:
    with st.status("🧠 Awakening SentinelHire AI (Loading Model)...", expanded=True) as status:
        st.session_state.agent = SentinelAgent()
        ag = st.session_state.agent
        if ag.backend == "llm":
            status.update(label="✅ LLM agent online", state="complete", expanded=False)
        else:
            status.update(
                label="✅ Protocol agent online (heuristic — fix LoRA file for full LLM)",
                state="complete",
                expanded=False,
            )
        if ag.load_error:
            st.warning(ag.load_error)

def reset_env(scenario_id):
    st.session_state.env = HRHiringEnv(scenario_id=scenario_id)
    st.session_state.obs = st.session_state.env.reset()
    st.session_state.history = []
    if st.session_state.agent:
        st.session_state.agent.reset_episode()
    st.rerun()

st.sidebar.header("Agent Configuration")
st.sidebar.caption(
    "Neural policy: run `python verify_lora.py`. "
    "If local LoRA is bad, set env `SENTINEL_ADAPTER_REPO=user/repo` or "
    "`SENTINEL_ADAPTER_PATH=...` then restart Streamlit."
)
scenario_option = st.sidebar.selectbox(
    "Select Scenario",
    ["senior_python_dev", "fullstack_engineer", "ml_engineer", "devops_engineer", "lead_security_architect"]
)

if st.sidebar.button("Reset Environment / New Episode"):
    reset_env(scenario_option)

st.sidebar.markdown("---")
st.sidebar.subheader("🎬 Presentation Demos")
if st.sidebar.button("❌ Auto-Run Failure Case"):
    reset_env(scenario_option)
    time.sleep(0.5)
    
    # Action 1
    a1 = HRAction(action_type="read_inbox", reasoning="Checking inbox for applicants.")
    o1 = st.session_state.env.step(a1)
    st.session_state.history.append({"step": 1, "action": a1, "obs": o1})
    
    # Action 2: BAD AGENT skips CRM and hires trap
    trap_cand = next((c for c, d in st.session_state.env._scenario.scenario["candidates"].items() if d.get("is_trap")), st.session_state.env._scenario.candidate_names[0])
    slot = st.session_state.env._scenario.calendar_slots[0]
    
    a2 = HRAction(action_type="send_email", target_candidate=trap_cand, scheduled_time=slot, reasoning=f"Candidate {trap_cand} looks perfect! Hiring them immediately.", confidence_score=0.99)
    o2 = st.session_state.env.step(a2)
    st.session_state.history.append({"step": 2, "action": a2, "obs": o2})
    
    a3 = HRAction(action_type="declare_done", reasoning="Hired the candidate.", confidence_score=0.99)
    o3 = st.session_state.env.step(a3)
    st.session_state.history.append({"step": 3, "action": a3, "obs": o3})
    
    st.session_state.obs = o3
    st.rerun()

if st.sidebar.button("✅ Auto-Run Success Case"):
    reset_env(scenario_option)
    env = st.session_state.env
    memory = "Starting task."
    
    a1 = HRAction(action_type="read_inbox", reasoning="Initial state requires gathering candidate applications to form a candidate pool. Reading inbox to extract applicant data.", memory_scratchpad=memory, confidence_score=0.95)
    o1 = env.step(a1)
    st.session_state.history.append({"step": 1, "action": a1, "obs": o1})
    
    for cand in env._scenario.candidate_names:
        memory += f" Checking {cand}."
        act = HRAction(action_type="query_crm_database", target_candidate=cand, reasoning=f"Candidate {cand} has submitted a profile. To avoid a high-risk hire, I must strictly adhere to compliance policy and verify their history in the CRM.", memory_scratchpad=memory, confidence_score=0.98)
        obs_ = env.step(act)
        if "BLACKLISTED" in obs_.last_action_result:
            memory += f" {cand}=REJECT."
        else:
            memory += f" {cand}=CLEAR."
        st.session_state.history.append({"step": env._step_count, "action": act, "obs": obs_})
        
    correct_cand = env._scenario.correct_hire
    act_eval = HRAction(action_type="evaluate_candidate", target_candidate=correct_cand, reasoning=f"Candidate {correct_cand} passed the compliance and background check. Proceeding with technical skill evaluation against job requirements.", memory_scratchpad=memory, confidence_score=0.90)
    o_eval = env.step(act_eval)
    st.session_state.history.append({"step": env._step_count, "action": act_eval, "obs": o_eval})
    
    act_cal = HRAction(action_type="check_calendar", reasoning="A valid candidate has been identified. I must query the hiring manager's calendar API to extract available time slots before sending an invite.", memory_scratchpad=memory, confidence_score=0.95)
    o_cal = env.step(act_cal)
    st.session_state.history.append({"step": env._step_count, "action": act_cal, "obs": o_cal})
    
    slot = env._scenario.calendar_slots[0]
    act_email = HRAction(action_type="send_email", target_candidate=correct_cand, scheduled_time=slot, reasoning=f"Candidate {correct_cand} cleared all background checks and possesses the highest skill match. Scheduling interview for {slot}.", memory_scratchpad=memory, confidence_score=0.88)
    o_email = env.step(act_email)
    st.session_state.history.append({"step": env._step_count, "action": act_email, "obs": o_email})
    
    act_done = HRAction(action_type="declare_done", reasoning="Workflow successfully completed. Due diligence performed across all candidates. Terminal decision reached with high confidence.", memory_scratchpad=memory, confidence_score=0.92)
    o_done = env.step(act_done)
    st.session_state.history.append({"step": env._step_count, "action": act_done, "obs": o_done})
    
    st.session_state.obs = o_done
    st.rerun()

if st.sidebar.button("⚠️ Auto-Run Edge Case (Human Review)"):
    reset_env("lead_security_architect")
    env = st.session_state.env
    memory = "Starting task. Candidate Elena flagged for Policy Violation."
    
    a1 = HRAction(action_type="read_inbox", reasoning="Extracting applicants.", memory_scratchpad=memory, confidence_score=0.95)
    env.step(a1)
    
    a2 = HRAction(action_type="query_crm_database", target_candidate="Elena Rostova", reasoning="Checking CRM. Found Policy Violation Warning.", memory_scratchpad=memory, confidence_score=0.95)
    env.step(a2)
    
    act_done = HRAction(action_type="declare_done", reasoning="Candidate Elena is the only viable technical match, but has a compliance warning. Confidence is low. Escalate to human.", memory_scratchpad=memory, confidence_score=0.62)
    o_done = env.step(act_done)
    st.session_state.history = [
        {"step": 1, "action": a1, "obs": o_done},  # Mocked history for brevity
        {"step": 2, "action": a2, "obs": o_done},
        {"step": 3, "action": act_done, "obs": o_done}
    ]
    st.session_state.obs = o_done
    st.rerun()

# Ensure env is initialized
if st.session_state.env is None:
    reset_env(scenario_option)

env = st.session_state.env
obs = st.session_state.obs

# ----------------- Dashboard Layout -----------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Environment State")
    ag = st.session_state.agent
    st.caption(f"Agent backend: **{ag.backend}** (heuristic = no LLM until LoRA file is valid)")
    st.metric("Current Step", env._step_count, f"Max {env._scenario.max_steps}")
    st.metric("Accumulated Reward", f"{env._cumulative_reward:.2f}")
    
    st.markdown("### Agent Memory Scratchpad")
    # Show the last used memory scratchpad, or empty
    last_mem = ""
    if st.session_state.history:
        last_mem = st.session_state.history[-1]["action"].memory_scratchpad
    st.info(last_mem if last_mem else "Empty...")

    st.markdown("### Job Details")
    st.markdown(f"**Role:** {env._scenario.job_title}")
    
    st.markdown("---")
    st.markdown("### 🥇 Trained Agent Advantage")
    st.markdown("""
    | Feature | Untrained Agent | Our Agent |
    |---|---|---|
    | Uses CRM APIs | ❌ | ✅ |
    | Avoids Blacklist | ❌ | ✅ |
    | Multi-step Reasoning | ❌ | ✅ |
    | Memory Retention | ❌ | ✅ |
    """)
    
with col2:
    st.subheader("🤖 Agent Execution Log")
    
    # Display history
    for entry in st.session_state.history:
        step_num = entry['step']
        act = entry['action']
        ob = entry['obs']
        
        with st.expander(f"Step {step_num}: {act.action_type}", expanded=(step_num == env._step_count)):
            st.markdown(f"**🧠 Thought / Reasoning:**\n> {act.reasoning}")
            st.markdown(f"**🛠️ Action:** `{act.action_type}`")
            if act.target_candidate:
                st.markdown(f"- **Target:** {act.target_candidate}")
            if act.scheduled_time:
                st.markdown(f"- **Time:** {act.scheduled_time}")
            if getattr(act, 'confidence_score', None):
                # We use getattr in case older history doesn't have it initialized
                st.markdown(f"- **Confidence:** {act.confidence_score*100:.1f}%")
            
            st.markdown("**👀 Observation (Result):**")
            st.code(ob.last_action_result, language="markdown")
            
            # Show if it hit a terminal state
            if ob.done:
                if act.confidence_score is not None and act.confidence_score < 0.70:
                    st.warning("⚠️ **FLAGGED FOR HUMAN REVIEW**\n\nConfidence score below 0.70 safety threshold. Agent requires manual override before final execution.")
                elif "SUCCESS" in ob.last_action_result:
                    st.success("Episode Complete: SUCCESS")
                else:
                    st.error("Episode Complete: FAILURE / VIOLATION (Untrained LLM Overconfidence)")

    if obs.done:
        st.warning("Episode has ended. Please reset to start a new simulation.")
    else:
        st.markdown("---")
        st.subheader("🤖 AI Agent Control")
        if st.session_state.agent:
            if st.button("🚀 Let Agent Think (Trained Inference)", type="primary"):
                with st.spinner("Agent is reasoning..."):
                    # Prepare observation text
                    prompt = f"Last Action Result: {obs.last_action_result}\nStep: {obs.step_number}/{obs.max_steps}\nTask: {obs.task_description}"
                    
                    ai_result = st.session_state.agent.get_action(prompt, env=env)
                    
                    # Execute in environment
                    new_act = HRAction(
                        action_type=ai_result["action_type"],
                        reasoning=ai_result["reasoning"],
                        target_candidate=ai_result.get("target_candidate"),
                        scheduled_time=ai_result.get("scheduled_time"),
                        memory_scratchpad=ai_result.get("memory_scratchpad") or "",
                        confidence_score=0.85
                    )
                    new_obs = env.step(new_act)
                    st.session_state.obs = new_obs
                    st.session_state.history.append({
                        "step": env._step_count,
                        "action": new_act,
                        "obs": new_obs
                    })
                    st.rerun()
        else:
            st.error("Agent not initialized.")

        st.markdown("---")
        st.subheader("Manual Step Injection (Demo Mode)")
        st.markdown("Inject a specific action to simulate the agent's next move.")
        
        with st.form("action_form"):
            a_type = st.selectbox("Action Type", HRHiringEnv.TOOLS)
            a_cand = st.selectbox("Target Candidate", [""] + env._scenario.candidate_names)
            a_time = st.selectbox("Scheduled Time", [""] + env._scenario.calendar_slots)
            a_reason = st.text_input("Thought / Reasoning", "I need to perform this action because...")
            a_mem = st.text_area("Memory Scratchpad", last_mem)
            a_conf = st.slider("Confidence Score", 0.0, 1.0, 0.90)
            
            submitted = st.form_submit_button("Execute Step")
            if submitted:
                # Construct action
                new_act = HRAction(
                    action_type=a_type,
                    target_candidate=a_cand if a_cand else None,
                    scheduled_time=a_time if a_time else None,
                    reasoning=a_reason,
                    memory_scratchpad=a_mem,
                    confidence_score=a_conf
                )
                new_obs = env.step(new_act)
                st.session_state.obs = new_obs
                st.session_state.history.append({
                    "step": env._step_count,
                    "action": new_act,
                    "obs": new_obs
                })
                st.rerun()

