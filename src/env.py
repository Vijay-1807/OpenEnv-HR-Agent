"""
HR Hiring Agent — OpenEnv-compliant Environment

A long-horizon, multi-tool environment that simulates a corporate HR hiring
workflow. The agent must read applicant emails, cross-reference the CRM database
for blacklisted candidates, check calendar availability, evaluate candidates,
and send interview invitations — all while maintaining memory state across steps.

Themes addressed:
  - Theme #2: Long-Horizon Planning & Instruction Following
  - Theme #3.1: World Modeling / Professional Tasks

OpenEnv compliance:
  - Extends no base class (same pattern as email-triage-env that passed Round 1)
  - reset() -> Observation (with done=False, reward=None)
  - step(action) -> Observation (with done, reward set)
  - state (property) -> State
  - close() -> None
  - SUPPORTS_CONCURRENT_SESSIONS = True
"""

from typing import Optional, Any
import json

from .models import HRAction, HRObservation, HRState
from .scenarios import ScenarioManager
from .graders import HRGrader


class HRHiringEnv:
    """
    OpenEnv-compliant HR Hiring Agent Environment.

    The agent is given a hiring task and must navigate a multi-step workflow
    using simulated corporate tools to find and hire the right candidate
    without violating company policies.

    Configuration:
        scenario_id: str   — pick a specific scenario (e.g. "senior_python_dev")
        difficulty: str    — pick a difficulty level ("easy", "medium", "hard")
        If neither is provided, a random scenario is chosen.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    TOOLS = [
        "read_inbox",
        "query_crm_database",
        "check_calendar",
        "evaluate_candidate",
        "send_email",
        "declare_done",
    ]

    def __init__(self, scenario_id: str = None, difficulty: str = None):
        self._scenario_id = scenario_id
        self._difficulty = difficulty
        self._scenario: Optional[ScenarioManager] = None
        self._grader: Optional[HRGrader] = None
        self._step_count = 0
        self._is_done = False
        self._emailed_candidate = None
        self._emailed_time = None
        self._cumulative_reward = 0.0
        self._decision_log = []
        self._episode_id = "hr-hiring-1"
        self._protocol_violations = []

    def close(self) -> None:
        """Clean up resources. Required by OpenEnv framework."""
        pass

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> HRObservation:
        """Reset the environment and return the initial observation."""
        self._scenario = ScenarioManager(
            scenario_id=self._scenario_id,
            difficulty=self._difficulty,
        )
        self._grader = HRGrader(
            max_steps=self._scenario.max_steps,
            num_candidates=len(self._scenario.candidate_names),
        )
        self._step_count = 0
        self._is_done = False
        self._emailed_candidate = None
        self._emailed_time = None
        self._cumulative_reward = 0.0
        self._decision_log = []
        self._protocol_violations = []

        if episode_id:
            self._episode_id = episode_id

        # Build the opening message
        policies_text = "\n".join(f"  - {p}" for p in self._scenario.company_policies)
        reqs_text = "\n".join(f"  - {r}" for r in self._scenario.job_requirements)

        welcome = (
            f"=== ENTERPRISE HR PORTAL ===\n"
            f"TASK: Hire a {self._scenario.job_title}\n\n"
            f"Job Requirements:\n{reqs_text}\n\n"
            f"Company Policies (MUST FOLLOW):\n{policies_text}\n\n"
            f"Available Tools: {', '.join(self.TOOLS)}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Read your inbox to see candidate applications.\n"
            f"2. Cross-reference EVERY candidate with the CRM database.\n"
            f"3. Evaluate candidates against job requirements.\n"
            f"4. Check the hiring manager's calendar for available slots.\n"
            f"5. Send an interview email to the best qualified, policy-compliant candidate.\n"
            f"6. Declare done.\n\n"
            f"WARNING: Use your memory_scratchpad EVERY step to track your findings.\n"
            f"Context window degradation WILL cause you to forget critical CRM flags.\n"
            f"You have {self._scenario.max_steps} steps maximum."
        )

        return self._make_obs(
            result=welcome,
            reward=None,
            done=False,
        )

    async def reset_async(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> HRObservation:
        return self.reset(seed=seed, episode_id=episode_id, **kwargs)

    def step(self, action: HRAction, timeout_s: Optional[float] = None, **kwargs) -> HRObservation:
        """Execute one action in the environment."""
        if self._is_done:
            return self._make_obs(
                result="Episode already complete.",
                reward=0.0,
                done=True,
            )

        self._step_count += 1

        # Handle raw dict input (for compatibility with training scripts)
        if isinstance(action, dict):
            try:
                action = HRAction(**action)
            except Exception as e:
                if self._step_count >= self._scenario.max_steps:
                    self._is_done = True
                return self._make_obs(
                    result=f"ACTION VALIDATION ERROR: {e}\nPlease provide a valid action.",
                    reward=-0.10,
                    done=self._is_done,
                )

        act = action.action_type
        cand = action.target_candidate
        time_slot = action.scheduled_time
        reasoning = action.reasoning
        memory = action.memory_scratchpad

        # Grade this step
        step_reward, step_feedback, violations = self._grader.grade_step(
            action_type=act,
            target_candidate=cand,
            reasoning=reasoning,
            memory_scratchpad=memory,
            step_number=self._step_count,
        )

        result_msg = ""

        # ──────────────── Tool Execution ────────────────

        if act == "read_inbox":
            inbox = self._scenario.get_inbox()
            result_msg = "=== INBOX ===\n"
            for i, email in enumerate(inbox, 1):
                result_msg += f"\n--- Email {i} from: {email['from']} ---\n{email['body']}\n"
            result_msg += f"\n({len(inbox)} candidate emails found)"
            self._decision_log.append(f"Step {self._step_count}: Read inbox ({len(inbox)} emails)")

        elif act == "query_crm_database":
            if not cand:
                result_msg = "ERROR: target_candidate is required for query_crm_database."
                step_reward -= 0.05
            else:
                record = self._scenario.get_crm_record(cand)
                if record is None:
                    result_msg = f"CRM LOOKUP: No records found for '{cand}'. Candidate not in system."
                else:
                    result_msg = (
                        f"=== CRM RECORD: {cand} ===\n"
                        f"Status: {record['status']}\n"
                        f"Flag: {record['flag']}\n"
                        f"Details: {record['reason']}\n"
                    )
                    if record.get("previous_role"):
                        result_msg += f"Previous Role: {record['previous_role']}\n"
                self._decision_log.append(f"Step {self._step_count}: CRM check for {cand}")

        elif act == "evaluate_candidate":
            if not cand:
                result_msg = "ERROR: target_candidate is required for evaluate_candidate."
                step_reward -= 0.05
            else:
                score = self._scenario.get_candidate_score(cand)
                reqs = self._scenario.job_requirements
                if score == 0:
                    result_msg = (
                        f"=== EVALUATION: {cand} ===\n"
                        f"Skill Match Score: CANNOT EVALUATE — candidate may have policy flags.\n"
                        f"Recommendation: Check CRM database before proceeding."
                    )
                elif score >= 70:
                    result_msg = (
                        f"=== EVALUATION: {cand} ===\n"
                        f"Skill Match Score: {score}/100 — STRONG MATCH\n"
                        f"Requirements Met: {', '.join(reqs[:len(reqs)-1])} (verified)\n"
                        f"Recommendation: Proceed to interview scheduling if CRM clear."
                    )
                elif score >= 40:
                    result_msg = (
                        f"=== EVALUATION: {cand} ===\n"
                        f"Skill Match Score: {score}/100 — PARTIAL MATCH\n"
                        f"Requirements Met: {', '.join(reqs[:1])} (partially)\n"
                        f"Recommendation: Consider if no stronger candidates available."
                    )
                else:
                    result_msg = (
                        f"=== EVALUATION: {cand} ===\n"
                        f"Skill Match Score: {score}/100 — WEAK MATCH\n"
                        f"Requirements Met: Minimal\n"
                        f"Recommendation: Not recommended for this role."
                    )
                self._decision_log.append(f"Step {self._step_count}: Evaluated {cand} (score: {score})")

        elif act == "check_calendar":
            slots = self._scenario.calendar_slots
            result_msg = (
                f"=== HIRING MANAGER CALENDAR ===\n"
                f"Available Slots:\n"
                + "\n".join(f"  [{i+1}] {slot}" for i, slot in enumerate(slots))
                + f"\n\n({len(slots)} slots available this week)"
            )
            self._decision_log.append(f"Step {self._step_count}: Checked calendar ({len(slots)} slots)")

        elif act == "send_email":
            if not cand or not time_slot:
                result_msg = "ERROR: Both target_candidate and scheduled_time are required for send_email."
                step_reward -= 0.05
            elif time_slot not in self._scenario.calendar_slots:
                result_msg = (
                    f"ERROR: '{time_slot}' is not a valid time slot.\n"
                    f"Available slots: {', '.join(self._scenario.calendar_slots)}\n"
                    f"Use check_calendar to see available times."
                )
                step_reward -= 0.05
            elif cand not in self._scenario.candidate_names:
                result_msg = f"ERROR: '{cand}' is not a recognized candidate."
                step_reward -= 0.05
            else:
                self._emailed_candidate = cand
                self._emailed_time = time_slot
                result_msg = (
                    f"=== EMAIL SENT ===\n"
                    f"To: {cand}\n"
                    f"Subject: Interview Invitation — {self._scenario.job_title}\n"
                    f"Scheduled: {time_slot}\n\n"
                    f"Interview email has been sent successfully.\n"
                    f"You must now call 'declare_done' to finalize the hiring decision."
                )
                self._decision_log.append(f"Step {self._step_count}: Sent interview email to {cand} for {time_slot}")

        elif act == "declare_done":
            self._is_done = True

            # Terminal grading
            is_trap = self._scenario.is_candidate_trap(self._emailed_candidate) if self._emailed_candidate else False
            terminal_reward, terminal_feedback, violations = self._grader.grade_terminal(
                hired_candidate=self._emailed_candidate,
                correct_hire=self._scenario.correct_hire,
                is_trap=is_trap,
                all_candidates=self._scenario.candidate_names,
            )
            step_reward = terminal_reward
            self._protocol_violations = violations

            if self._emailed_candidate and is_trap:
                result_msg = (
                    f"=== TERMINAL FAILURE ===\n"
                    f"You hired {self._emailed_candidate}, who has CRITICAL FLAGS in the CRM!\n"
                    f"This is a severe HR protocol violation.\n\n"
                    f"Grading: {terminal_feedback}\n"
                    f"Violations: {', '.join(violations) if violations else 'None'}"
                )
            elif self._emailed_candidate == self._scenario.correct_hire:
                result_msg = (
                    f"=== SUCCESS ===\n"
                    f"You correctly hired {self._emailed_candidate} after following all protocols!\n\n"
                    f"Grading: {terminal_feedback}\n"
                    f"Violations: {', '.join(violations) if violations else 'None'}"
                )
            elif self._emailed_candidate:
                result_msg = (
                    f"=== PARTIAL SUCCESS ===\n"
                    f"You hired {self._emailed_candidate}. They are policy-compliant but not optimal.\n"
                    f"The best candidate was {self._scenario.correct_hire}.\n\n"
                    f"Grading: {terminal_feedback}\n"
                    f"Violations: {', '.join(violations) if violations else 'None'}"
                )
            else:
                result_msg = (
                    f"=== FAILURE ===\n"
                    f"You declared done without hiring anyone.\n\n"
                    f"Grading: {terminal_feedback}\n"
                    f"Violations: {', '.join(violations) if violations else 'None'}"
                )
            self._decision_log.append(f"Step {self._step_count}: Declared done (hired: {self._emailed_candidate})")

        # ── Check max steps ──
        if self._step_count >= self._scenario.max_steps and not self._is_done:
            self._is_done = True
            result_msg += "\n\nTIMEOUT: Maximum steps reached. Workflow failed."
            step_reward -= 0.5

        self._cumulative_reward += step_reward

        # Clamp reward to (0, 1) range for OpenEnv compatibility
        clamped_reward = max(0.01, min(0.99, (step_reward + 2.0) / 4.0))  # map [-2, 2] -> [0, 1]

        return self._make_obs(
            result=result_msg + f"\n\n[Step Reward: {step_feedback}]",
            reward=clamped_reward,
            done=self._is_done,
        )

    async def step_async(self, action: HRAction, timeout_s: Optional[float] = None, **kwargs) -> HRObservation:
        return self.step(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> HRState:
        """Get the current environment state."""
        return HRState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            scenario_id=self._scenario.id if self._scenario else "none",
            is_done=self._is_done,
            cumulative_reward=self._cumulative_reward,
            hired_candidate=self._emailed_candidate,
            protocol_violations=self._protocol_violations,
        )

    def _make_obs(self, result: str, reward: Optional[float], done: bool) -> HRObservation:
        """Build an HRObservation with OpenEnv-required done/reward fields."""
        return HRObservation(
            task_description=(
                f"Hire a {self._scenario.job_title}. "
                f"Cross-reference all candidates with CRM. "
                f"Do not hire blacklisted or flagged individuals. "
                f"Follow all company policies."
            ) if self._scenario else "Environment not initialized. Call reset() first.",
            last_action_result=result,
            step_number=self._step_count,
            max_steps=self._scenario.max_steps if self._scenario else 0,
            tools_available=self.TOOLS,
            decision_log=list(self._decision_log),
            done=done,
            reward=reward,
        )
