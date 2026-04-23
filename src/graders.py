"""
Composable grading / reward system for the HR Hiring Agent environment.

Design principles:
  - Rich, informative signal at every step (not just 0/1 at the end)
  - Multiple rubric dimensions (protocol compliance, memory usage, efficiency)
  - Hard to game: an agent that skips CRM checks gets punished even if it
    accidentally picks the right candidate
  - Uses OpenEnv's composable rubric philosophy
"""

from typing import Dict, List, Optional, Tuple


class HRGrader:
    """
    Grades the agent's actions using a composable rubric.
    
    Rubric components:
      1. PROTOCOL COMPLIANCE (40%) — Did the agent follow all HR policies?
      2. MEMORY QUALITY (20%)     — Did the agent use its scratchpad effectively?
      3. DECISION QUALITY (30%)   — Did the agent hire the right person?
      4. EFFICIENCY (10%)         — Did the agent complete the task without wasting steps?
    """

    def __init__(self, max_steps: int, num_candidates: int):
        self.max_steps = max_steps
        self.num_candidates = num_candidates
        self.reset()

    def reset(self):
        """Reset the grader state for a new episode."""
        self.checked_inbox = False
        self.crm_checked: Dict[str, bool] = {}
        self.evaluated: Dict[str, bool] = {}
        self.checked_calendar = False
        self.memory_usage_count = 0
        self.total_steps = 0
        self.action_history: List[str] = []
        self.protocol_violations: List[str] = []
        self.reasoning_quality_scores: List[float] = []

    def grade_step(
        self,
        action_type: str,
        target_candidate: Optional[str],
        reasoning: str,
        memory_scratchpad: str,
        step_number: int,
        is_terminal: bool = False,
    ) -> Tuple[float, str, List[str]]:
        """
        Grade a single step. Returns (reward, feedback_message, violations).
        
        Rewards are in the range (-2.0, 1.0) to provide strong signal.
        """
        reward = 0.0
        feedback_parts = []
        self.total_steps = step_number
        self.action_history.append(action_type)

        # ── Memory usage bonus ──
        if memory_scratchpad and len(memory_scratchpad.strip()) > 10:
            self.memory_usage_count += 1
            reward += 0.02  # Small consistent bonus for maintaining memory
            feedback_parts.append("+0.02 memory maintenance")

        # ── Reasoning quality bonus ──
        if reasoning and len(reasoning.strip()) > 15:
            reward += 0.01
            feedback_parts.append("+0.01 reasoning provided")

        # ── Step-specific rewards ──
        if action_type == "read_inbox":
            if not self.checked_inbox:
                self.checked_inbox = True
                reward += 0.10
                feedback_parts.append("+0.10 inbox checked (protocol step 1)")
            else:
                reward -= 0.02
                feedback_parts.append("-0.02 duplicate inbox read (wasted step)")

        elif action_type == "query_crm_database":
            if target_candidate:
                if target_candidate not in self.crm_checked:
                    self.crm_checked[target_candidate] = True
                    reward += 0.15
                    feedback_parts.append(f"+0.15 CRM check for {target_candidate} (critical due diligence)")
                else:
                    reward -= 0.02
                    feedback_parts.append(f"-0.02 duplicate CRM check for {target_candidate}")
            else:
                reward -= 0.05
                feedback_parts.append("-0.05 CRM query without specifying candidate")

        elif action_type == "evaluate_candidate":
            if target_candidate:
                if target_candidate not in self.evaluated:
                    self.evaluated[target_candidate] = True
                    reward += 0.08
                    feedback_parts.append(f"+0.08 evaluated {target_candidate}")
                else:
                    reward -= 0.02
                    feedback_parts.append(f"-0.02 duplicate evaluation for {target_candidate}")
            else:
                reward -= 0.05
                feedback_parts.append("-0.05 evaluation without specifying candidate")

        elif action_type == "check_calendar":
            if not self.checked_calendar:
                self.checked_calendar = True
                reward += 0.08
                feedback_parts.append("+0.08 calendar checked (scheduling prerequisite)")
            else:
                reward -= 0.02
                feedback_parts.append("-0.02 duplicate calendar check")

        elif action_type == "send_email":
            if not self.checked_calendar:
                self.protocol_violations.append("Sent email without checking calendar")
                reward -= 0.15
                feedback_parts.append("-0.15 VIOLATION: email sent before checking calendar")

        elif action_type == "declare_done":
            pass  # Terminal grading handled separately

        # ── Time pressure penalty (escalating) ──
        if not is_terminal and step_number > self.max_steps * 0.7:
            pressure = -0.03 * (step_number / self.max_steps)
            reward += pressure
            feedback_parts.append(f"{pressure:.3f} time pressure")

        # Clamp step reward to a reasonable range
        reward = max(-0.5, min(0.5, reward))

        return reward, " | ".join(feedback_parts), self.protocol_violations

    def grade_terminal(
        self,
        hired_candidate: Optional[str],
        correct_hire: str,
        is_trap: bool,
        all_candidates: List[str],
    ) -> Tuple[float, str, List[str]]:
        """
        Grade the final outcome when the agent declares done.
        Returns (terminal_reward, feedback, violations).
        """
        reward = 0.0
        feedback_parts = []

        # ── 1. Did they hire the right person? (30% of total) ──
        if hired_candidate is None:
            reward -= 0.5
            feedback_parts.append("-0.50 FAILURE: No candidate hired")
            self.protocol_violations.append("Failed to hire anyone")

        elif is_trap:
            reward -= 2.0
            feedback_parts.append(f"-2.00 CRITICAL FAILURE: Hired {hired_candidate} who is BLACKLISTED/FLAGGED")
            self.protocol_violations.append(f"Hired blacklisted candidate: {hired_candidate}")

        elif hired_candidate == correct_hire:
            reward += 1.0
            feedback_parts.append(f"+1.00 PERFECT: Hired the best candidate ({hired_candidate})")
        else:
            # Hired a valid but non-optimal candidate
            reward += 0.30
            feedback_parts.append(f"+0.30 ACCEPTABLE: Hired {hired_candidate} (not optimal, best was {correct_hire})")

        # ── 2. Protocol compliance bonus (40% of total) ──
        if self.checked_inbox:
            reward += 0.05
            feedback_parts.append("+0.05 protocol: inbox checked")
        else:
            reward -= 0.10
            self.protocol_violations.append("Never checked inbox")
            feedback_parts.append("-0.10 VIOLATION: never checked inbox")

        if self.checked_calendar:
            reward += 0.05
            feedback_parts.append("+0.05 protocol: calendar checked")
        else:
            reward -= 0.10
            self.protocol_violations.append("Never checked calendar")
            feedback_parts.append("-0.10 VIOLATION: never checked calendar")

        # CRM compliance: all candidates should be checked
        unchecked = [c for c in all_candidates if c not in self.crm_checked]
        if not unchecked:
            reward += 0.20
            feedback_parts.append("+0.20 protocol: ALL candidates CRM-checked (thorough!)")
        else:
            penalty = -0.10 * len(unchecked)
            reward += penalty
            self.protocol_violations.append(f"CRM not checked for: {', '.join(unchecked)}")
            feedback_parts.append(f"{penalty:.2f} VIOLATION: CRM unchecked for {', '.join(unchecked)}")

        # ── 3. Memory usage bonus (20% of total) ──
        memory_ratio = self.memory_usage_count / max(1, self.total_steps)
        if memory_ratio > 0.7:
            reward += 0.15
            feedback_parts.append("+0.15 excellent memory discipline (used scratchpad consistently)")
        elif memory_ratio > 0.3:
            reward += 0.05
            feedback_parts.append("+0.05 some memory usage")
        else:
            feedback_parts.append("+0.00 poor memory discipline (rarely used scratchpad)")

        # ── 4. Efficiency bonus (10% of total) ──
        efficiency = 1.0 - (self.total_steps / self.max_steps)
        if efficiency > 0.5:
            eff_bonus = 0.10 * efficiency
            reward += eff_bonus
            feedback_parts.append(f"+{eff_bonus:.2f} efficiency bonus (completed quickly)")

        return reward, " | ".join(feedback_parts), self.protocol_violations
