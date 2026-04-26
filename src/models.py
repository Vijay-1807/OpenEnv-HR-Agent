"""
Pydantic models for the HR Hiring Agent environment.
Extends OpenEnv base types (Action, Observation, State) so the framework
can introspect schemas, validate payloads, and serialize everything cleanly.
"""

from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Standalone base types to remove openenv.core dependency
class Action(BaseModel): pass
class Observation(BaseModel): 
    reward: Optional[float] = None
    done: bool = False
class State(BaseModel): pass

# ─────────────────────────────────── Action ───────────────────────────────────

class HRAction(Action):
    """
    The agent's action at each step.
    
    Tools available:
      - read_inbox          : View all applicant emails
      - query_crm_database  : Look up a candidate's internal history
      - check_calendar      : See the hiring manager's available slots
      - evaluate_candidate  : Score a candidate based on job requirements
      - send_email          : Send an interview invitation to a candidate
      - declare_done        : Finalize the hiring decision
    """
    action_type: Literal[
        "read_inbox",
        "query_crm_database",
        "check_calendar",
        "evaluate_candidate",
        "send_email",
        "declare_done",
    ] = Field(
        ...,
        description="The tool or action to execute in the corporate HR environment."
    )
    target_candidate: Optional[str] = Field(
        default=None,
        description=(
            "Name of the candidate (e.g., 'Priya Sharma'). "
            "Required for query_crm_database, evaluate_candidate, and send_email."
        ),
    )
    scheduled_time: Optional[str] = Field(
        default=None,
        description=(
            "Time slot for the interview (e.g., 'Tuesday 2 PM'). "
            "Required for send_email."
        ),
    )
    reasoning: str = Field(
        default="",
        description=(
            "Explain WHY you are taking this action. "
            "Good reasoning is rewarded; it proves you are not guessing."
        ),
    )
    memory_scratchpad: str = Field(
        default="",
        description=(
            "CRITICAL — You suffer from context-window degradation over long episodes. "
            "Use this field every step to record bullet-point notes: candidate statuses, "
            "CRM findings, calendar slots, and your current plan. "
            "Failure to maintain memory will cause you to hire blacklisted candidates."
        ),
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description=(
            "Agent's confidence in this specific action (0.0 to 1.0). "
            "Particularly important when calling declare_done."
        ),
    )


# ──────────────────────────────── Observation ─────────────────────────────────

class HRObservation(Observation):
    """Observation returned to the agent after each step."""
    task_description: str = Field(
        ...,
        description="High-level description of the hiring task."
    )
    last_action_result: str = Field(
        ...,
        description="The result / output of the last action taken."
    )
    step_number: int = Field(
        ...,
        description="Current step number in this episode."
    )
    max_steps: int = Field(
        ...,
        description="Maximum steps allowed before timeout."
    )
    tools_available: List[str] = Field(
        default_factory=list,
        description="List of tool names the agent can call."
    )
    decision_log: List[str] = Field(
        default_factory=list,
        description="Transparency log showing the agent's past actions this episode."
    )


# ──────────────────────────────────  State  ───────────────────────────────────

class HRState(State):
    """Internal environment state exposed via the `state` property."""
    scenario_id: str = Field(
        ...,
        description="Which hiring scenario is active."
    )
    is_done: bool = Field(
        default=False,
        description="Whether the episode has terminated."
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated so far."
    )
    hired_candidate: Optional[str] = Field(
        default=None,
        description="Name of the candidate the agent chose to hire (if any)."
    )
    protocol_violations: List[str] = Field(
        default_factory=list,
        description="List of HR protocol violations committed by the agent."
    )
