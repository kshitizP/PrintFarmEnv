"""
Decision-point wrapper — wraps the full PrintFarmEnv as a single-step
decision environment for GRPO training.

Each training "episode" is:
1. env.reset() + step forward with rule-based agent until a decision point.
2. Present observation + context to the model. Model outputs ONE action.
3. Execute action, then continue with rules for K more steps.
4. Reward = reward delta over K steps vs. letting rules handle the decision point.

Decision points:
  - operator_notes appeared (any note)
  - customer_messages arrived
  - anomaly_flags raised
  - a spool is <15% of job weight remaining
  - a printer has fatigue >= 7 (approaching catastrophic)
  - a printer is in ERROR state with no ticket
"""

from typing import Any, Dict, List, Optional, Tuple

from submission.env.env import PrintFarmEnvironment
from submission.env.models import FarmAction, FarmActionEnum, FarmObservation
from submission.shared.serialize import serialize_obs


# Number of steps to run after the LLM decision to measure outcome
K_HORIZON = 10


def _is_decision_point(obs: FarmObservation) -> bool:
    """Check if the current observation warrants a decision point."""
    # Unstructured signals
    if obs.operator_notes:
        return True
    if obs.customer_messages:
        return True
    if obs.anomaly_flags:
        return True

    # Structured signals that benefit from LLM reasoning
    for p in obs.printers:
        # Spool running low on a printing printer
        if str(p.state) in ("PRINTING",) and p.spool_weight_g < 50:
            return True
        # High fatigue approaching catastrophe
        if p.fatigue_level >= 7 and str(p.state) not in ("OFFLINE", "MAINTENANCE", "MAINTENANCE_QUEUED"):
            return True
        # ERROR with no ticket
        if str(p.state) == "ERROR" and p.outstanding_ticket_id is None:
            return True

    return False


def _rules_action(obs: FarmObservation) -> FarmAction:
    """Simple rule-based agent for non-decision-point steps.

    This is a minimal version of the full rule agent — handles the easy 80%.
    """
    printers = obs.printers
    operators = obs.operators
    jobs = obs.active_queue

    SKILL_RANK = {"junior": 0, "senior": 1, "lead": 2}
    TICKET_SKILL_REQ = {
        "spool_swap": "junior", "maintenance_basic": "senior",
        "unjam_printer": "senior", "diagnostic_physical": "junior",
    }

    def _best_op(ticket_type):
        req = SKILL_RANK.get(TICKET_SKILL_REQ.get(ticket_type, "junior"), 0)
        cands = [o for o in operators
                 if SKILL_RANK.get(o.skill_level, 0) >= req
                 and o.is_on_shift and o.queue_size < o.queue_capacity]
        return min(cands, key=lambda o: (o.queue_size, o.operator_id)) if cands else None

    # ERROR recovery
    for p in printers:
        if str(p.state) == "ERROR" and p.outstanding_ticket_id is None:
            op = _best_op("unjam_printer")
            if op:
                return FarmAction(action=FarmActionEnum.DISPATCH_TICKET,
                                  printer_id=p.printer_id, operator_id=op.operator_id,
                                  ticket_type="unjam_printer")

    # Catastrophe prevention
    for p in printers:
        if p.fatigue_level >= 8 and str(p.state) == "IDLE" and p.outstanding_ticket_id is None:
            op = _best_op("maintenance_basic")
            if op:
                return FarmAction(action=FarmActionEnum.DISPATCH_TICKET,
                                  printer_id=p.printer_id, operator_id=op.operator_id,
                                  ticket_type="maintenance_basic")

    # Runout recovery
    for p in printers:
        if str(p.state) == "PAUSED_RUNOUT" and p.current_job_id:
            mat = p.current_material or "PLA"
            for j in jobs:
                if j.job_id == p.current_job_id:
                    mat = j.material_required
                    break
            return FarmAction(action=FarmActionEnum.REQUEST_SPOOL_SWAP,
                              printer_id=p.printer_id, material=mat)

    # Resume paused
    for p in printers:
        if str(p.state) == "IDLE" and p.current_job_id:
            return FarmAction(action=FarmActionEnum.RESUME_JOB,
                              printer_id=p.printer_id, job_id=p.current_job_id)

    # Job assignment
    pending = sorted([j for j in jobs if str(j.state) == "PENDING"],
                     key=lambda j: (-j.priority, j.deadline_steps or 9999))
    idle = [p for p in printers
            if str(p.state) == "IDLE" and p.current_job_id is None
            and p.fatigue_level < 8 and p.outstanding_ticket_id is None]

    for job in pending:
        for p in idle:
            if p.current_material == job.material_required and p.spool_weight_g >= job.weight_required_g:
                return FarmAction(action=FarmActionEnum.ASSIGN_JOB,
                                  printer_id=p.printer_id, job_id=job.job_id)

    return FarmAction(action=FarmActionEnum.WAIT)


class DecisionPointEnv:
    """Wraps PrintFarmEnvironment as a single-decision GRPO training env.

    Usage:
        dp_env = DecisionPointEnv()
        prompt, obs = dp_env.reset(seed=42, task_id="task_1")
        # prompt is the serialized observation at the decision point
        # Agent produces an action string
        reward = dp_env.step(agent_action)
    """

    def __init__(self, k_horizon: int = K_HORIZON):
        self.inner_env = PrintFarmEnvironment()
        self.k_horizon = k_horizon
        self._decision_obs: Optional[FarmObservation] = None
        self._decision_reward: float = 0.0
        self._decision_tags: Dict[str, Any] = {}

    def reset(
        self,
        seed: int = 42,
        task_id: str = "task_1",
        max_steps_to_decision: int = 60,
    ) -> Tuple[str, FarmObservation]:
        """Reset and advance to the first decision point.

        Returns:
            (serialized_obs, raw_obs) at the decision point.
            If no decision point is reached, returns the final obs.
        """
        obs = self.inner_env.reset(seed=seed, task_id=task_id)

        steps = 0
        while not obs.done and steps < max_steps_to_decision:
            if _is_decision_point(obs):
                self._decision_obs = obs
                self._decision_reward = obs.reward
                self._decision_tags = self.inner_env.get_ground_truth_tags()
                return serialize_obs(obs), obs

            action = _rules_action(obs)
            obs = self.inner_env.step(action)
            steps += 1

        # No decision point found — return current state anyway
        self._decision_obs = obs
        self._decision_reward = obs.reward
        self._decision_tags = self.inner_env.get_ground_truth_tags()
        return serialize_obs(obs), obs

    def step(self, action: FarmAction) -> Tuple[float, Dict[str, float]]:
        """Execute the LLM's action, then run rules for K steps.

        Returns:
            (total_reward_delta, component_rewards)
        """
        if self._decision_obs is None:
            raise RuntimeError("Must call reset() before step()")

        pre_reward = self._decision_reward

        # Execute the LLM's action
        obs = self.inner_env.step(action)

        # Run rules for K more steps
        for _ in range(self.k_horizon):
            if obs.done:
                break
            action = _rules_action(obs)
            obs = self.inner_env.step(action)

        # Measure the total reward delta
        post_reward = obs.reward
        reward_delta = post_reward - pre_reward

        # Also compute: what would rules-only have done?
        # (We'd need a separate env rollout for the baseline — done at reward computation time)

        return reward_delta, {
            "total_delta": reward_delta,
            "pre_reward": pre_reward,
            "post_reward": post_reward,
            "decision_tags": self._decision_tags,
        }

    def get_decision_tags(self) -> Dict[str, Any]:
        """Get ground-truth tags for the decision point."""
        return self._decision_tags
