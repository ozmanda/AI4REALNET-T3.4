from typing import Generic, TypeVar, Optional
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.step_utils.env_utils import AgentTransitionData
from flatland.envs.step_utils.states import TrainState

RewardType = TypeVar('RewardType')


class SimpleStepPenalty(Generic[RewardType]):
    """
    Flatland reward:
      - per step: step_penalty (default -1.0)
      - optional: +progress_coef * (prev_dist - curr_dist) >= 0 (disabled if progress_coef == 0.0)
      - one-time terminal_bonus when the agent first reaches DONE
    """
    def __init__(self, step_penalty: float = -1.0, terminal_bonus: float = 0.0, progress_coef: float = 0.0):
        self.step_penalty = float(step_penalty)
        self.terminal_bonus = float(terminal_bonus)
        self.progress_coef = float(progress_coef)
        self._last_distance = {}

    def _get_curr_distance(self, agent: EnvAgent, distance_map: Optional[DistanceMap]) -> float:
        if distance_map is None or agent.position is None or agent.target is None:
            return 0.0
        y, x = agent.position
        try:
            d = float(distance_map.distance_map[agent.handle, y, x, agent.direction])
        except Exception:
            d = 0.0
        if d == float('inf') or d != d:  # inf or NaN
            d = 0.0
        return d

    def on_episode_start(self, agents):
        self._last_distance = {a.handle: 0.0 for a in agents}

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData,
                    distance_map: Optional[DistanceMap], elapsed_steps: int) -> float:
        r = self.step_penalty
        if self.progress_coef != 0.0:
            curr_d = self._get_curr_distance(agent, distance_map)
            last_d = self._last_distance.get(agent.handle, curr_d)
            progress = max(0.0, last_d - curr_d)
            r += self.progress_coef * progress
            self._last_distance[agent.handle] = curr_d
        return float(r)

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: Optional[DistanceMap],
                              elapsed_steps: int) -> float:
        if self.terminal_bonus != 0.0 and agent.state == TrainState.DONE:
            return float(self.terminal_bonus)
        return 0.0

    def cumulate(self, *rewards: float) -> float:
        return float(sum(rewards))

    def empty(self) -> float:
        return 0.0