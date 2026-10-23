from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable

import numpy as np


ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("Влево", "Вправо", "Вниз", "Вверх")


def _argmax_random(values: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(rng.choice(best))


@dataclass(frozen=True)
class Maze:
    scenario: str = "static"
    width: int = 9
    height: int = 6

    @property
    def start(self) -> tuple[int, int]:
        return (0, 3) if self.scenario == "static" else (3, 0)

    @property
    def goal(self) -> tuple[int, int]:
        return (8, 5)

    @property
    def change_step(self) -> int | None:
        return {"blocking": 1_000, "shortcut": 3_000}.get(self.scenario)

    @property
    def default_horizon(self) -> int:
        return 6_000 if self.scenario == "shortcut" else 3_000

    def walls(self, step: int = 0) -> set[tuple[int, int]]:
        if self.scenario == "static":
            return {(2, 2), (2, 3), (2, 4), (5, 1), (7, 3), (7, 4), (7, 5)}
        if self.scenario == "blocking":
            before = {(x, 2) for x in range(0, 8)}
            after = {(x, 2) for x in range(1, 9)}
            return before if step < 1_000 else after
        if self.scenario == "shortcut":
            before = {(x, 2) for x in range(1, 9)}
            after = {(x, 2) for x in range(1, 8)}
            return before if step < 3_000 else after
        raise ValueError(f"Неизвестный сценарий: {self.scenario}")

    def transition(
        self, state: tuple[int, int], action: int, step: int = 0
    ) -> tuple[tuple[int, int], float, bool]:
        dx, dy = ACTIONS[action]
        candidate = (
            min(self.width - 1, max(0, state[0] + dx)),
            min(self.height - 1, max(0, state[1] + dy)),
        )
        next_state = state if candidate in self.walls(step) else candidate
        terminal = next_state == self.goal
        return next_state, float(terminal), terminal


@dataclass
class DynamicRun:
    rewards: np.ndarray
    q_values: np.ndarray
    path: list[tuple[int, int]]


def greedy_path(
    maze: Maze, q_values: np.ndarray, step: int, max_steps: int = 100
) -> list[tuple[int, int]]:
    state = maze.start
    path = [state]
    visited = {state}
    for _ in range(max_steps):
        action = int(np.argmax(q_values[state[0], state[1]]))
        next_state, _, terminal = maze.transition(state, action, step)
        path.append(next_state)
        if terminal or next_state in visited:
            break
        visited.add(next_state)
        state = next_state
    return path


def _epsilon_action(
    q_values: np.ndarray,
    state: tuple[int, int],
    rng: np.random.Generator,
    epsilon: float,
    bonus: np.ndarray | None = None,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(ACTIONS)))
    values = q_values[state[0], state[1]].copy()
    if bonus is not None:
        values += bonus[state[0], state[1]]
    return _argmax_random(values, rng)


def run_dynamic(
    algorithm: str,
    scenario: str,
    planning_steps: int = 20,
    total_steps: int | None = None,
    seed: int = 2,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    kappa: float = 0.001,
    priority_threshold: float = 1e-4,
) -> DynamicRun:
    """Run one of the planning agents in the changing Sutton maze."""
    allowed = {"dyna_q", "dyna_q_plus", "action_bonus", "prioritized", "trajectory"}
    if algorithm not in allowed:
        raise ValueError(f"Неизвестный алгоритм: {algorithm}")

    maze = Maze(scenario)
    horizon = total_steps or maze.default_horizon
    rng = np.random.default_rng(seed)
    q_values = np.zeros((maze.width, maze.height, len(ACTIONS)), dtype=float)
    model: dict[tuple[tuple[int, int], int], tuple[float, tuple[int, int]]] = {}
    last_seen = np.zeros_like(q_values, dtype=np.int64)
    predecessors: dict[tuple[int, int], set[tuple[tuple[int, int], int]]] = {}
    priority_queue: list[tuple[float, int, tuple[int, int], int]] = []
    queue_counter = 0

    state = maze.start
    cumulative_reward = 0.0
    reward_curve = np.zeros(horizon, dtype=float)

    def update(s: tuple[int, int], a: int, reward: float, nxt: tuple[int, int]) -> float:
        target = reward if nxt == maze.goal else reward + gamma * np.max(q_values[nxt[0], nxt[1]])
        td_error = target - q_values[s[0], s[1], a]
        q_values[s[0], s[1], a] += alpha * td_error
        return float(td_error)

    for step in range(1, horizon + 1):
        if algorithm == "action_bonus":
            time_bonus = kappa * np.sqrt(np.maximum(0, step - last_seen))
        else:
            time_bonus = None
        action = _epsilon_action(q_values, state, rng, epsilon, time_bonus)
        next_state, reward, terminal = maze.transition(state, action, step)
        td_error = update(state, action, reward, next_state)

        if algorithm in {"dyna_q_plus", "action_bonus"}:
            for candidate_action in range(len(ACTIONS)):
                model.setdefault((state, candidate_action), (0.0, state))

        old_transition = model.get((state, action))
        if old_transition is not None:
            old_next = old_transition[1]
            predecessors.get(old_next, set()).discard((state, action))
        model[(state, action)] = (reward, next_state)
        last_seen[state[0], state[1], action] = step
        predecessors.setdefault(next_state, set()).add((state, action))

        if algorithm == "prioritized" and abs(td_error) > priority_threshold:
            queue_counter += 1
            heapq.heappush(priority_queue, (-abs(td_error), queue_counter, state, action))

        if algorithm in {"dyna_q", "dyna_q_plus", "action_bonus"}:
            keys = tuple(model)
            for _ in range(planning_steps):
                plan_state, plan_action = keys[int(rng.integers(len(keys)))]
                plan_reward, plan_next = model[(plan_state, plan_action)]
                if algorithm == "dyna_q_plus":
                    elapsed = max(0, step - int(last_seen[plan_state[0], plan_state[1], plan_action]))
                    plan_reward += kappa * np.sqrt(elapsed)
                update(plan_state, plan_action, plan_reward, plan_next)

        elif algorithm == "prioritized":
            for _ in range(planning_steps):
                if not priority_queue:
                    break
                _, _, plan_state, plan_action = heapq.heappop(priority_queue)
                plan_reward, plan_next = model[(plan_state, plan_action)]
                update(plan_state, plan_action, plan_reward, plan_next)
                for pred_state, pred_action in predecessors.get(plan_state, set()):
                    pred_reward, _ = model[(pred_state, pred_action)]
                    pred_target = pred_reward + gamma * np.max(q_values[plan_state[0], plan_state[1]])
                    priority = abs(pred_target - q_values[pred_state[0], pred_state[1], pred_action])
                    if priority > priority_threshold:
                        queue_counter += 1
                        heapq.heappush(
                            priority_queue,
                            (-priority, queue_counter, pred_state, pred_action),
                        )

        elif algorithm == "trajectory":
            simulated_state = state
            for _ in range(planning_steps):
                simulated_action = _epsilon_action(
                    q_values, simulated_state, rng, epsilon
                )
                transition = model.get((simulated_state, simulated_action))
                if transition is None:
                    break
                plan_reward, plan_next = transition
                update(simulated_state, simulated_action, plan_reward, plan_next)
                if plan_next == maze.goal:
                    break
                simulated_state = plan_next

        cumulative_reward += reward
        reward_curve[step - 1] = cumulative_reward
        state = maze.start if terminal else next_state

    return DynamicRun(
        rewards=reward_curve,
        q_values=q_values,
        path=greedy_path(maze, q_values, horizon),
    )


def aggregate_dynamic(
    algorithms: Iterable[str],
    scenario: str,
    planning_steps: int,
    runs: int,
    total_steps: int | None = None,
    seed: int = 2,
) -> dict[str, dict[str, np.ndarray | float | list[tuple[int, int]]]]:
    output: dict[str, dict[str, np.ndarray | float | list[tuple[int, int]]]] = {}
    for algorithm in algorithms:
        curves: list[np.ndarray] = []
        representative: DynamicRun | None = None
        for run_index in range(runs):
            result = run_dynamic(
                algorithm=algorithm,
                scenario=scenario,
                planning_steps=planning_steps,
                total_steps=total_steps,
                seed=seed + run_index * 997,
            )
            curves.append(result.rewards)
            representative = result
        stacked = np.vstack(curves)
        assert representative is not None
        output[algorithm] = {
            "mean": stacked.mean(axis=0),
            "std": stacked.std(axis=0),
            "final_reward": float(stacked[:, -1].mean()),
            "path": representative.path,
        }
    return output


def run_static_dyna(
    episodes: int,
    planning_steps: int,
    seed: int,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
) -> np.ndarray:
    maze = Maze("static")
    rng = np.random.default_rng(seed)
    q_values = np.zeros((maze.width, maze.height, len(ACTIONS)))
    model: dict[tuple[tuple[int, int], int], tuple[float, tuple[int, int]]] = {}
    counts: list[int] = []
    for _ in range(episodes):
        state = maze.start
        steps = 0
        while state != maze.goal and steps < 5_000:
            steps += 1
            action = _epsilon_action(q_values, state, rng, epsilon)
            next_state, reward, _ = maze.transition(state, action)
            target = reward if next_state == maze.goal else reward + gamma * np.max(q_values[next_state])
            q_values[state][action] += alpha * (target - q_values[state][action])
            model[(state, action)] = (reward, next_state)
            keys = tuple(model)
            for _ in range(planning_steps):
                plan_state, plan_action = keys[int(rng.integers(len(keys)))]
                plan_reward, plan_next = model[(plan_state, plan_action)]
                plan_target = (
                    plan_reward
                    if plan_next == maze.goal
                    else plan_reward + gamma * np.max(q_values[plan_next])
                )
                q_values[plan_state][plan_action] += alpha * (
                    plan_target - q_values[plan_state][plan_action]
                )
            state = next_state
        counts.append(steps)
    return np.asarray(counts)


def run_n_step_sarsa(
    episodes: int,
    n_steps: int,
    seed: int,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
) -> np.ndarray:
    maze = Maze("static")
    rng = np.random.default_rng(seed)
    q_values = np.zeros((maze.width, maze.height, len(ACTIONS)))
    episode_lengths: list[int] = []

    for _ in range(episodes):
        states = [maze.start]
        actions = [_epsilon_action(q_values, maze.start, rng, epsilon)]
        rewards = [0.0]
        terminal_time = float("inf")
        time = 0
        while True:
            if time < terminal_time:
                next_state, reward, terminal = maze.transition(states[time], actions[time])
                states.append(next_state)
                rewards.append(reward)
                if terminal or time >= 4_999:
                    terminal_time = time + 1
                else:
                    actions.append(_epsilon_action(q_values, next_state, rng, epsilon))

            tau = time - n_steps + 1
            if tau >= 0:
                upper = (
                    tau + n_steps
                    if terminal_time == float("inf")
                    else min(tau + n_steps, int(terminal_time))
                )
                return_value = sum(
                    (gamma ** (index - tau - 1)) * rewards[index]
                    for index in range(tau + 1, upper + 1)
                )
                if tau + n_steps < terminal_time:
                    bootstrap_state = states[tau + n_steps]
                    bootstrap_action = actions[tau + n_steps]
                    return_value += (gamma**n_steps) * q_values[bootstrap_state][bootstrap_action]
                state_tau, action_tau = states[tau], actions[tau]
                q_values[state_tau][action_tau] += alpha * (
                    return_value - q_values[state_tau][action_tau]
                )
            if tau == terminal_time - 1:
                break
            time += 1
        episode_lengths.append(int(terminal_time))
    return np.asarray(episode_lengths)


def aggregate_static(episodes: int, runs: int, seed: int = 2) -> dict[str, np.ndarray]:
    configurations = {
        "Dyna-Q, n=0": lambda run_seed: run_static_dyna(episodes, 0, run_seed),
        "Dyna-Q, n=5": lambda run_seed: run_static_dyna(episodes, 5, run_seed),
        "Dyna-Q, n=50": lambda run_seed: run_static_dyna(episodes, 50, run_seed),
        "n-шаговый SARSA, n=4": lambda run_seed: run_n_step_sarsa(episodes, 4, run_seed),
        "n-шаговый SARSA, n=8": lambda run_seed: run_n_step_sarsa(episodes, 8, run_seed),
    }
    result: dict[str, np.ndarray] = {}
    for label, runner in configurations.items():
        values = [runner(seed + run * 997) for run in range(runs)]
        result[label] = np.mean(values, axis=0)
    return result
