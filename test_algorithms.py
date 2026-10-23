import numpy as np

from algorithms import Maze, aggregate_static, run_dynamic


def test_dynamic_maze_changes_at_expected_step():
    blocking = Maze("blocking")
    shortcut = Maze("shortcut")
    assert blocking.walls(999) != blocking.walls(1_000)
    assert shortcut.walls(2_999) != shortcut.walls(3_000)


def test_all_dynamic_agents_return_finite_curves():
    for name in ("dyna_q", "dyna_q_plus", "action_bonus", "prioritized", "trajectory"):
        result = run_dynamic(name, "blocking", planning_steps=3, total_steps=80, seed=2)
        assert result.rewards.shape == (80,)
        assert np.isfinite(result.rewards).all()
        assert result.path[0] == Maze("blocking").start


def test_static_comparison_has_all_required_methods():
    result = aggregate_static(episodes=3, runs=1, seed=2)
    assert len(result) == 5
    assert all(values.shape == (3,) for values in result.values())
