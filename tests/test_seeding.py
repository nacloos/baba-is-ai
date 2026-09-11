import numpy as np
import pytest

import baba

# Environments whose _gen_grid draws random samples (object choice, placement, agent dir, ...)
ENV_IDS = [
    "env/you_win",
    "env/goto_win",
    "env/make_win",
    "env/two_room-goto_win",
]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_reset_same_seed_is_deterministic(env_id):
    env = baba.make(env_id)

    obs1 = env.reset(seed=123)
    obs2 = env.reset(seed=123)

    assert np.array_equal(obs1, obs2)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_reset_different_seeds_can_differ(env_id):
    env = baba.make(env_id)

    observations = [env.reset(seed=s) for s in range(20)]

    # At least one pair of resets with different seeds should produce a
    # different initial observation (randomized init is not seed-independent).
    assert any(
        not np.array_equal(observations[0], obs) for obs in observations[1:]
    )


def test_reset_without_seed_can_vary_across_instances():
    # Two freshly constructed envs reset without an explicit seed should not
    # be forced into lockstep by shared global RNG state.
    obs = []
    for _ in range(20):
        env = baba.make("env/goto_win")
        obs.append(env.reset())

    assert any(not np.array_equal(obs[0], o) for o in obs[1:])
