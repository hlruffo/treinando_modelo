import os
import tempfile

import pytest

from src.agent import QLearningAgent


def make_agent(**kwargs) -> QLearningAgent:
    return QLearningAgent(**kwargs)


STATE_A = (0, 0, 0, 0, 0, 0, 0, 0, 0)
STATE_B = (1, 0, 0, 0, 0, 0, 0, 0, 0)
ACTIONS = [0, 1, 2]


# --- choose_action ---


def test_choose_action_explores_when_epsilon_one():
    agent = make_agent(epsilon=1.0)
    counts = {a: 0 for a in ACTIONS}
    for _ in range(300):
        a = agent.choose_action(STATE_A, ACTIONS)
        counts[a] += 1
    # com epsilon=1 todos os actions devem aparecer
    assert all(c > 0 for c in counts.values())


def test_choose_action_greedy_when_epsilon_zero():
    agent = make_agent(epsilon=0.0)
    # torna a ação 2 a melhor
    agent.q[(STATE_A, 2)] = 10.0
    for _ in range(50):
        assert agent.choose_action(STATE_A, ACTIONS) == 2


def test_choose_action_greedy_picks_max_q():
    agent = make_agent(epsilon=0.0)
    agent.q[(STATE_A, 0)] = 1.0
    agent.q[(STATE_A, 1)] = 5.0
    agent.q[(STATE_A, 2)] = 3.0
    assert agent.choose_action(STATE_A, ACTIONS) == 1


# --- update ---


def test_update_terminal_state():
    agent = make_agent(alpha=1.0, gamma=0.9)
    agent.update(
        STATE_A, 0, reward=1.0, next_state=STATE_B, next_available=[], done=True
    )
    # com alpha=1 e done=True: Q = 0 + 1*(1.0 - 0) = 1.0
    assert agent.q[(STATE_A, 0)] == pytest.approx(1.0)


def test_update_non_terminal_state():
    agent = make_agent(alpha=1.0, gamma=0.9)
    agent.q[(STATE_B, 1)] = 2.0  # max next Q
    agent.update(
        STATE_A, 0, reward=0.0, next_state=STATE_B, next_available=[1], done=False
    )
    # target = 0 + 0.9*2.0 = 1.8
    assert agent.q[(STATE_A, 0)] == pytest.approx(1.8)


def test_update_accumulates_across_calls():
    agent = make_agent(alpha=0.5, gamma=0.9)
    agent.update(
        STATE_A, 0, reward=1.0, next_state=STATE_B, next_available=[], done=True
    )
    first = agent.q[(STATE_A, 0)]
    agent.update(
        STATE_A, 0, reward=1.0, next_state=STATE_B, next_available=[], done=True
    )
    second = agent.q[(STATE_A, 0)]
    # segunda atualização deve aproximar mais de 1.0
    assert second > first


# --- decay_epsilon ---


def test_decay_epsilon_reduces_value():
    agent = make_agent(epsilon=0.3, epsilon_min=0.05)
    agent.decay_epsilon(factor=0.5)
    assert agent.epsilon == pytest.approx(0.15)


def test_decay_epsilon_respects_minimum():
    agent = make_agent(epsilon=0.06, epsilon_min=0.05)
    agent.decay_epsilon(factor=0.5)
    assert agent.epsilon == pytest.approx(0.05)


def test_decay_epsilon_already_at_min():
    agent = make_agent(epsilon=0.05, epsilon_min=0.05)
    agent.decay_epsilon(factor=0.5)
    assert agent.epsilon == pytest.approx(0.05)


# --- save / load ---


def test_save_and_load_roundtrip():
    agent = make_agent(alpha=0.1, gamma=0.8, epsilon=0.2)
    agent.q[(STATE_A, 0)] = 3.14

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name

    try:
        agent.save(path)
        loaded = QLearningAgent.load(path)
        assert loaded.alpha == agent.alpha
        assert loaded.gamma == agent.gamma
        assert loaded.epsilon == agent.epsilon
        assert loaded.q[(STATE_A, 0)] == pytest.approx(3.14)
    finally:
        os.unlink(path)
