import pytest

from src.train import train


@pytest.mark.integration
def test_train_completes_without_error():
    history = train(episodes=200)
    assert isinstance(history, dict)


@pytest.mark.integration
def test_train_history_keys():
    history = train(episodes=200)
    assert set(history.keys()) == {"rewards_x", "win", "draw", "loss"}


@pytest.mark.integration
def test_train_history_values_are_lists():
    history = train(episodes=200)
    for key, values in history.items():
        assert isinstance(values, list), f"history['{key}'] deveria ser list"


@pytest.mark.integration
def test_train_history_rates_sum_to_one():
    history = train(episodes=1_000)
    for i, (w, d, l) in enumerate(
        zip(history["win"], history["draw"], history["loss"])
    ):
        total = w + d + l
        assert total == pytest.approx(1.0, abs=1e-9), (
            f"checkpoint {i}: win+draw+loss={total} != 1.0"
        )
