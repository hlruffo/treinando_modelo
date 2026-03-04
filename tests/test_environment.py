import pytest
from src.environment import TicTacToeEnv


def make_env() -> TicTacToeEnv:
    env = TicTacToeEnv()
    env.reset()
    return env


def test_reset_state():
    env = make_env()
    assert env.reset() == (0,) * 9
    assert env.current_player == 1


def test_available_actions_full_board():
    env = make_env()
    assert env.available_actions() == list(range(9))


def test_available_actions_after_moves():
    env = make_env()
    env.board[0] = 1
    env.board[4] = -1

    avail = env.available_actions()
    assert 0 not in avail
    assert 4 not in avail
    assert len(avail) == 7


def test_invalid_move_raises():
    env = make_env()
    env.step(0)
    env.step(1)
    with pytest.raises(ValueError):
        env.step(0)


def test_row_wins():
    for row in range(3):
        env = make_env()
        for col in range(3):
            state, reward, done, info = env.step(row * 3 + col)
            if not done:
                env.current_player = 1
    assert done, f"Linha  {row} deveria produzir vitória."
    assert reward == 1.0
    assert info["winner"] == 1


def test_colum_wins():
    for col in range(3):
        env = make_env()
        for row in range(3):
            state, reward, done, info = env.step(row * 3 + col)
            if not done:
                env.current_player = 1
    assert done, f"Coluna {col} deveria produzir uma vitória."
    assert reward == 1.0
    assert info["winner"] == 1


def test_main_diagonal():
    env = make_env()
    for pos in [0, 4, 8]:
        state, reward, done, info = env.step(pos)
        if not done:
            env.current_player = 1

    assert done
    assert info["winner"] == 1


def test_anti_diagonal():
    env = make_env()
    for pos in [2, 4, 6]:
        state, reward, done, info = env.step(pos)
        if not done:
            env.current_player = 1
    assert done
    assert info["winner"] == 1


def test_draw():
    """
    Sequência que produz empate:
    X O X / O X X / O X O
    Posições: X=0,2,4,5,7  O=1,3,6,8
    """
    env = make_env()
    move_sequence = [0, 1, 2, 3, 4, 6, 5, 8, 7]
    done = False
    for action in move_sequence:
        state, reward, done, info = env.step(action)
    assert done
    assert reward == 0.0
    assert info["winner"] is None
