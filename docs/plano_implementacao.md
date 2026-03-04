# Plano de Implementação — Jogo da Velha com Q-Learning

## Context

Projeto de portfólio de RL (Reinforcement Learning) conforme o PRD em `docs/PRD_JogoDaVelha_QLearning.md`. O objetivo é implementar um agente Q-Learning capaz de aprender Jogo da Velha via self-play, atingindo ≥85% de vitória contra jogador aleatório após 50.000 episódios de treinamento. Nenhum código de implementação existe ainda — apenas o PRD e a venv configurada.

## Estrutura de Diretórios a Criar

```
tic-tac-toe-rl/
├── src/
│   ├── __init__.py
│   ├── environment.py     # TicTacToeEnv
│   ├── agent.py           # QLearningAgent
│   ├── train.py           # Loop de treinamento self-play
│   ├── play.py            # Interface terminal (humano vs agente)
│   └── visualize.py       # Geração de gráficos PNG
├── models/                # Agente treinado salvo (pickle)
├── outputs/               # Gráficos exportados
├── tests/
│   └── test_environment.py
├── requirements.txt       # Dependências do sub-projeto
└── README.md              # Será criado na Fase 5
```

## Arquivos a Modificar

- `requirements.txt` (raiz) — adicionar numpy, matplotlib, pytest

---

## Fase 1 — Ambiente + Testes

### `tic-tac-toe-rl/src/environment.py`

```python
from __future__ import annotations


class TicTacToeEnv:
    WIN_COMBINATIONS = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # linhas
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colunas
        (0, 4, 8), (2, 4, 6),             # diagonais
    ]

    def __init__(self) -> None:
        self.board: list[int] = [0] * 9
        self.current_player: int = 1  # 1=X, -1=O

    def reset(self) -> tuple[int, ...]:
        self.board = [0] * 9
        self.current_player = 1
        return tuple(self.board)

    def available_actions(self) -> list[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action: int) -> tuple[tuple[int, ...], float, bool, dict]:
        if self.board[action] != 0:
            raise ValueError(f"Ação inválida: posição {action} já está ocupada.")

        self.board[action] = self.current_player
        winner = self._check_winner()

        if winner is not None:
            return tuple(self.board), 1.0, True, {"winner": winner}

        if not self.available_actions():
            return tuple(self.board), 0.0, True, {"winner": None}

        self.current_player *= -1
        return tuple(self.board), 0.0, False, {"winner": None}

    def _check_winner(self) -> int | None:
        for a, b, c in self.WIN_COMBINATIONS:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        return None

    def render(self) -> None:
        symbols = {0: ".", 1: "X", -1: "O"}
        for row in range(3):
            print(" ".join(symbols[self.board[row * 3 + col]] for col in range(3)))
        print()
```

### `tic-tac-toe-rl/tests/test_environment.py`

```python
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
        env.step(0)  # posição já ocupada


def test_row_wins():
    for row in range(3):
        env = make_env()
        for col in range(3):
            state, reward, done, info = env.step(row * 3 + col)
            if not done:
                env.current_player = 1  # força X a continuar jogando
        assert done, f"Linha {row} deveria produzir vitória"
        assert reward == 1.0
        assert info["winner"] == 1


def test_column_wins():
    for col in range(3):
        env = make_env()
        for row in range(3):
            state, reward, done, info = env.step(row * 3 + col)
            if not done:
                env.current_player = 1
        assert done, f"Coluna {col} deveria produzir vitória"
        assert reward == 1.0
        assert info["winner"] == 1


def test_main_diagonal_win():
    env = make_env()
    for pos in [0, 4, 8]:
        state, reward, done, info = env.step(pos)
        if not done:
            env.current_player = 1
    assert done
    assert info["winner"] == 1


def test_anti_diagonal_win():
    env = make_env()
    for pos in [2, 4, 6]:
        state, reward, done, info = env.step(pos)
        if not done:
            env.current_player = 1
    assert done
    assert info["winner"] == 1


def test_draw():
    # Sequência que produz empate:
    # X O X / O X X / O X O
    # Posições: X=0,2,4,5,7  O=1,3,6,8
    env = make_env()
    move_sequence = [0, 1, 2, 3, 4, 6, 5, 8, 7]
    done = False
    for action in move_sequence:
        state, reward, done, info = env.step(action)
    assert done
    assert reward == 0.0
    assert info["winner"] is None
```

---

## Fase 2 — Agente Q-Learning

### `tic-tac-toe-rl/src/agent.py`

```python
from __future__ import annotations

import pickle
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.3,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        epsilon_min: float = 0.05,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.q: defaultdict[tuple, float] = defaultdict(float)

    def choose_action(self, state: tuple, available_actions: list[int]) -> int:
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        q_values = {a: self.q[(state, a)] for a in available_actions}
        return max(q_values, key=q_values.get)  # type: ignore[arg-type]

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_available: list[int],
        done: bool,
    ) -> None:
        current_q = self.q[(state, action)]
        if done or not next_available:
            target = reward
        else:
            max_next_q = max(self.q[(next_state, a)] for a in next_available)
            target = reward + self.gamma * max_next_q
        self.q[(state, action)] += self.alpha * (target - current_q)

    def decay_epsilon(self, factor: float = 0.95) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * factor)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "QLearningAgent":
        with open(path, "rb") as f:
            return pickle.load(f)
```

---

## Fase 3 — Treinamento

### `tic-tac-toe-rl/src/train.py`

Estratégia de atualização diferida: a experiência `(s, a)` de cada agente só é atualizada quando o *oponente* faz a jogada seguinte — pois somente então sabemos o estado que o agente enfrentará a seguir. Na saída do episódio, o perdedor recebe -1 pelo seu último movimento.

```python
from __future__ import annotations

import os

from src.agent import QLearningAgent
from src.environment import TicTacToeEnv


def train(episodes: int = 50_000) -> dict:
    env = TicTacToeEnv()
    agents: dict[int, QLearningAgent] = {
        1: QLearningAgent(),
        -1: QLearningAgent(),
    }

    history: dict[str, list] = {"rewards_x": [], "win": [], "draw": [], "loss": []}
    log_rewards: list[float] = []
    log_results: list[str] = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        # Última experiência (state, action) de cada jogador — atualização diferida
        last_sa: dict[int, tuple | None] = {1: None, -1: None}
        done = False
        reward_x = 0.0
        result = "draw"

        while not done:
            player = env.current_player
            agent = agents[player]
            avail = env.available_actions()
            action = agent.choose_action(state, avail)
            next_state, _, done, info = env.step(action)

            if done:
                winner = info["winner"]
                if winner is not None:
                    # Quem ganhou recebe +1 pelo movimento vencedor
                    agent.update(state, action, 1.0, next_state, [], True)
                    # Oponente recebe -1 pelo seu último movimento
                    other = -player
                    if last_sa[other] is not None:
                        ps, pa = last_sa[other]
                        agents[other].update(ps, pa, -1.0, next_state, [], True)
                    result = "win" if winner == 1 else "loss"
                    reward_x = float(winner)
                else:
                    # Empate: ambos recebem 0
                    agent.update(state, action, 0.0, next_state, [], True)
                    other = -player
                    if last_sa[other] is not None:
                        ps, pa = last_sa[other]
                        agents[other].update(ps, pa, 0.0, next_state, [], True)
                    result = "draw"
                    reward_x = 0.0
            else:
                # Movimento intermediário: atualiza a experiência do OPONENTE
                # pois ele agora sabe qual estado enfrentará a seguir (next_state)
                other = -player
                if last_sa[other] is not None:
                    ps, pa = last_sa[other]
                    agents[other].update(
                        ps, pa, 0.0, next_state, env.available_actions(), False
                    )
                last_sa[player] = (state, action)

            state = next_state

        log_rewards.append(reward_x)
        log_results.append(result)

        if ep % 5_000 == 0:
            agents[1].decay_epsilon()
            agents[-1].decay_epsilon()

        if ep % 1_000 == 0:
            recent_r = log_rewards[-1_000:]
            recent_res = log_results[-1_000:]
            avg_r = sum(recent_r) / 1_000
            wr = recent_res.count("win") / 1_000
            dr = recent_res.count("draw") / 1_000
            lr = recent_res.count("loss") / 1_000
            history["rewards_x"].append(avg_r)
            history["win"].append(wr)
            history["draw"].append(dr)
            history["loss"].append(lr)
            print(
                f"Ep {ep:>6} | ε={agents[1].epsilon:.3f} | "
                f"avg_r={avg_r:+.3f} | W={wr:.1%} D={dr:.1%} L={lr:.1%} | "
                f"Q={len(agents[1].q)}"
            )

    os.makedirs("models", exist_ok=True)
    agents[1].save("models/agent_x.pkl")
    print("\nAgente X salvo em models/agent_x.pkl")
    return history


if __name__ == "__main__":
    train()
```

---

## Fase 4 — Visualizações + Interface Terminal

### `tic-tac-toe-rl/src/visualize.py`

```python
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_reward_curve(history: dict, window: int = 500) -> None:
    rewards = history["rewards_x"]
    x = list(range(1, len(rewards) + 1))

    kernel = min(window // 1_000, len(rewards))
    if kernel > 1:
        smoothed = np.convolve(rewards, np.ones(kernel) / kernel, mode="valid").tolist()
        x_smooth = list(range(kernel, len(rewards) + 1))
    else:
        smoothed = rewards
        x_smooth = x

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, rewards, alpha=0.3, color="steelblue", label="Reward médio (1k eps)")
    ax.plot(x_smooth, smoothed, color="navy", linewidth=2, label=f"Média móvel ({window} eps)")
    ax.set_xlabel("Checkpoint (×1.000 episódios)")
    ax.set_ylabel("Reward Médio do Agente X")
    ax.set_title("Curva de Aprendizado — Q-Learning Jogo da Velha")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/reward_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Salvo: outputs/reward_curve.png")


def plot_win_rate(history: dict) -> None:
    x = list(range(1, len(history["win"]) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, history["win"], color="green", label="Vitória")
    ax.plot(x, history["draw"], color="orange", label="Empate")
    ax.plot(x, history["loss"], color="red", label="Derrota")
    ax.set_xlabel("Checkpoint (×1.000 episódios)")
    ax.set_ylabel("Taxa")
    ax.set_title("Taxas de Resultado — Agente X (Self-Play)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/win_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Salvo: outputs/win_rate.png")


if __name__ == "__main__":
    history_path = "outputs/history.json"
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_reward_curve(history)
        plot_win_rate(history)
    else:
        print("Nenhum histórico encontrado. Execute train.py primeiro.")
```

### `tic-tac-toe-rl/src/play.py`

```python
from __future__ import annotations

import time

from src.agent import QLearningAgent
from src.environment import TicTacToeEnv


def play() -> None:
    env = TicTacToeEnv()
    agent = QLearningAgent.load("models/agent_x.pkl")
    agent.epsilon = 0.0  # modo greedy puro

    print("=== Jogo da Velha — Você vs. Agente Q-Learning ===")
    print("Posições do tabuleiro:")
    print("0 | 1 | 2\n3 | 4 | 5\n6 | 7 | 8\n")
    print("Você joga como O. O agente joga como X.\n")

    state = env.reset()
    env.render()

    while True:
        # Agente joga (X = 1)
        avail = env.available_actions()
        t0 = time.perf_counter()
        action = agent.choose_action(state, avail)
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        print(f"Agente jogou na posição {action} ({elapsed_ms:.1f}ms)")
        state, _, done, info = env.step(action)
        env.render()
        if done:
            print("Agente X venceu!" if info["winner"] == 1 else "Empate!")
            break

        # Humano joga (O = -1)
        avail = env.available_actions()
        while True:
            try:
                human_action = int(input(f"Sua jogada {avail}: "))
                if human_action in avail:
                    break
                print("Posição inválida ou ocupada. Tente novamente.")
            except ValueError:
                print("Digite um número de 0 a 8.")

        state, _, done, info = env.step(human_action)
        env.render()
        if done:
            print("Você venceu!" if info["winner"] == -1 else "Empate!")
            break


if __name__ == "__main__":
    play()
```

---

## Fase 5 — README (fora do escopo imediato)

Será criado após validação das métricas de sucesso.

---

## Atualização do requirements.txt (raiz)

Adicionar ao `requirements.txt` existente:
```
numpy
matplotlib
pytest
```
(manter `docling` existente)

---

## Verificação / Testes

1. `cd tic-tac-toe-rl && pytest tests/ -v` — todos os testes unitários passam
2. `python src/train.py` — treina 50k episódios, salva `models/agent_x.pkl`
3. `python src/visualize.py` — gera PNGs em `outputs/`
4. Avaliar vs. jogador aleatório (100 partidas): win rate ≥85%, loss rate ≤2%
5. `python src/play.py` — interface terminal funcional
6. `len(agent.q) < 10_000` — tamanho da Q-table dentro do limite
