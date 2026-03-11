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
    ax.plot(
        x_smooth,
        smoothed,
        color="navy",
        linewidth=2,
        label=f"Média móvel ({window} eps)",
    )
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
