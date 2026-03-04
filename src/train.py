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
        last_sa: dict[int, tuple | None] = {
            1: None,
            -1: None,
        }  # Última experiência (state, action) de cada jogador — atualização diferida
        reward_x = 0.0
        result = "draw"
        done = False

        while not done:
            player = env.current_player
            agent = agents[player]
            avail = env.available_actions()
            action = agent.choose_action(state, avail)
            next_state, _, done, info = env.step(action)

            if done:
                winner = info["winner"]
                if winner is not None:
                    agent.update(state, action, 1.0, next_state, [], True)
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

        if ep % 50_000 == 0:
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
