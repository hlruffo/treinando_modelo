# Jogo da Velha com Q-Learning

Agente de Inteligência Artificial que aprende a jogar Jogo da Velha de forma autônoma via **Q-Learning** (Reinforcement Learning tabular). O agente treina exclusivamente por **self-play** — sem regras de estratégia explícitas — e após 50.000 episódios atinge taxa de vitória ≥ 85% contra um jogador aleatório.

> Projeto de portfólio — IA / Reinforcement Learning — Março 2026

---

## Como funciona

O Q-Learning é um algoritmo de RL model-free que aprende uma função de valor de ação **Q(s, a)** por meio de interações com o ambiente. A cada passo, a Q-table é atualizada pela equação de Bellman:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') − Q(s, a)]
```

| Símbolo | Significado                      | Valor padrão |
|---------|----------------------------------|--------------|
| `α`     | Taxa de aprendizado (learning rate) | 0.3       |
| `γ`     | Fator de desconto (discount factor) | 0.9       |
| `ε`     | Taxa de exploração inicial (epsilon-greedy) | 0.3 |
| `ε_min` | Epsilon mínimo após decaimento   | 0.05         |

**Self-play:** dois agentes (X e O) treinam simultaneamente. O agente perdedor recebe recompensa `-1` pelo seu último movimento; o vencedor recebe `+1`. Empates resultam em `0` para ambos.

**Decaimento de epsilon:** a cada 5.000 episódios, `ε *= 0.95`, incentivando exploração no início e exploração pura ao final.

---

## Estrutura do projeto

```
tic-tac-toe-rl/
├── src/
│   ├── __init__.py
│   ├── environment.py     # TicTacToeEnv — ambiente do jogo
│   ├── agent.py           # QLearningAgent — Q-table + política epsilon-greedy
│   ├── train.py           # Loop de treinamento self-play (50k episódios)
│   ├── play.py            # Interface terminal: humano vs. agente
│   └── visualize.py       # Geração de gráficos PNG
├── models/
│   └── agent_x.pkl        # Agente X treinado (salvo via pickle)
├── outputs/
│   ├── reward_curve.png   # Curva de reward médio por episódio
│   └── win_rate.png       # Taxas de vitória/empate/derrota ao longo do tempo
├── tests/
│   └── test_environment.py
├── requirements.txt
└── README.md
```

---

## Instalação

**Requisitos:** Python 3.11+

```bash
git clone <repo-url>
cd tic-tac-toe-rl
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy
matplotlib
pytest
```

---

## Como usar

### 1. Treinar o agente

```bash
python src/train.py
```

Executa 50.000 episódios de self-play. O progresso é logado a cada 1.000 episódios:

```
Ep   1000 | ε=0.300 | avg_r=+0.123 | W=42.3% D=31.1% L=26.6% | Q=312
Ep   5000 | ε=0.285 | avg_r=+0.401 | W=61.8% D=25.4% L=12.8% | Q=1847
...
Ep  50000 | ε=0.072 | avg_r=+0.712 | W=87.2% D=10.1% L=2.7%  | Q=5823
```

O agente treinado é salvo em `models/agent_x.pkl`.

### 2. Gerar visualizações

```bash
python src/visualize.py
```

Gera dois gráficos em `outputs/`:
- `reward_curve.png` — curva de reward médio com média móvel
- `win_rate.png` — taxas de vitória, empate e derrota ao longo do treinamento

### 3. Jogar contra o agente

```bash
python src/play.py
```

Interface em terminal. Você joga como **O**, o agente como **X**.

```
=== Jogo da Velha — Você vs. Agente Q-Learning ===
Posições do tabuleiro:
0 | 1 | 2
3 | 4 | 5
6 | 7 | 8

Você joga como O. O agente joga como X.

. . .
. . .
. . .

Agente jogou na posição 4 (0.3ms)
. . .
. X .
. . .

Sua jogada [0, 1, 2, 3, 5, 6, 7, 8]:
```

### 4. Rodar os testes

```bash
cd tic-tac-toe-rl
pytest tests/ -v
```

---

## Métricas de sucesso

| Métrica | Meta |
|---------|------|
| Taxa de vitória vs. aleatório | **≥ 85%** (em 100 partidas) |
| Taxa de derrota vs. aleatório | **≤ 2%** (em 100 partidas) |
| Convergência da Q-table | Estável após 40k episódios |
| Tempo de resposta do agente | < 100ms por jogada |
| Tamanho da Q-table | < 10.000 entradas |

---

## Componentes principais

### `TicTacToeEnv`

Ambiente do jogo seguindo o padrão OpenAI Gym:

| Método | Descrição |
|--------|-----------|
| `reset()` | Reinicia o tabuleiro, retorna estado inicial |
| `step(action)` | Executa uma jogada, retorna `(state, reward, done, info)` |
| `available_actions()` | Lista posições livres no tabuleiro |
| `render()` | Imprime o tabuleiro em ASCII |

**Estado:** tupla de 9 inteiros — `0` (vazio), `1` (X), `-1` (O).

### `QLearningAgent`

| Método | Descrição |
|--------|-----------|
| `choose_action(state, available)` | Política epsilon-greedy |
| `update(s, a, r, s', available', done)` | Atualiza Q-table via equação de Bellman |
| `decay_epsilon(factor=0.95)` | Reduz epsilon até o mínimo configurado |
| `save(path)` / `load(path)` | Serializa/carrega o agente via pickle |

A Q-table é implementada como `defaultdict(float)` para eficiência de memória — apenas estados visitados são armazenados.

---

## Decisões de design

- **Atualização diferida:** a experiência `(s, a)` de cada agente só é atualizada quando o oponente faz a jogada seguinte, pois somente então o estado `s'` é conhecido.
- **Recompensa espelhada:** o agente perdedor recebe `-1` pelo seu último movimento ao final do episódio.
- **Q-table como defaultdict:** evita pré-alocar todos os ~5.478 estados únicos do jogo; estados não visitados retornam `0.0` por padrão.

---

## Roadmap

### v1.1 — Melhorias de RL
- Normalização de estado por simetria (rotações/reflexões) para reduzir Q-table
- Experience Replay com mini-batches

### v2.0 — Deep Q-Network (DQN)
- Substituir Q-table por rede neural (PyTorch)
- Replay buffer e target network

### v3.0 — Interface Web
- Frontend React com tabuleiro interativo
- Backend FastAPI servindo o agente como API REST

---

## Tecnologias

- **Python 3.11+**
- **NumPy** — operações numéricas e suavização de curvas
- **Matplotlib** — geração de gráficos de aprendizado
- **pytest** — testes unitários do ambiente

---

*Hugo — Março 2026*
