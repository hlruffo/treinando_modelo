# train.py — Documentação

## Função `train(episodes)`

Treina dois agentes Q-Learning (`agents[1]` = X, `agents[-1]` = O) jogando Jogo da Velha um contra o outro por `episodes` partidas.

---

## Gatilhos periódicos (`if ep`)

### `if ep % 50_000 == 0` — Decaimento do epsilon

```python
agents[1].decay_epsilon()
agents[-1].decay_epsilon()
```

A cada **50.000 episódios**, reduz o epsilon de ambos os agentes.

Epsilon controla o balanço **exploração vs. explotação** (ε-greedy):
- Epsilon alto → agente escolhe ações aleatórias para explorar o espaço de estados.
- Epsilon baixo → agente confia no que já aprendeu (explotação).

Com o avanço do treino faz sentido explorar menos e exploitar mais, por isso o epsilon decai periodicamente.

---

### `if ep % 1_000 == 0` — Log e métricas

```python
recent_r   = log_rewards[-1_000:]
recent_res = log_results[-1_000:]
```

A cada **1.000 episódios**, coleta as últimas 1.000 partidas e calcula:

| Variável | Descrição |
|---|---|
| `avg_r` | Recompensa média do agente X nas últimas 1.000 partidas |
| `wr` | Taxa de vitória do agente X |
| `dr` | Taxa de empate |
| `lr` | Taxa de derrota do agente X |

Essas métricas são:
1. **Salvas em `history`** — para plotar curvas de aprendizado ao final do treino.
2. **Impressas no terminal** — para acompanhar o progresso em tempo real.

---

## Bug corrigido

Na linha 74, `recent_res` estava incorretamente atribuído a `log_rewards` (lista de floats) em vez de `log_results` (lista de strings `"win"`, `"draw"`, `"loss"`), fazendo com que `wr`, `dr` e `lr` sempre retornassem `0`.

**Correção aplicada:**
```python
# antes (bugado)
recent_res = log_rewards[-1_000:]

# depois (correto)
recent_res = log_results[-1_000:]
```
