# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Instalar dependências
pip install -r requirements.txt pytest pytest-cov ruff

# Lint
ruff check .

# Rodar todos os testes unitários
pytest tests/ -v -m "not integration" --cov=src --cov-report=term-missing

# Rodar um único teste
pytest tests/test_environment.py::test_draw -v

# Treinar os agentes
python -m src.train
```

## Architecture

O projeto implementa um agente Q-Learning que aprende a jogar Jogo da Velha via self-play.

- **`src/environment.py`** — `TicTacToeEnv`: ambiente OpenAI-Gym-like. O tabuleiro é uma `list[int]` de 9 posições com valores `0` (vazio), `1` (X) e `-1` (O). `step()` retorna `(state_tuple, reward, done, info)`. O estado retornado é sempre um `tuple[int, ...]` para ser hashável (chave da Q-table).

- **`src/agent.py`** — `QLearningAgent`: Q-table implementada como `defaultdict(float)` com chave `(state, action)`. Usa ε-greedy para exploração. Suporta serialização via `pickle` (`save`/`load`).

- **`src/train.py`** — Treina dois agentes (`agents[1]` = X, `agents[-1]` = O) em self-play. Usa **atualização diferida**: o agente adversário só recebe seu update quando o jogador atual faz o próximo movimento (pois só então o `next_state` do adversário é conhecido. Ver `last_sa`).

## Padrão de commits

Seguir Conventional Commits:

```
<tipo>: <descrição curta em inglês>
```

Tipos usados no projeto: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`.

Exemplos dos commits anteriores:
- `feat: implement QLearningAgent`
- `fix: correct log_results bug in train loop`
- `docs: add train.md documenting periodic triggers`
- `chore: add pytest config and GitHub workflows`
