# Dependências e Importações do Projeto

Referência de todos os pacotes externos e módulos nativos usados no projeto **Jogo da Velha com Q-Learning**.

---

## Pacotes Externos (`requirements.txt`)

### `numpy`
**Usado em:** `src/visualize.py`

Biblioteca de computação numérica. No projeto é usada exclusivamente para `np.convolve`, que calcula a média móvel suavizada da curva de recompensa nos gráficos de aprendizado.

### `matplotlib`
**Usado em:** `src/visualize.py`

Biblioteca de geração de gráficos 2D. Usada para plotar e salvar em PNG:
- `outputs/reward_curve.png` — curva de recompensa média do Agente X ao longo do treinamento
- `outputs/win_rate.png` — taxas de vitória, empate e derrota por checkpoint

### `pytest`
**Usado em:** `tests/test_environment.py`

Framework de testes unitários. Usado para validar o comportamento do `TicTacToeEnv`: reset do estado, detecção de vitórias por linha/coluna/diagonal, empate, e rejeição de movimentos inválidos.

### `docling`
**Usado em:** outros scripts do repositório raiz (`docs/convert_to_md.py`)

Ferramenta de conversão de documentos (PDF, DOCX, HTML → Markdown). Não faz parte do subprojeto `tic-tac-toe-rl`, mas está listada no `requirements.txt` da raiz pois é usada nos utilitários de documentação do repositório.

---

## Módulos da Biblioteca Padrão Python

### `from __future__ import annotations`
**Usado em:** `environment.py`, `agent.py`, `train.py`, `visualize.py`, `play.py`

Ativa a avaliação lazy (postergada) das anotações de tipo. Permite usar sintaxes modernas como `list[int]`, `tuple[int, ...]` e `int | None` em versões do Python anteriores à 3.10 sem causar erro em tempo de execução.

### `pickle`
**Usado em:** `src/agent.py`

Módulo de serialização de objetos Python. Usado nos métodos `agent.save(path)` e `QLearningAgent.load(path)` para persistir e recuperar o agente treinado (incluindo toda a Q-table) em disco no formato `.pkl`.

### `random`
**Usado em:** `src/agent.py`

Gerador de números pseudoaleatórios. Usado em `choose_action` para implementar a política ε-greedy: com probabilidade `ε`, o agente escolhe uma ação aleatória (exploração); caso contrário, escolhe a ação de maior valor Q (explotação).

### `collections` (`defaultdict`)
**Usado em:** `src/agent.py`

Estrutura de dados que retorna um valor padrão para chaves inexistentes. A Q-table do agente é implementada como `defaultdict(float)`, mapeando pares `(estado, ação)` → valor Q, sem precisar inicializar explicitamente cada entrada.

### `os`
**Usado em:** `src/train.py`, `src/visualize.py`

Módulo de interface com o sistema operacional. Usado exclusivamente para `os.makedirs(..., exist_ok=True)`, que cria os diretórios `models/` e `outputs/` caso ainda não existam antes de salvar arquivos.

### `json`
**Usado em:** `src/visualize.py`

Módulo de serialização JSON. Usado no bloco `__main__` de `visualize.py` para carregar o histórico de treinamento salvo em `outputs/history.json` e regenerar os gráficos sem re-treinar o agente.

### `time`
**Usado em:** `src/play.py`

Módulo de temporização. Usado via `time.perf_counter()` para medir e exibir o tempo de resposta do agente em milissegundos durante a partida interativa no terminal.
