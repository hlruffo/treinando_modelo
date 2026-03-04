# Parâmetros do QLearningAgent

## Hiperparâmetros de treinamento

### `alpha` (taxa de aprendizado)
- **Tipo:** `float` | **Padrão:** `0.3`
- Controla o quanto o agente atualiza seus valores Q a cada experiência.
- `alpha = 1.0` → o agente substitui completamente o valor anterior pelo novo.
- `alpha → 0` → o agente aprende muito devagar, quase não atualiza.
- Valores típicos: `0.1` – `0.5`.

### `gamma` (fator de desconto)
- **Tipo:** `float` | **Padrão:** `0.9`
- Determina quanto o agente valoriza recompensas **futuras** em relação às imediatas.
- `gamma = 1.0` → recompensas futuras valem tanto quanto as imediatas (horizonte infinito).
- `gamma = 0.0` → o agente é completamente míope, só considera a recompensa imediata.
- Valores típicos: `0.9` – `0.99`.

### `epsilon` (exploração)
- **Tipo:** `float` | **Padrão:** `0.3`
- Probabilidade de o agente escolher uma ação **aleatória** em vez da melhor conhecida.
- Implementa a estratégia **ε-greedy**: equilibra exploração (tentar coisas novas) vs. explotação (usar o que já sabe).
- Começa alto e vai decaindo ao longo do treinamento via `decay_epsilon()`.

### `epsilon_min` (exploração mínima)
- **Tipo:** `float` | **Padrão:** `0.05`
- Limite inferior para `epsilon` após os decaimentos.
- Garante que o agente nunca pare completamente de explorar, evitando ficar preso em ótimos locais.

---

## Estado interno

### `q`
- **Tipo:** `defaultdict[tuple, float]`
- A **tabela Q** do agente: mapeia pares `(estado, ação)` para um valor numérico que representa a qualidade estimada de tomar aquela ação naquele estado.
- Inicializada com `0.0` para todos os pares não vistos.
- Atualizada pela equação de Bellman a cada chamada de `update()`:

```
Q(s, a) ← Q(s, a) + α * (r + γ * max_a' Q(s', a') - Q(s, a))
```

onde:
- `s` = estado atual, `a` = ação tomada
- `r` = recompensa recebida
- `s'` = próximo estado, `a'` = ações disponíveis no próximo estado
