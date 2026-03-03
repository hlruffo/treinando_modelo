**PRODUCT REQUIREMENTS DOCUMENT**

**Jogo da Velha com Q-Learning**

Projeto de Portfólio — Inteligência Artificial / Reinforcement Learning

## 1. Ficha do Produto

| **Produto**     | Jogo da Velha com Agente Q-Learning                                             |
|-----------------|---------------------------------------------------------------------------------|
| **Versão**      | 1.0.0 — MVP de Portfólio                                                        |
| **Autor**       | Hugo                                                                            |
| **Data**        | Março 2026                                                                      |
| **Status**      | Em desenvolvimento                                                              |
| **Tecnologias** | Python 3.11+, NumPy, defaultdict, Matplotlib                                    |
| **Objetivo**    | Demonstrar domínio de Reinforcement Learning via agente treinado com Q-Learning |

## 2. Visão Geral

Este projeto consiste no desenvolvimento de um agente de Inteligência Artificial capaz de aprender a jogar Jogo da Velha de forma autônoma, utilizando o algoritmo Q-Learning — uma abordagem clássica de Reinforcement Learning (RL) baseada em tabela de valores.

O agente aprende exclusivamente pela interação consigo mesmo (self-play), sem nenhuma regra explícita programada sobre estratégia do jogo. Após 50.000 episódios de treinamento, o agente deve ser capaz de jogar de forma próxima ao ótimo, nunca perdendo uma partida para um humano que jogue corretamente.

O projeto será publicado no portfólio do desenvolvedor como demonstração prática de conceitos de IA/ML, incluindo código limpo, visualizações de aprendizado e documentação técnica.

## 3. Objetivos

### 3.1 Objetivos Técnicos

- Implementar o ambiente do Jogo da Velha como classe reutilizável, seguindo o padrão OpenAI Gym.
- Implementar o agente Q-Learning com suporte a epsilon-greedy, taxa de aprendizado (α) e fator de desconto (γ) configuráveis.
- Treinar dois agentes via self-play por no mínimo 50.000 episódios.
- Implementar decaimento automático do epsilon ao longo do treinamento.
- Gerar visualizações do aprendizado (curva de reward, taxa de vitória, mapa de calor de Q-values).
- Permitir jogar contra o agente treinado via interface de terminal.

### 3.2 Objetivos de Portfólio

- Código bem documentado com docstrings e type hints.
- README completo com explicação do algoritmo, como rodar e resultados esperados.
- Gráficos exportados em PNG para uso no LinkedIn/GitHub.
- Estrutura de projeto organizada e publicada no GitHub com commits semânticos.

## 4. Escopo

### 4.1 Incluso (In Scope)

- Ambiente do Jogo da Velha 3×3 com validação de movimentos e detecção de fim de jogo.
- Agente Q-Learning com Q-table implementada via defaultdict.
- Self-play: dois agentes (X e O) treinando simultaneamente.
- Sistema de recompensas: +1 vitória, -1 derrota, 0 empate ou movimento neutro.
- Decaimento de epsilon (exploração → exploração pura).
- Interface de terminal para jogar contra o agente treinado.
- Exportação da Q-table treinada (pickle/JSON) para reutilização.
- Visualizações: curva de reward médio por episódio, taxa de vitória ao longo do tempo.

### 4.2 Fora do Escopo (Out of Scope)

- Interface gráfica (GUI) — apenas terminal na v1.0.
- Algoritmo Minimax ou MCTS — versões futuras.
- Deep Q-Network (DQN) com PyTorch/TensorFlow — versão futura.
- Deploy web ou API REST.
- Suporte a tabuleiros maiores que 3×3.

## 5. User Stories &amp; Critérios de Aceite

| **ID**    | **User Story**                                                                                                            | **Critério de Aceite**                                                                                                                    |
|-----------|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| **US-01** | Como desenvolvedor, quero um ambiente do jogo isolado e testável, para garantir que a lógica do tabuleiro esteja correta. | Testes unitários cobrem: movimentos inválidos, detecção de vitória em todos os eixos, empate e reset correto do estado.                   |
| **US-02** | Como agente IA, quero aprender a jogar através de self-play, para que meu desempenho melhore ao longo do tempo.           | Após 50k episódios, a taxa de vitória do agente X vs. aleatório deve ser ≥ 85%. Curva de aprendizado deve mostrar tendência crescente.    |
| **US-03** | Como usuário, quero jogar contra o agente treinado no terminal, para ver o resultado do treinamento na prática.           | Interface exibe tabuleiro em ASCII após cada jogada. Agente responde em < 100ms. Resultado (vitória/derrota/empate) é exibido claramente. |
| **US-04** | Como recrutador, quero ver visualizações do aprendizado exportadas, para entender o progresso do treinamento.             | Gráficos gerados em PNG: curva de reward médio (janela 500 ep.) e taxa de vitória acumulada. Legenda e título incluídos.                  |
| **US-05** | Como desenvolvedor, quero salvar e carregar o agente treinado, para não precisar retreinar a cada execução.               | Agente treinado é salvo em arquivo e carregado corretamente, mantendo a mesma performance sem novo treinamento.                           |

## 6. Requisitos Técnicos

### 6.1 Ambiente (TicTacToeEnv)

- Estado: tupla de 9 inteiros (0=vazio, 1=X, -1=O).
- Métodos obrigatórios: reset(), step(action), available\_actions(), render().
- Detecção de vitória: checar 8 combinações (3 linhas, 3 colunas, 2 diagonais).
- Recompensa imediata: +1 (ganhou), -1 (perdeu), 0 (neutro ou empate).

### 6.2 Agente (QLearningAgent)

Equação de atualização da Q-table:

**Q(s,a) ← Q(s,a) + α · [r + γ · max Q(s',a') - Q(s,a)]**

- Parâmetros padrão: α = 0.3, γ = 0.9, ε inicial = 0.3, ε mínimo = 0.05.
- Política epsilon-greedy: exploração aleatória com prob. ε, greedy caso contrário.
- Q-table implementada como defaultdict(float) para eficiência de memória.

### 6.3 Treinamento

- Mínimo de 50.000 episódios de self-play (dois agentes simultâneos).
- Decaimento do epsilon a cada 5.000 episódios: ε *= 0.95.
- Recompensa espelhada: agente perdedor recebe -reward do ganhador no mesmo passo.
- Log de métricas a cada 1.000 episódios: reward médio, taxa de vitória/empate/derrota.

## 7. Estrutura do Projeto

**tic-tac-toe-rl/**

├── src/
  │   ├── environment.py    # TicTacToeEnv
  │   ├── agent.py          # QLearningAgent
  │   ├── train.py          # Loop de treinamento
  │   ├── play.py           # Interface terminal
  │   └── visualize.py      # Geração de gráficos
  ├── models/
  │   └── agent\_x.pkl       # Agente treinado
  ├── outputs/
  │   ├── reward\_curve.png
  │   └── win\_rate.png
  ├── tests/
  │   └── test\_environment.py
  ├── requirements.txt
  └── README.md

## 8. Métricas de Sucesso

| **Métrica**                           | **Meta**                  | **Como medir**                |
|---------------------------------------|---------------------------|-------------------------------|
| Taxa de vitória vs. jogador aleatório | **≥ 85%**                 | 100 partidas após treinamento |
| Taxa de derrota vs. jogador aleatório | **≤ 2%**                  | 100 partidas após treinamento |
| Convergência da Q-table               | **Estável após 40k eps.** | Variação do reward < 0.05     |
| Tempo de resposta do agente           | **&lt; 100ms por jogada** | time.perf_counter()           |
| Tamanho da Q-table                    | **&lt; 10.000 entradas**  | len(agent.q)                  |

## 9. Milestones &amp; Cronograma

| **Fase**   | **Entregas**                          | **Prazo**   | **Status**    |
|------------|---------------------------------------|-------------|---------------|
| **Fase 1** | Ambiente do jogo + testes unitários   | Semana 1    | **Planejado** |
| **Fase 2** | Agente Q-Learning + loop de self-play | Semana 1    | **Planejado** |
| **Fase 3** | Treinamento 50k episódios + métricas  | Semana 2    | **Planejado** |
| **Fase 4** | Visualizações + interface terminal    | Semana 2    | **Planejado** |
| **Fase 5** | README + publicação no GitHub         | Semana 3    | **Planejado** |

## 10. Riscos e Mitigações

- **Risco:** Convergência lenta:
- Q-table pode não convergir com parâmetros ruins. Mitigação: usar α=0.3, γ=0.9 como padrão; documentar sensibilidade dos hiperparâmetros.
- **Risco:** Overfitting para uma abertura:
- Agente pode não generalizar. Mitigação: garantir epsilon ≥ 0.05 mesmo após decaimento; baralhar ordem de jogadas no início..
- **Risco:** Q-table muito grande:
- Estados equivalentes por simetria inflam a tabela. Mitigação: implementar normalização de estado por simetria na v1.1 se necessário.

## 11. Backlog — Versões Futuras

### v1.1 — Melhorias de RL

- Simetria de estado: reduzir Q-table aproveitando rotações/reflexões do tabuleiro.
- Experience Replay: armazenar transições e amostrar mini-batches.

### v2.0 — Deep Q-Network

- Substituir Q-table por rede neural (PyTorch) — DQN com replay buffer e target network.
- Visualização do gradient flow e loss durante treinamento.

### v3.0 — Interface Web

- Frontend React com tabuleiro interativo.
- Backend FastAPI servindo o agente treinado como API REST.

*Documento interno de portfólio — v1.0 — Março 2026*