from __future__ import annotations


class TicTacToeEnv:
    WIN_COMBINATIONS = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),  # linhas
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),  # colunas
        (0, 4, 8),
        (2, 4, 6),  # diagonal
    ]

    def __init__(self) -> None:
        self.board: list[int] = [0] * 9
        self.current_player: int = 1

    def reset(self) -> tuple[int, ...]:
        self.board = [0] * 9
        self.current_player = 1
        return tuple(self.board)

    def available_actions(self) -> list[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action: int) -> tuple[tuple[int, ...], float, bool, dict]:
        if self.board[action] != 0:
            raise ValueError(f"Ação inválida: posição {action} já está ocupada")

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

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        for row in range(3):
            print("".join(symbols[self.board[row * 3 + col]] for col in range(3)))
        print()
