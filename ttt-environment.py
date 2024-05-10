import numpy as np
import torch as t
import tqdm
from enum import Enum, auto
# from typing import List, Tuple

class Game_result(Enum):
    Undecided = auto()
    Won_X = auto()
    Won_O = auto()
    Drawn = auto()

class Symbol(Enum):
    X = -1
    O = 1

class Ttt_Game:
    __slots__ = ["size", "streak", "board", "terminated", "result"]

    def __init__(self, size:int, streak:int):
        if size < streak:
            raise AttributeError(f"TTT Game board of size {size} cannot be smaller than the streak {streak}.")
        self.size = size
        self.streak = streak
        self.board = np.zeros((size, size), dtype=int)
        self.terminated = False
        self.result = Game_result.Undecided

    def Add_symbol(self, row:int, column:int, symbol:Symbol) -> None:
        if not 0 <= row < self.size:
            raise IndexError(f"Invalid row index to play at: {row}.")
        if not 0 <= column < self.size:
            raise IndexError(f"Invalid column index to play at: {column}.")
        if self.board[row, column] != 0:
            raise IndexError(f"Played in an occupied square at ({row}, {column}).")
        self.board[row, column] = symbol.value

    def Evaluate_game_v1(self) -> Game_result:
        for i in range(self.size):
            for j in range(self.size - self.streak + 1):
                if self.board[i,j] == 0:
                    continue
                if np.all(self.board[i,j:j+self.streak] == self.board[i,j]):
                    self.terminated = True
                    if self.board[i,j] == Symbol.X.value: 
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O

        for i in range(self.size - self.streak + 1):
            for j in range(self.size):
                if self.board[i,j] == 0:
                    continue
                if np.all(self.board[i:i+self.streak,j] == self.board[i,j]):
                    self.terminated = True
                    if self.board[i,j] == Symbol.X.value: 
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O

        for i in range(self.size - self.streak + 1):
            for j in range(self.size - self.streak + 1):
                if self.board[i,j] == 0:
                    continue
                if np.all((self.board[i+offset,j+offset] for offset in range(self.streak)) == self.board[i,j]):
                    self.terminated = True
                    if self.board[i,j] == Symbol.X.value: 
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O


        if np.all(self.board != 0) and self.result == Game_result.Undecided:
            self.terminated = True
            self.result = Game_result.Drawn
        return self.result
    
    def Evaluate_game_v2(self) -> Game_result:
        for i in range(self.size - self.streak + 1):
            for j in range(self.size - self.streak + 1):
                if self.board[i,j] == 0:
                    continue
                if (np.all(self.board[i,j:j+self.streak] == self.board[i,j]) or 
                    np.all(self.board[i:i+self.streak,j] == self.board[i,j]) or 
                    np.all((self.board[i+offset,j+offset] for offset in range(self.streak)) == self.board[i,j])):
                    self.terminated = True
                    if self.board[i,j] == Symbol.X.value: 
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O
        for i in range(self.size - self.streak + 1, self.size):
            for j in range(self.size - self.streak + 1):
                if self.board[i,j] == 0:
                    continue
                if np.all(self.board[i,j:j+self.streak] == self.board[i,j]):
                    self.terminated = True
                    if self.board[i,j] == Symbol.X.value: 
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O

        for i in range(self.size - self.streak + 1):
            for j in range(self.size - self.streak + 1, self.size):
                if self.board[i,j] == 0:
                    continue
                if np.all(self.board[i:i+self.streak,j] == self.board[i,j]):
                    self.terminated = True
                    if self.board[i,j] == Symbol.X.value: 
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O




        if np.all(self.board != 0) and self.result == Game_result.Undecided:
            self.terminated = True
            self.result = Game_result.Drawn
        return self.result

    def Valid_moves(self) -> np.ndarray[np.ndarray]:
        # print(np.indices(self.board))
        indices = np.where(self.board == 0)
        stacked_indices = np.transpose(np.vstack(indices))
        return stacked_indices
    
def Main():
    game = Ttt_Game(200,3)
    game.Add_symbol(1,2,Symbol.X)
    print(game.board)
    print(game.Valid_moves())
    print(game.Evaluate_game_v2())
    game.Add_symbol(0,2,Symbol.X)
    print(game.Evaluate_game_v2())

    game.Add_symbol(2,2,Symbol.X)
    print(game.board)
    print(game.Evaluate_game_v2())




if __name__ == "__main__":
    Main()

