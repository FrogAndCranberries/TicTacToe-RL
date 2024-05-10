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
    __slots__ = ["size", "streak", "board", "terminal", "result"]

    def __init__(self, size:int, streak:int):
        if size < streak:
            raise AttributeError(f"TTT Game board of size {size} cannot be smaller than the streak {streak}.")
        self.size = size
        self.streak = streak
        self.board = np.zeros((size, size), dtype=int)
        self.terminal = False
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
                    self.terminal = True
                    if self.board[i,j] == Symbol.X.value: 
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O
                    return self.result

        for i in range(self.size - self.streak + 1):
            for j in range(self.size):
                if self.board[i,j] == 0:
                    continue
                if np.all(self.board[i:i+self.streak,j] == self.board[i,j]):
                    self.terminal = True
                    if self.board[i,j] == Symbol.X.value: 
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O
                    return self.result

        for i in range(self.size - self.streak + 1):
            for j in range(self.size - self.streak + 1):
                if self.board[i,j] == 0:
                    continue
                if np.all((self.board[i+offset,j+offset] for offset in range(self.streak)) == self.board[i,j]):
                    self.terminal = True
                    if self.board[i,j] == Symbol.X.value: 
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O
                    return self.result


        if np.all(self.board != 0) and self.result == Game_result.Undecided:
            self.terminal = True
            self.result = Game_result.Drawn
        return self.result
    
    def Evaluate_game_v2(self) -> Game_result:
        for i in range(self.size - self.streak + 1):
            for j in range(self.size - self.streak + 1):
                if self.board[i,j] == 0:
                    continue
                if (np.all(self.board[i,j:j+self.streak] == self.board[i,j]) or 
                    np.all(self.board[i:i+self.streak,j] == self.board[i,j]) or 
                    np.all(np.array(self.board[i+offset,j+offset] for offset in range(self.streak)) == self.board[i,j])):
                    self.terminal = True
                    if self.board[i,j] == Symbol.X.value:
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O
                    return self.result

        for i in range(self.size - self.streak + 1, self.size):
            for j in range(self.size - self.streak + 1):
                if self.board[i,j] == 0:
                    continue
                if np.all(self.board[i,j:j+self.streak] == self.board[i,j]):
                    self.terminal = True
                    if self.board[i,j] == Symbol.X.value: 
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O
                    return self.result

        for i in range(self.size - self.streak + 1):
            for j in range(self.size - self.streak + 1, self.size):
                if self.board[i,j] == 0:
                    continue
                if np.all(self.board[i:i+self.streak,j] == self.board[i,j]):
                    self.terminal = True
                    if self.board[i,j] == Symbol.X.value: 
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O
                    return self.result

        for i in range(self.size - self.streak + 1):
            for j in range(self.streak - 1, self.size):
                if self.board[i,j] == 0:
                    continue
                if np.all(np.array(self.board[i+offset,j-offset] for offset in range(self.streak)) == self.board[i,j]):
                    self.terminal = True
                    if self.board[i,j] == Symbol.X.value:
                        self.result = Game_result.Won_X
                    else:
                        self.result = Game_result.Won_O
                    return self.result

        if np.all(self.board != 0) and self.result == Game_result.Undecided:
            self.terminal = True
            self.result = Game_result.Drawn
        return self.result
    
    def Is_valid_move(self, row:int, column:int) -> bool:
        if not 0 <= row < self.size:
            return False
        if not 0 <= column < self.size:
            return False
        if self.board[row, column] != 0:
            return False
        return True

    def Valid_moves(self) -> np.ndarray[np.ndarray]:
        # print(np.indices(self.board))
        indices = np.where(self.board == 0)
        stacked_indices = np.transpose(np.vstack(indices))
        return stacked_indices
    
class Ttt_environment:
    def __init__(self, size:int, streak:int, first_player:Symbol = Symbol.X, single_player:bool = True, reward_values:dict = {"valid_move": 0, "invalid_move": -10, "draw": -1, "win": 10, "loss": -10}):
        self.game = Ttt_Game(size, streak)
        self.size = size
        self.streak = streak
        self.obs_shape = size ** 2
        self.single_player = single_player
        self.player_symbol = first_player
        self.symbol_to_play = first_player
        self.reward_values = reward_values
        self.terminal = False
        self.result = Game_result.Undecided

    def Reset(self) -> np.ndarray:
        self.game = Ttt_Game(self.size, self.streak)
        self.terminal = False
        self.result = Game_result.Undecided
        return self.game.board.flatten()
        

    def Get_observation(self) -> np.ndarray:
        return self.game.board.flatten()
    
    def Get_reward(self) -> t.float16:
        match self.result:
            case Game_result.Undecided:
                return 0
            case Game_result.Drawn:
                return -1
            case Game_result.Won_O:
                if self.player_symbol == Symbol.O:
                    return 10
                else:
                    return -10
            case Game_result.Won_X:
                if self.player_symbol == Symbol.X:
                    return 10
                else:
                    return -10 
            
    
    def Action(self, row:int, col:int, symbol:Symbol) -> tuple[np.ndarray, t.float16, bool]:
        
        if symbol != self.symbol_to_play:
            raise Exception(f"Wrong symbol to play: got {symbol} but expected {self.symbol_to_play}.")
        if not self.game.Is_valid_move(row, col):
            return self.game.board.flatten(), self.reward_values["invalid_move"], self.terminal
        self.game.Add_symbol(row, col, symbol)
        self.result = self.game.Evaluate_game_v1()
        self.terminal = self.game.terminal



        reward = self.Get_reward()
        return self.game.board.flatten(), reward, self.terminal

    


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

