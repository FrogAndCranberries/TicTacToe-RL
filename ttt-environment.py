import numpy as np
import torch as t
import tqdm
from enum import Enum, auto
from scipy.signal import convolve2d

class Game_result(Enum):
    Undecided = auto()
    Won_X = auto()
    Won_O = auto()
    Drawn = auto()

class Symbol(Enum):
    Empty = 0
    X = -1
    O = 1

class Ttt_Game:
    __slots__ = ["size", "streak", "board", "terminal", "result"]

    def __init__(self, size:int, streak:int):
        if size < streak:
            raise AttributeError(f"TTT Game board of size {size} cannot be smaller than the streak {streak}.")
        self.size = size
        self.streak = streak
        self.board = np.full((size, size), Symbol.Empty.value, dtype=int)
        self.terminal = False
        self.result = Game_result.Undecided

    def add_symbol(self, row:int, column:int, symbol:Symbol) -> None:
        if not 0 <= row < self.size:
            raise IndexError(f"Invalid row index to play at: {row}.")
        if not 0 <= column < self.size:
            raise IndexError(f"Invalid column index to play at: {column}.")
        if self.board[row, column] != Symbol.Empty.value:
            raise IndexError(f"Played in an occupied square at ({row}, {column}).")
        self.board[row, column] = symbol.value

    def evaluate_game(self) -> Game_result:

        def check_for_win(bool_board:np.ndarray) -> bool:
            horizontal = convolve2d(bool_board, np.ones((self.streak,1),dtype=int), mode="valid")
            vertical = convolve2d(bool_board, np.ones((1,self.streak),dtype=int), mode="valid")
            diagonal = convolve2d(bool_board, np.eye((self.streak,),dtype=int), mode="valid")
            anti_diagonal = convolve2d(bool_board, np.fliplr(np.eye((self.streak,),dtype=int)), mode="valid")

            return (np.any(horizontal == self.streak) or 
                    np.any(vertical == self.streak) or 
                    np.any(diagonal == self.streak) or 
                    np.any(anti_diagonal == self.streak))
        
        board_X = self.board == Symbol.X
        board_O = self.board == Symbol.O

        if check_for_win(board_X):
            self.terminal = True
            self.result = Game_result.Won_X
            return self.result
        
        if check_for_win(board_O):
            self.terminal = True
            self.result = Game_result.Won_O
            return self.result
        
        if not np.any(self.board == Symbol.Empty.value):
            self.terminal = True
            self.result = Game_result.Drawn
        return self.result
    
    def is_valid_move(self, row:int, column:int) -> bool:
        if not 0 <= row < self.size:
            return False
        if not 0 <= column < self.size:
            return False
        if self.board[row, column] != Symbol.Empty.value:
            return False
        return True

    def get_valid_moves(self) -> np.ndarray[np.ndarray]:
        # print(np.indices(self.board))
        indices = np.where(self.board == Symbol.Empty.value)
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

    def reset(self) -> np.ndarray:
        self.game = Ttt_Game(self.size, self.streak)
        self.terminal = False
        self.result = Game_result.Undecided
        return self.game.board.flatten()
        

    def get_observation(self) -> np.ndarray:
        return self.game.board.flatten()
    
    def get_reward(self) -> t.float16:
        match self.result:
            case Game_result.Undecided:
                return self.reward_values["valid_move"]
            case Game_result.Drawn:
                return self.reward_values["draw"]
            case Game_result.Won_O:
                if self.player_symbol == Symbol.O:
                    return self.reward_values["win"]
                else:
                    return self.reward_values["loss"]
            case Game_result.Won_X:
                if self.player_symbol == Symbol.X:
                    return self.reward_values["win"]
                else:
                    return self.reward_values["loss"]
            
    
    def action(self, row:int, col:int, symbol:Symbol) -> tuple[np.ndarray, t.float16, bool]:
        
        if symbol != self.symbol_to_play:
            raise Exception(f"Wrong symbol to play: got {symbol} but expected {self.symbol_to_play}.")
        if not self.game.Is_valid_move(row, col):
            return self.game.board.flatten(), self.reward_values["invalid_move"], self.terminal
        self.game.Add_symbol(row, col, symbol)
        self.result = self.game.Evaluate_game_v1()
        self.terminal = self.game.terminal



        reward = self.Get_reward()
        return self.game.board.flatten(), reward, self.terminal

    


def main():
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
    main()

