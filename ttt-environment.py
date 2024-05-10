import numpy as np
import torch as t
import tqdm
from enum import Enum, auto
from scipy.signal import convolve2d
from dataclasses import dataclass

class Game_result(Enum):
    Undecided = auto()
    Won_X = auto()
    Won_O = auto()
    Drawn = auto()

class Symbol(Enum):
    Empty = 0
    X = -1
    O = 1

@dataclass
class Action:
    row:int
    column:int
    symbol:Symbol

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

    def play(self, action:Action) -> None:
        self.board[action.row, action.column] = action.symbol.value

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
    
    def is_valid_move(self, action:Action) -> bool:
        if not 0 <= action.row < self.size:
            return False
        if not 0 <= action.column < self.size:
            return False
        if self.board[action.row, action.column] != Symbol.Empty.value:
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
        observation = self.get_observation()

        return observation
        

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
            
    
    def next_obs(self, action:Action) -> tuple[np.ndarray, t.float16, bool]:
        
        # Action validity checks
        if action.symbol != self.symbol_to_play:
            raise Exception(f"Wrong symbol to play: got {action.symbol} but expected {self.symbol_to_play}.")
        if not self.game.is_valid_move(action):
            return self.game.board.flatten(), self.reward_values["invalid_move"], self.terminal
        
        # Play and get new observation, reward, and terminal flag 
        self.game.play(action)

        self.result = self.game.evaluate_game()
        self.terminal = self.game.terminal
        reward = self.get_reward()
        observation = self.get_observation()

        return observation, reward, self.terminal

    


def main():

    game = Ttt_Game(10,4)
    game.add_symbol(1,2,Symbol.X)
    print(game.board)




if __name__ == "__main__":
    main()

