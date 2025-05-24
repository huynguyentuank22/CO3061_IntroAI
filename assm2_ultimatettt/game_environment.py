import numpy as np
import torch

class UltimateTTTEnv:
    def __init__(self):
        # 9x9 board: 0 = empty, 1 = player 1, 2 = player 2
        self.board = np.zeros((9, 9), dtype=int)
        # 3x3 macroboard: 0 = playable, 1 = won by player 1, 2 = won by player 2, -1 = draw
        self.macroboard = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.last_move = None
        self.move_number = 0
        self.done = False
        self.winner = None
        
    def reset(self):
        self.board = np.zeros((9, 9), dtype=int)
        self.macroboard = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.last_move = None
        self.move_number = 0
        self.done = False
        self.winner = None
        return self.get_state()
    
    def get_valid_moves(self):
        """Return list of valid moves indices (0-80)"""
        valid_moves = []
        
        # If it's the first move or we're sent to a finished board
        if self.last_move is None:
            # All empty cells are valid
            for i in range(9):
                for j in range(9):
                    if self.board[i][j] == 0:
                        valid_moves.append(i * 9 + j)
            return valid_moves
        
        # Get the local board based on the last move
        target_local_i, target_local_j = self.last_move % 9 // 3, self.last_move % 9 % 3
        
        # Check if the target local board is already decided
        macro_state = self.macroboard[target_local_i][target_local_j]
        if macro_state != 0:  # If board is already won or drawn
            # Can play in any empty cell on any active board
            for local_i in range(3):
                for local_j in range(3):
                    if self.macroboard[local_i][local_j] == 0:  # Board is active
                        for i in range(3):
                            for j in range(3):
                                global_i = local_i * 3 + i
                                global_j = local_j * 3 + j
                                if self.board[global_i][global_j] == 0:
                                    valid_moves.append(global_i * 9 + global_j)
        else:
            # Must play in the specific local board
            for i in range(3):
                for j in range(3):
                    global_i = target_local_i * 3 + i
                    global_j = target_local_j * 3 + j
                    if self.board[global_i][global_j] == 0:
                        valid_moves.append(global_i * 9 + global_j)
        
        return valid_moves
    
    def check_local_win(self, board, player):
        """Check if player has won on the given 3x3 board"""
        # Check rows
        for i in range(3):
            if board[i][0] == player and board[i][1] == player and board[i][2] == player:
                return True
        # Check columns
        for j in range(3):
            if board[0][j] == player and board[1][j] == player and board[2][j] == player:
                return True
        # Check diagonals
        if board[0][0] == player and board[1][1] == player and board[2][2] == player:
            return True
        if board[0][2] == player and board[1][1] == player and board[2][0] == player:
            return True
        return False
    
    def check_local_board_status(self, local_i, local_j):
        """Check status of local 3x3 board (win, draw, or still playable)"""
        local_board = np.zeros((3, 3))
        
        # Extract the local board
        for i in range(3):
            for j in range(3):
                local_board[i][j] = self.board[local_i * 3 + i][local_j * 3 + j]
        
        # Check if player 1 won
        if self.check_local_win(local_board, 1):
            return 1
        
        # Check if player 2 won
        if self.check_local_win(local_board, 2):
            return 2
        
        # Check if it's a draw (all cells filled)
        if np.all(local_board != 0):
            return -1
        
        # Still playable
        return 0
    
    def update_macroboard(self):
        """Update the status of each local board in the macroboard"""
        for i in range(3):
            for j in range(3):
                self.macroboard[i][j] = self.check_local_board_status(i, j)
    
    def check_game_over(self):
        """Check if the game is over"""
        # Check if player 1 won
        if self.check_local_win(self.macroboard, 1):
            self.done = True
            self.winner = 1
            return True
            
        # Check if player 2 won
        if self.check_local_win(self.macroboard, 2):
            self.done = True
            self.winner = 2
            return True
            
        # Check for draw (all local boards decided)
        if np.all(self.macroboard != 0):
            self.done = True
            self.winner = None
            return True
            
        # Check if no valid moves
        if len(self.get_valid_moves()) == 0:
            self.done = True
            self.winner = None
            return True
            
        return False
    
    def step(self, action):
        """Take action (0-80) and return next state, reward, done, info"""
        if self.done:
            return self.get_state(), 0, True, {}
            
        valid_moves = self.get_valid_moves()
        if action not in valid_moves:
            # Invalid move, punish heavily
            return self.get_state(), -10, False, {"invalid_move": True}
        
        # Apply move
        row, col = divmod(action, 9)
        self.board[row][col] = self.current_player
        self.last_move = action
        self.move_number += 1
        
        # Update macroboard
        self.update_macroboard()
        
        # Check if game is over
        game_over = self.check_game_over()
        
        reward = 0
        if game_over:
            if self.winner == self.current_player:
                reward = 1.0  # Win
            elif self.winner is None:
                reward = 0.1  # Draw
            else:
                reward = -1.0  # Loss
        
        # Switch player
        self.current_player = 3 - self.current_player  # Toggle between 1 and 2
        
        return self.get_state(), reward, self.done, {}
    
    def get_state(self):
        """Return the current state of the environment"""
        return {
            "board": self.board.copy(),
            "macroboard": self.macroboard.copy(),
            "current_player": self.current_player,
            "last_move": self.last_move,
            "move_number": self.move_number,
            "valid_moves": self.get_valid_moves()
        }
    
    def get_state_tensors(self):
        """Convert state to tensors for the neural network"""
        # Channel 1: board state
        board_channel = self.board.copy()
        
        # Channel 2: macroboard (3x3 → broadcast to 9x9)
        macro_full = np.repeat(np.repeat(self.macroboard, 3, axis=0), 3, axis=1)
        
        # Channel 3: current player
        player_channel = np.full((9, 9), self.current_player)
        
        # Channel 4: valid moves
        valid_channel = np.zeros((9, 9))
        valid_ids = self.get_valid_moves()
        for move_id in valid_ids:
            i, j = divmod(move_id, 9)
            valid_channel[i][j] = 1
            
        # Stack channels
        board_tensor = np.stack([board_channel, macro_full, player_channel, valid_channel], axis=0).astype(np.float32)
        
        # MLP features
        move_number = self.move_number / 81.0  # Normalize
        agent_level = 1.0  # Can be adjusted based on difficulty
        game_result = 0.0  # Placeholder, will be determined at the end
        
        mlp_features = np.array([move_number, agent_level, game_result], dtype=np.float32)
        
        # Valid moves mask for action selection
        mask = np.full(81, -np.inf, dtype=np.float32)
        mask[valid_ids] = 0.0
        
        return torch.tensor(board_tensor), torch.tensor(mlp_features), torch.tensor(mask)
