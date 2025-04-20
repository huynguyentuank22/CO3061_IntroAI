import random
import time
import math
from abc import ABC, abstractmethod
from copy import deepcopy
import torch
import numpy as np
from model import UltimateTicTacToeModel

class Player(ABC):
    def __init__(self, mark):
        self.mark = mark
        
    @abstractmethod
    def get_move(self, board):
        """Return a move as (board_row, board_col, row, col)"""
        pass

class HumanPlayer(Player):
    def get_move(self, board):
        """Human player moves are handled by the UI"""
        pass

class RandomPlayer(Player):
    def get_move(self, board):
        """Make a random valid move"""
        moves = board.get_available_moves()
        if not moves:
            return None
        return random.choice(moves)

class MinimaxPlayer(Player):
    def __init__(self, mark, depth=3):
        super().__init__(mark)
        self.depth = depth
        self.opponent_mark = 'O' if mark == 'X' else 'X'
    
    def get_move(self, board):
        """Use minimax with alpha-beta pruning to find the best move"""
        moves = board.get_available_moves()
        if not moves:
            return None
        
        best_score = -float('inf')
        best_move = None
        
        for move in moves:
            board_copy = deepcopy(board)
            board_row, board_col, row, col = move
            board_copy.make_move(board_row, board_col, row, col)
            
            # Calculate score for this move
            score = self._minimax(board_copy, self.depth - 1, False, -float('inf'), float('inf'))
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move
    
    def _minimax(self, board, depth, is_maximizing, alpha, beta):
        """Minimax algorithm with alpha-beta pruning"""
        # Check terminal states
        if board.winner == self.mark:
            return 100 + depth
        elif board.winner == self.opponent_mark:
            return -100 - depth
        elif board.is_draw or depth == 0:
            return self._evaluate_board(board)
        
        moves = board.get_available_moves()
        
        if is_maximizing:
            best_score = -float('inf')
            for move in moves:
                board_copy = deepcopy(board)
                board_row, board_col, row, col = move
                board_copy.make_move(board_row, board_col, row, col)
                score = self._minimax(board_copy, depth - 1, False, alpha, beta)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for move in moves:
                board_copy = deepcopy(board)
                board_row, board_col, row, col = move
                board_copy.make_move(board_row, board_col, row, col)
                score = self._minimax(board_copy, depth - 1, True, alpha, beta)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score
    
    def _evaluate_board(self, board):
        """Simple heuristic evaluation function"""
        score = 0
        
        # Count small boards won by each player
        for row in range(3):
            for col in range(3):
                if board.boards[row][col].winner == self.mark:
                    score += 3
                elif board.boards[row][col].winner == self.opponent_mark:
                    score -= 3
        
        return score

class MCTSPlayer(Player):
    def __init__(self, mark, simulation_time=1.0):
        super().__init__(mark)
        self.simulation_time = simulation_time  # Time in seconds for MCTS
        self.opponent_mark = 'O' if mark == 'X' else 'X'
    
    def get_move(self, board):
        """Use Monte Carlo Tree Search to find the best move"""
        moves = board.get_available_moves()
        if not moves:
            return None
        
        if len(moves) == 1:
            return moves[0]
        
        # Initialize statistics for MCTS
        wins = {move: 0 for move in moves}
        plays = {move: 0 for move in moves}
        
        # Run simulations for the specified amount of time
        end_time = time.time() + self.simulation_time
        while time.time() < end_time:
            for move in moves:
                board_copy = deepcopy(board)
                board_row, board_col, row, col = move
                board_copy.make_move(board_row, board_col, row, col)
                
                # Run a random simulation from this move
                result = self._simulate(board_copy)
                
                # Update statistics
                if result == self.mark:
                    wins[move] += 1
                plays[move] += 1
        
        # Choose move with the best win rate
        best_move = None
        best_win_rate = -float('inf')
        
        for move in moves:
            if plays[move] == 0:
                continue
                
            win_rate = wins[move] / plays[move]
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_move = move
        
        return best_move
    
    def _simulate(self, board):
        """Simulate a random game from the current position"""
        board_copy = deepcopy(board)
        current_mark = self.opponent_mark  # We just made a move, so it's opponent's turn
        
        # Play random moves until the game is over
        while board_copy.winner is None and not board_copy.is_draw:
            moves = board_copy.get_available_moves()
            if not moves:
                break
                
            # Make a random move
            board_row, board_col, row, col = random.choice(moves)
            board_copy.make_move(board_row, board_col, row, col)
            current_mark = self.mark if current_mark == self.opponent_mark else self.opponent_mark
        
        return board_copy.winner

class ModelPlayer(Player):
    def __init__(self, mark, model_path='model.pt', temperature=1.0):
        super().__init__(mark)
        self.temperature = temperature  # Temperature for controlling exploration/exploitation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained model
        self.model = UltimateTicTacToeModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def _preprocess_board(self, board):
        """Convert board state to the format expected by the model"""
        # Channel 1: board state (9x9)
        board_state = np.zeros((9, 9), dtype=np.float32)
        
        # Channel 2: macroboard (which boards are won, 9x9)
        macro_board = np.zeros((9, 9), dtype=np.float32)
        
        # Channel 3: current player (9x9)
        player_channel = np.ones((9, 9), dtype=np.float32)
        if self.mark == 'O':
            player_channel = -player_channel
        
        # Channel 4: valid moves mask (9x9)
        valid_moves_mask = np.zeros((9, 9), dtype=np.float32)
        
        # Fill board state
        for board_row in range(3):
            for board_col in range(3):
                for row in range(3):
                    for col in range(3):
                        r, c = board_row * 3 + row, board_col * 3 + col
                        cell = board.boards[board_row][board_col].board[row][col]
                        if cell == 'X':
                            board_state[r, c] = 1
                        elif cell == 'O':
                            board_state[r, c] = -1
                
                # Fill macroboard
                winner = board.boards[board_row][board_col].winner
                value = 0
                if winner == 'X':
                    value = 1
                elif winner == 'O':
                    value = -1
                
                for row in range(3):
                    for col in range(3):
                        r, c = board_row * 3 + row, board_col * 3 + col
                        macro_board[r, c] = value
        
        # Fill valid moves mask
        valid_moves = board.get_available_moves()
        for move in valid_moves:
            board_row, board_col, row, col = move
            r, c = board_row * 3 + row, board_col * 3 + col
            valid_moves_mask[r, c] = 1
        
        # Stack all channels
        board_tensor = np.stack([board_state, macro_board, player_channel, valid_moves_mask], axis=0)
        
        # MLP features: move_number, agent_level, game_result
        # Estimate move number based on occupied cells
        occupied_cells = np.count_nonzero(board_state)
        move_number = occupied_cells / 81.0  # Normalize
        
        mlp_features = np.array([move_number, 1.0, 0.0], dtype=np.float32)  # agent_level=1, game_result=0 (ongoing)
        
        # Create mask for valid moves (for the output probabilities)
        mask = np.full(81, -np.inf, dtype=np.float32)
        for move in valid_moves:
            board_row, board_col, row, col = move
            idx = (board_row * 3 + row) * 9 + (board_col * 3 + col)
            mask[idx] = 0.0
            
        return board_tensor, mlp_features, mask
    
    def get_move(self, board):
        """Use the trained model to select a move"""
        moves = board.get_available_moves()
        if not moves:
            return None
        
        # If there's only one valid move, return it immediately
        if len(moves) == 1:
            return moves[0]
        
        # Preprocess the board
        board_tensor, mlp_features, mask = self._preprocess_board(board)
        
        # Convert numpy arrays to PyTorch tensors
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        mlp_features = torch.tensor(mlp_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            logits = self.model(board_tensor, mlp_features).squeeze(0)
            
            # Apply mask to invalid moves
            logits = logits + mask
            
            # Apply temperature
            if self.temperature != 1.0:
                logits = logits / self.temperature
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=0)
            
        # Sample a move based on the probabilities or take the best move
        if self.temperature > 0.01:  # With temperature > 0, sample probabilistically
            move_idx = torch.multinomial(probs, 1).item()
        else:  # With temperature = 0, take the best move
            move_idx = torch.argmax(probs).item()
        
        # Convert flat index to board coordinates
        global_row, global_col = divmod(move_idx, 9)
        board_row, row = divmod(global_row, 3)
        board_col, col = divmod(global_col, 3)
        
        return (board_row, board_col, row, col)
