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
    def __init__(self, mark, model_path='model.pt', temperature=1.0, use_mcts=False, 
                 simulation_time=1.0, num_simulations=800, c_puct=1.0, top_k_moves=5):
        super().__init__(mark)
        self.temperature = temperature  # Temperature for controlling exploration/exploitation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opponent_mark = 'O' if mark == 'X' else 'X'
        
        # MCTS parameters
        self.use_mcts = use_mcts  # Whether to use MCTS or just the raw model
        self.simulation_time = simulation_time  # Maximum search time in seconds
        self.num_simulations = num_simulations  # Maximum number of simulations
        self.c_puct = c_puct  # Exploration constant
        self.top_k_moves = top_k_moves  # Number of top moves to consider from policy network
        
        # Load the trained model
        self.model = UltimateTicTacToeModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        mode = "MCTS-enhanced" if use_mcts else "pure"
        print(f"Model Player ({mode}) loaded on {self.device}")
    
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
    
    def _get_model_predictions(self, board):
        """Get policy and value predictions from the model"""
        # Preprocess the board
        board_tensor, mlp_features, mask = self._preprocess_board(board)
        
        # Convert numpy arrays to PyTorch tensors
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        mlp_features = torch.tensor(mlp_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            output_moves, output_result = self.model(board_tensor, mlp_features)
            # Apply mask to invalid moves
            masked_logits = output_moves + mask
            
            # Apply temperature
            if self.temperature != 1.0:
                masked_logits /= self.temperature
                
            # Convert to probabilities
            move_probs = torch.softmax(masked_logits, dim=1).squeeze(0).cpu().numpy()
            
            # Get win probability
            result_probs = torch.softmax(output_result, dim=1).squeeze(0).cpu().numpy()
            value = result_probs[1]  # Probability of winning
            
        return move_probs, value, mask
    
    def get_move(self, board):
        """Get the best move using either the raw model or MCTS"""
        moves = board.get_available_moves()
        if not moves:
            return None
        
        # If there's only one valid move, return it immediately
        if len(moves) == 1:
            return moves[0]
            
        if not self.use_mcts:
            # Use the raw model for move selection (original behavior)
            return self._get_model_move(board)
        else:
            # Use MCTS with model guidance
            return self._get_mcts_move(board)
    
    def _get_model_move(self, board):
        """Use the raw model to select a move (original behavior)"""
        # Preprocess the board
        board_tensor, mlp_features, mask = self._preprocess_board(board)
        
        # Convert numpy arrays to PyTorch tensors
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        mlp_features = torch.tensor(mlp_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            output_moves, output_result = self.model(board_tensor, mlp_features)
            # Apply mask to invalid moves
            masked_logits = output_moves + mask
            
            # Apply temperature
            if self.temperature != 1.0:
                masked_logits /= self.temperature
            # Convert to probabilities
            probs = torch.softmax(masked_logits, dim=1)
            
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
    def _stimulate(self, board):
        board_copy = deepcopy(board)
        current_mark = self.opponent_mark  # We just made a move, so it's opponent's turn
        while board_copy.winner is None and not board_copy.is_draw:
            moves = board_copy.get_available_moves()
            if not moves:
                break
            # Make a random move
            board_row, board_col, row, col = random.choice(moves)
            board_copy.make_move(board_row, board_col, row, col)
            current_mark = self.mark if current_mark == self.opponent_mark else self.opponent_mark
        return board_copy.winner
    
    def _get_mcts_move(self, board):
        """Use MCTS with model guidance to find the best move"""
        moves = board.get_available_moves()
        if not moves:
            return None
        
        if len(moves) == 1:
            return moves[0]
        
        # Get policy and value predictions from the model
        move_probs, _, _ = self._get_model_predictions(board)
        
        # Create mapping for moves and initialize statistics
        move_stats = {}
        for move in moves:
            board_row, board_col, row, col = move
            idx = (board_row * 3 + row) * 9 + (board_col * 3 + col)
            prior_prob = float(move_probs[idx])
            move_stats[move] = {
                'visits': 0,
                'wins': 0,
                'prior': prior_prob
            }
        
        # Option to filter to only top-k moves based on prior probability
        if self.top_k_moves > 0 and len(moves) > self.top_k_moves:
            # Sort moves by prior probability
            sorted_moves = sorted(moves, key=lambda m: move_stats[m]['prior'], reverse=True)
            # Keep only top-k
            top_moves = sorted_moves[:self.top_k_moves]
            # Filter move_stats to only include top moves
            move_stats = {m: move_stats[m] for m in top_moves}
        
        # Run MCTS simulations
        total_simulations = 0
        end_time = time.time() + self.simulation_time
        
        while total_simulations < self.num_simulations and time.time() < end_time:
            # Selection - use UCB formula to select a promising move
            selected_move = self._select_move(moves, move_stats)
            
            # Expansion & Simulation - play out from selected move
            board_copy = deepcopy(board)
            board_row, board_col, row, col = selected_move
            board_copy.make_move(board_row, board_col, row, col)
            
            # Use model to evaluate position or run a random simulation
            result = self._simulate_game(board_copy)
            
            # Backpropagation - update statistics
            move_stats[selected_move]['visits'] += 1
            if result == self.mark:  # Win for us
                move_stats[selected_move]['wins'] += 1
            
            total_simulations += 1
        
        # Choose best move based on visit count (most robust policy)
        best_move = None
        most_visits = -1
        
        for move, stats in move_stats.items():
            if stats['visits'] > most_visits:
                most_visits = stats['visits']
                best_move = move
                
        return best_move
    
    def _select_move(self, moves, move_stats):
        """Select a move using UCB formula"""
        # Calculate total visits
        total_visits = sum(stats['visits'] for stats in move_stats.values())
        
        # If some moves have not been visited, prioritize them
        unexplored = [move for move in move_stats if move_stats[move]['visits'] == 0]
        if unexplored:
            # Select based on prior probability for unexplored moves
            return max(unexplored, key=lambda m: move_stats[m]['prior'])
        
        best_score = -float('inf')
        best_move = None
        
        # Calculate UCB score for each move
        for move, stats in move_stats.items():
            # Exploitation term: win ratio
            if stats['visits'] > 0:
                win_rate = stats['wins'] / stats['visits']
            else:
                win_rate = 0
                
            # Exploration term: UCB1 formula with prior probability
            if total_visits > 0:
                exploration = self.c_puct * stats['prior'] * math.sqrt(math.log(total_visits) / (1 + stats['visits']))
            else:
                exploration = self.c_puct * stats['prior']
                
            ucb_score = win_rate + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                
        return best_move
    
    def _simulate_game(self, board):
        """Simulate a game from the current position to determine outcome"""
        # Option 1: Use fast random rollout (more simulations, less accuracy)
        if random.random() < 0.7:  # 70% of the time use random rollout for speed
            return self._simulate_random(board)
        
        # Option 2: Use model evaluation (slower, more accurate)
        else:
            # Get the model's evaluation of the position
            _, value, _ = self._get_model_predictions(board)
            
            # If model predicts > 60% win probability, count as win
            if value > 0.6 and self.mark == board.current_player:
                return self.mark
            elif value < 0.4 and self.mark != board.current_player:
                return self.mark
            else:
                return None  # No clear winner - results in partial reward
    
    def _simulate_random(self, board):
        """Simulate a random game from the current position"""
        board_copy = deepcopy(board)
        current_player = board_copy.current_player
        
        # Play random moves until game over
        while board_copy.winner is None and not board_copy.is_draw:
            moves = board_copy.get_available_moves()
            if not moves:
                break
                
            # Make a random move
            board_row, board_col, row, col = random.choice(moves)
            board_copy.make_move(board_row, board_col, row, col)
            
        return board_copy.winner


