import csv
import os
import numpy as np

class TrainingDataLogger:
    """Logger for collecting neural network training data from games"""
    
    def __init__(self, log_file='train.csv'):
        """Initialize the training data logger"""
        self.log_file = log_file
        self.ensure_log_file_exists()
        self.buffer = []  # Buffer to hold moves before writing to file
        self.buffer_size = 1000  # Write to disk every 1000 moves
        
    def ensure_log_file_exists(self):
        """Create the training data file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [
                    'board_state',
                    'macroboard',
                    'current_player',
                    'valid_moves',
                    'chosen_move',
                    'game_result',
                    'agent_level',
                    'move_number',
                    'final_result'
                ]
                writer.writerow(headers)
    
    def log_move(self, board, chosen_move, agent_level, move_number):
        """Log a single move to the buffer"""
        # Convert the UltimateBoard to the required format
        board_state = self._get_board_state(board)
        macroboard = self._get_macroboard(board)
        current_player = 1 if board.current_player == 'X' else -1
        valid_moves = self._get_valid_moves(board)
        
        # Game is not finished yet, so we'll set these temporarily
        game_result = 0
        final_result = 0
        
        # Store in buffer
        move_data = {
            'board_state': board_state,
            'macroboard': macroboard,
            'current_player': current_player,
            'valid_moves': valid_moves,
            'chosen_move': chosen_move,
            'game_result': game_result,
            'agent_level': agent_level,
            'move_number': move_number,
            'final_result': final_result,
            'is_complete': False  # Flag to indicate if we've updated with final result
        }
        
        self.buffer.append(move_data)
        
        # If buffer is full, write to disk
        if len(self.buffer) >= self.buffer_size:
            self._write_complete_moves()
    
    def update_game_results(self, winner):
        """Update buffered moves with game results once game is completed"""
        game_result = 0  # Draw
        if winner == 'X':
            game_result = 1
        elif winner == 'O':
            game_result = -1
            
        # Update all moves in buffer that aren't complete
        for move in self.buffer:
            if not move['is_complete']:
                move['game_result'] = game_result
                
                # Set final_result based on whether the player who made this move won
                if (move['current_player'] == 1 and winner == 'X') or \
                   (move['current_player'] == -1 and winner == 'O'):
                    move['final_result'] = 1
                else:
                    move['final_result'] = 0
                    
                move['is_complete'] = True
                
        # Write completed moves to disk
        self._write_complete_moves()
    
    def _write_complete_moves(self):
        """Write all completed moves from buffer to disk"""
        complete_moves = [m for m in self.buffer if m['is_complete']]
        
        if complete_moves:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for move in complete_moves:
                    writer.writerow([
                        move['board_state'],
                        move['macroboard'],
                        move['current_player'],
                        move['valid_moves'],
                        move['chosen_move'],
                        move['game_result'],
                        move['agent_level'],
                        move['move_number'],
                        move['final_result']
                    ])
            
            # Remove written moves from buffer
            self.buffer = [m for m in self.buffer if not m['is_complete']]
    
    def _get_board_state(self, board):
        """Convert the game board to a flattened 81-dim vector
        X = 1, O = -1, Empty = 0
        """
        board_vector = []
        for i in range(3):
            for j in range(3):
                small_board = board.boards[i][j]
                for row in range(3):
                    for col in range(3):
                        cell = small_board.board[row][col]
                        if cell == 'X':
                            board_vector.append(1)
                        elif cell == 'O':
                            board_vector.append(-1)
                        else:
                            board_vector.append(0)
        
        return ','.join(map(str, board_vector))
    
    def _get_macroboard(self, board):
        """Convert the macroboard (small board winners) to a 9-dim vector
        X = 1, O = -1, Draw = 0, Still playing = 0
        """
        macroboard = []
        for i in range(3):
            for j in range(3):
                small_board = board.boards[i][j]
                if small_board.winner == 'X':
                    macroboard.append(1)
                elif small_board.winner == 'O':
                    macroboard.append(-1)
                else:
                    macroboard.append(0)  # Either draw or still playing
        
        return ','.join(map(str, macroboard))
    
    def _get_valid_moves(self, board):
        """Get valid moves as a comma-separated list of indices
        Each move is encoded as a number from 0-80
        """
        valid_moves = []
        for board_row, board_col, row, col in board.get_available_moves():
            # Convert to a single index from 0-80
            index = (board_row * 3 + board_col) * 9 + (row * 3 + col)
            valid_moves.append(index)
            
        return ','.join(map(str, valid_moves))
    
    def flush(self):
        """Flush all remaining data to disk"""
        self._write_complete_moves()
