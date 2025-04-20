import csv
import os
import time
import datetime

class BattleLogger:
    """Class for logging the results of agent battles"""
    
    def __init__(self, log_file='battle_logs.csv'):
        """Initialize the logger with the specified log file"""
        self.log_file = log_file
        self.ensure_log_file_exists()
        
    def ensure_log_file_exists(self):
        """Create the log file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [
                    'Date', 
                    'Time',
                    'Agent_X_Type', 
                    'Agent_O_Type',
                    'Winner', 
                    'Num_Moves', 
                    'Game_Duration_Seconds',
                    'Final_Board_State'
                ]
                writer.writerow(headers)
                
    def log_battle(self, agent_x_type, agent_o_type, winner, num_moves, duration, board_state):
        """Log a battle result to the CSV file"""
        now = datetime.datetime.now()
        date = now.strftime('%Y-%m-%d')
        current_time = now.strftime('%H:%M:%S')
        
        # Convert board state to a string representation
        board_str = self._board_to_string(board_state)
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                date,
                current_time,
                agent_x_type,
                agent_o_type,
                winner if winner else 'Draw',
                num_moves,
                round(duration, 2),
                board_str
            ]
            writer.writerow(row)
            
    def _board_to_string(self, board):
        """Convert the ultimate board state to a string for logging"""
        board_str = ""
        for i in range(3):
            for j in range(3):
                small_board = board.boards[i][j]
                for row in range(3):
                    for col in range(3):
                        cell = small_board.board[row][col]
                        board_str += cell if cell is not None else " "
        return board_str
        
    def get_battle_data(self):
        """Read and return all battle data from the log file"""
        battles = []
        with open(self.log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                battles.append(row)
        return battles
