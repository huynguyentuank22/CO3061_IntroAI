from memory_profiler import profile
import time
import tracemalloc  # For memory tracking
import heapq
from copy import deepcopy
import psutil
import os
import random
import sys
import csv
class Sudoku:
    def __init__(self, diffculty, board, algorithm):
        # with open('testcases/' + testcase + '.txt', 'r') as file:
        self.difficulty = diffculty
        self.board = self.generate_board(diffculty, board=board)
        self.algorithm = algorithm
        self.num_steps = 0  # Count the number of steps
        self.time = 0       # Execution time
        self.memory = 0     # Memory usage
        if self.algorithm == 'A*':
            self.algorithm = 'A_star'
        self.output_file = 'output/output_' + diffculty + "__" + self.algorithm # File to save output
        self.solving_steps = []  # Store solving steps for visualization
        self.back_track = 0
    def generate_board(self, difficulty, board):
        """Generate a random Sudoku puzzle based on difficulty."""
        base_board = [row[:] for row in board]

        # Remove more numbers for harder puzzles
        difficulty_map = {"Easy": 25, "Medium": 40, "Hard": 55, "Expert": 64}
        num_remove = difficulty_map[difficulty]
        random.seed(81)
        for _ in range(num_remove):
            row, col = random.randint(0, 8), random.randint(0, 8)
            while base_board[row][col] == 0:
                row, col = random.randint(0, 8), random.randint(0, 8)
            base_board[row][col] = 0

        return base_board
    def write_to_file(self, message):
        """Write a message to the output file."""
        with open(self.output_file, "a") as file:
            file.write(message + "\n")

    def print_board(self, board=None):
        """Write the Sudoku board to the output file."""
        if board is None:
            board = self.board
        with open(self.output_file, "a") as file:
            for i in range(9):
                if i % 3 == 0 and i != 0:
                    file.write("-" * 21 + "\n")
                for j in range(9):
                    if j % 3 == 0 and j != 0:
                        file.write("| ")
                    file.write(f"{board[i][j] if board[i][j] != 0 else '.'} ")
                file.write("\n")
            file.write("\n")

    def find_empty_cell(self):
        """Find an empty cell (represented by 0)."""
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == 0:
                    return row, col
        return None

    def is_valid(self, num, pos):
        """
        Check if 'num' is valid at position 'pos' (row, col).
        :param num: Number to check
        :param pos: Tuple (row, col)
        """
        row, col = pos

        # Check row
        if num in self.board[row]:
            return False

        # Check column
        if num in [self.board[i][col] for i in range(9)]:
            return False

        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if self.board[i][j] == num:
                    return False

        return True
    def solve_by_DFS(self):
        """Solve the Sudoku puzzle using Backtracking."""
        empty_cell = self.find_empty_cell()
        if not empty_cell:
            return True
        
        row, col = empty_cell

        for num in range(1, 10):
            if self.is_valid(num, (row, col)):
                self.board[row][col] = num
                self.num_steps += 1
                
                # Log the step and current board state
                self.write_to_file(f"Step {self.num_steps}: Placed {num} at ({row}, {col})")
                self.print_board()

                if self.solve_by_DFS():
                    return True

                # Backtrack if the current number doesn't lead to a solution
                self.board[row][col] = 0
                self.back_track += 1

        return False

    def solve_by_A_star(self):
        """Solve the Sudoku puzzle using A* search."""
        # Create initial state tuple (board, empty_cells)
        initial_board = [row[:] for row in self.board]
        empty_cells = [(i, j) for i in range(9) for j in range(9) if self.board[i][j] == 0]
        
        # Priority queue for A* search: (f_score, step_count, board, empty_cells)
        open_set = []
        heapq.heappush(open_set, (len(empty_cells), 0, self.board, empty_cells))
        # Track visited states using string representation of boards
        visited = set()
        
        while open_set:
            f_score, steps, current_board, remaining_cells = heapq.heappop(open_set)
            board_tuple = tuple(tuple(row) for row in current_board)
            
            if board_tuple in visited:
                continue
                
            visited.add(board_tuple)
            self.num_steps = steps
            
            # If no empty cells remain, we've found a solution
            if not remaining_cells:
                self.board = current_board
                return True
                
            # Get the cell with the fewest possible valid numbers (MRV heuristic)
            min_options = 10
            best_cell = None
            best_numbers = []
            
            for cell in remaining_cells:
                row, col = cell
                valid_numbers = [num for num in range(1, 10) 
                            if self.is_valid_for_board(num, (row, col), current_board)]
                if len(valid_numbers) < min_options:
                    min_options = len(valid_numbers)
                    best_cell = cell
                    best_numbers = valid_numbers
                    
            if not best_numbers:  # If no valid numbers for the best cell, this branch is dead
                self.back_track += 1
                continue
                
            row, col = best_cell
            new_remaining = [cell for cell in remaining_cells if cell != best_cell]
            
            # Try each valid number for the selected cell
            for num in best_numbers:
                # Create new board state
                new_board = [row[:] for row in current_board]
                new_board[row][col] = num
                
                # Calculate f_score = g_score + h_score
                # g_score is the depth (steps taken)
                # h_score is the remaining empty cells
                g_score = steps + 1
                h_score = len(new_remaining)
                f_score = g_score + h_score
                
                # Add new state to open set
                heapq.heappush(open_set, (f_score, g_score, new_board, new_remaining))
                
                # Only log when we actually place a number
                self.write_to_file(f"Step {g_score}: Placed {num} at ({row}, {col})")
                self.print_board(new_board)
        
        return False
    def is_valid_for_board(self, num, pos, board):
        """
        Check if 'num' is valid at position 'pos' (row, col) for the given board.
        :param num: Number to check
        :param pos: Tuple (row, col)
        :param board: The board state to check against
        """
        row, col = pos
        
        # Check row
        if num in board[row]:
            return False
            
        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False
            
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == num:
                    return False
                    
        return True

    def heuristic(self, board):
        """Heuristic for A* search: count the number of empty cells."""
        return sum(row.count(0) for row in board)
    def log_to_csv(self, result):
        """Log solving results into a CSV file."""
        file_name = "sudoku_results.csv"
        file_exists = False
        try:
            with open(file_name, "r") as f:
                file_exists = True
        except FileNotFoundError:
            pass

        with open(file_name, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if not file_exists:
                writer.writeheader()  # Write header if new file
            writer.writerow(result)
            
    def solver(self, algorithm='DFS'):
        """Solve the Sudoku puzzle using the specified algorithm."""
        # Reset step counter
        self.num_steps = 0
        
        # Write initial board state to file
        self.write_to_file("Initial Sudoku board:")
        self.print_board()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Solve using specified algorithm
        start_time = time.time()
        if algorithm == 'DFS':
            solved = self.solve_by_DFS()
        else:  # A*
            solved = self.solve_by_A_star()
        end_time = time.time()
        
        # Stop memory tracking and calculate peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result = {
            "Algorithm": algorithm,
            "Difficulty": self.difficulty,
            "Solved": solved,
            "Steps": self.num_steps,
            "Time (s)": round(end_time - start_time, 4),
            "Memory (KB)": round(peak / 1024, 2),
            "Backtracks": self.back_track
        }
        self.log_to_csv(result)
        if solved:
            self.write_to_file(f"\nSolved Sudoku board ({algorithm}):")
            self.print_board()
            self.write_to_file(f"\nNumber of steps: {self.num_steps}")
            self.write_to_file(f"Execution time: {end_time - start_time:.4f} seconds")
            self.write_to_file(f"Peak memory usage: {peak / 1024:.2f} KB\n")
            self.write_to_file(f"Number of backtracks: {self.back_track}")
        
        return solved

if __name__ == '__main__':
    tc = [
      [3, 4, 8, 7, 9, 6, 5, 1, 2],
    [2, 6, 9, 1, 3, 5, 4, 7, 8],
    [1, 5, 7, 8, 2, 4, 3, 9, 6],
    [5, 7, 1, 4, 6, 2, 9, 8, 3],
    [4, 9, 6, 3, 1, 8, 7, 2, 5],
    [8, 2, 3, 5, 7, 9, 1, 6, 4],
    [7, 1, 5, 6, 8, 3, 2, 4, 9],
    [9, 8, 4, 2, 5, 1, 6, 3, 7],
    [6, 3, 2, 9, 4, 7, 8, 5, 1]
]
    difficulties = ["Easy", "Medium", "Hard", "Expert"]
    for dif in difficulties:
        sudoku = Sudoku(board = tc, diffculty=dif, algorithm="A*")
        sudoku.solver("A*")
    