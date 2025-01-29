import time
import tracemalloc
import heapq
from copy import deepcopy
import psutil
import os

class Sudoku:
    def __init__(self, testcase):
        with open('testcases/' + testcase + '.txt', 'r') as file:
            self.board = [[int(num) for num in line.split()] for line in file]
        self.num_steps = 0  # Count the number of steps
        self.time = 0       # Execution time
        self.memory = 0     # Memory usage
        self.output_file = 'output/output_' + testcase  # File to save output
        self.solving_steps = []  # Store solving steps for visualization

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

                # Backtrack
                self.board[row][col] = 0

        return False

    def solve_by_A_star(self):
        """Solve the Sudoku puzzle using A* search."""
        # Create initial state tuple (board, empty_cells)
        initial_board = [row[:] for row in self.board]
        empty_cells = [(i, j) for i in range(9) for j in range(9) if self.board[i][j] == 0]
        
        # Priority queue for A* search: (f_score, step_count, board, empty_cells)
        open_set = [(len(empty_cells), 0, initial_board, empty_cells)]
        # Track visited states using string representation of boards
        visited = set()
        
        while open_set:
            f_score, steps, current_board, remaining_cells = heapq.heappop(open_set)
            board_str = str(current_board)
            
            if board_str in visited:
                continue
                
            visited.add(board_str)
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

    def solver(self, algorithm='DFS'):
        """Solve the Sudoku puzzle using the specified algorithm."""
        # Reset step counter
        self.num_steps = 0
        
        # Write initial board state to file
        self.write_to_file("Initial Sudoku board:")
        self.print_board()
        
        # Solve using specified algorithm
        start_time = time.time()
        if algorithm == 'DFS':
            solved = self.solve_by_DFS()
        else:  # A*
            solved = self.solve_by_A_star()
        end_time = time.time()
        
        # Write solution details
        if solved:
            self.write_to_file(f"\nSolved Sudoku board ({algorithm}):")
            self.print_board()
            self.write_to_file(f"\nNumber of steps: {self.num_steps}")
            self.write_to_file(f"Execution time: {end_time - start_time:.4f} seconds")
            self.write_to_file(f"Memory usage: {self.get_memory_usage():.2f} KB\n")
        
        return solved

    def get_memory_usage(self):
        """Get the current memory usage in KB"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024  # Convert bytes to KB

if __name__ == '__main__':
    sudoku = Sudoku("tc2")
    sudoku.solver('A*')
