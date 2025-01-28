import time
import psutil
from copy import deepcopy

class Mosaic:
    def __init__(self, input_file):
        """Initialize the Mosaic class with the puzzle from an input file."""
        self.board = []
        with open(input_file, 'r') as file:
            for line in file:
                self.board.append([int(cell) if cell.isdigit() else None for cell in line.strip().split()])
        self.steps = 0

    def is_valid(self, board):
        """Check if the current board configuration is valid."""
        for r in range(len(board)):
            for c in range(len(board[0])):
                if self.board[r][c] is not None:  # It's a number cell
                    count_black = sum(
                        1
                        for nr in range(max(0, r - 1), min(len(board), r + 2))
                        for nc in range(max(0, c - 1), min(len(board[0]), c + 2))
                        if 0 <= nr < len(board) and 0 <= nc < len(board[0]) and board[nr][nc] == 1
                    )
                    if count_black != self.board[r][c]:
                        return False
        return True

    def print_board(self, board):
        """Print the current state of the board with 0 and 1 for marked/unmarked cells."""
        for row in board:
            print(" ".join([str(cell) if cell is not None else '.' for cell in row]))
        print("\n")

    def solve_by_DFS(self):
        """Solve the puzzle using a Depth-First Search (DFS) algorithm."""
        start_time = time.time()
        process = psutil.Process()

        def dfs(board, row, col):
            self.steps += 1
            if row == len(board):
                return board if self.is_valid(board) else None

            next_row, next_col = (row, col + 1) if col + 1 < len(board[0]) else (row + 1, 0)

            if self.board[row][col] is not None:  # Skip number cells
                return dfs(board, next_row, next_col)

            # Try placing a black cell (value=1)
            board[row][col] = 1
            if self.is_valid(board):
                result = dfs(deepcopy(board), next_row, next_col)
                if result:
                    return result

            # Backtrack and try no black cell (value=0)
            board[row][col] = 0
            if self.is_valid(board):
                result = dfs(deepcopy(board), next_row, next_col)
                if result:
                    return result

            # Undo placement (reset)
            board[row][col] = None
            return None

        initial_board = [[None if cell is None else 0 for cell in row] for row in self.board]
        solved_board = dfs(initial_board, 0, 0)

        elapsed_time = time.time() - start_time
        memory_usage = process.memory_info().rss / 1024 / 1024  # Memory usage in MB

        print("Steps:", self.steps)
        print("Time elapsed:", elapsed_time, "seconds")
        print("Memory usage:", memory_usage, "MB")

        if solved_board:
            print("Solved Board:")
            self.print_board(solved_board)
        else:
            print("No solution found.")

# Example Usage
# Save a text file "mosaic_input.txt" with the puzzle.
# mosaic = Mosaic("mosaic_input.txt")
# mosaic.solve_by_DFS()



# Example Usage
# Save a text file "mosaic_input.txt" with the puzzle.
mosaic = Mosaic("testcases/tc2.txt")
mosaic.solve_by_DFS()
