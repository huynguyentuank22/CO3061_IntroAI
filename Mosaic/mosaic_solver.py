import time
import psutil
from copy import deepcopy
from collections import deque
import heapq
import os

class Mosaic:
    def __init__(self, input_file):
        """Initialize the Mosaic class with the puzzle from an input file."""
        self.board = []
        with open(input_file, 'r') as file:
            for line in file:
                self.board.append([int(cell) if cell.isdigit() else None for cell in line.strip().split()])
        self.steps = 0
        # Create output directory if it doesn't exist
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def is_solved(self, board, state):
        """Check if the current board configuration is solved."""
        for r in range(len(board)):
            for c in range(len(board[0])):
                if self.board[r][c] is not None:  # It's a number cell
                    count_black = sum(
                        1
                        for nr in range(max(0, r - 1), min(len(board), r + 2))
                        for nc in range(max(0, c - 1), min(len(board[0]), c + 2))
                        if state[nr][nc] == 1
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

        initial_board = [[None if cell is None else 0 for cell in row] for row in self.board]

        solved_board = self.__dfs(initial_board)

        elapsed_time = time.time() - start_time
        memory_usage = process.memory_info().rss / 1024 / 1024  # Memory usage in MB

        print("Steps:", self.steps)
        print("Time elapsed:", elapsed_time, "seconds")
        print("Memory usage:", memory_usage, "MB")

        if solved_board:
            print("Solved Board:")
            self.print_board(solved_board)
            print(f"Steps: {self.steps}")
        else:
            print("No solution found.")

    def solve_by_BFS(self):
        """Solve the puzzle using a Breadth-First Search (BFS) algorithm."""
        start_time = time.time()
        process = psutil.Process()

        initial_board = [[None if cell is None else 0 for cell in row] for row in self.board]

        solved_board = self.__bfs(initial_board)

        elapsed_time = time.time() - start_time
        memory_usage = process.memory_info().rss / 1024 / 1024  # Memory usage in MB

        print("Steps:", self.steps)
        print("Time elapsed:", elapsed_time, "seconds")
        print("Memory usage:", memory_usage, "MB")

        if solved_board:
            print("Solved Board:")
            self.print_board(solved_board)
            print(f"Steps: {self.steps}")
        else:
            print("No solution found.")

    def solve_by_A_star(self):
        """Solve the puzzle using the A* algorithm."""
        start_time = time.time()
        process = psutil.Process()

        initial_board = [[None if cell is None else 0 for cell in row] for row in self.board]

        solved_board = self.__a_star(initial_board)

        elapsed_time = time.time() - start_time
        memory_usage = process.memory_info().rss / 1024 / 1024  # Memory usage in MB

        print("Steps:", self.steps)
        print("Time elapsed:", elapsed_time, "seconds")
        print("Memory usage:", memory_usage, "MB")

        if solved_board:
            print("Solved Board:")
            self.print_board(solved_board)
            print(f"Steps: {self.steps}")
        else:
            print("No solution found.")

    def log_state(self, state, step, algorithm, mode='a', row=None, col=None, value=None):
        """Log the current state to a file."""
        filename = f"{self.output_dir}/output_{algorithm}.txt"
        with open(filename, mode) as f:
            if mode == 'w':
                f.write("Initial board:\n")
                # Write zero matrix for initial state
                for _ in range(len(self.board)):
                    row_str = "0 " * len(self.board[0])
                    f.write(row_str.strip() + "\n")
                f.write("\n")
            else:
                if row is not None and col is not None and value is not None:
                    f.write(f"Step {step}: Placed 1 at ({row}, {col})\n")
            
            # Write binary representation
            reshape = lambda arr, n: [arr[i*n:(i+1)*n] for i in range(n)]
            if isinstance(state, list) and not isinstance(state[0], list):
                state = reshape(state, len(self.board))
            
            for row in range(len(self.board)):
                row_str = ""
                for col in range(len(self.board[0])):
                    row_str += f"{state[row][col]} "
                f.write(row_str.strip() + "\n")
            f.write("\n")

    def write_board_state(self, f, state):
        """Helper method to write board state in the required format."""
        reshape = lambda arr, n: [arr[i*n:(i+1)*n] for i in range(n)]
        if isinstance(state, list) and not isinstance(state[0], list):
            state = reshape(state, len(self.board))
        
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col] is not None:
                    f.write(f" {self.board[row][col]} ")
                elif state[row][col] == 1:
                    f.write(" # ")
                else:
                    f.write(" . ")
                if col % 3 == 2 and col < len(self.board[0]) - 1:
                    f.write("|")
            f.write("\n")
            if row % 3 == 2 and row < len(self.board) - 1:
                f.write("-" * (len(self.board[0]) * 3 + 2) + "\n")

    def __dfs(self, board):
        n = len(board)
        total_cells = n * n
        stack = [([], 0)]
        step_count = 0
        
        # Initialize the log file
        self.log_state(board, 0, "dfs", mode='w')
        
        while stack:
            state, idx = stack.pop()
            row = idx // n
            col = idx % n
            
            if idx == total_cells:
                reshape = lambda arr, n: [arr[i*n:(i+1)*n] for i in range(n)]
                res = reshape(state, n)
                if (self.is_solved(board, res)):
                    return res
            else:
                for value in [1, 0]:
                    self.steps += 1
                    new_state = state + [value]
                    if value == 1:  # Only log when placing a black cell
                        step_count += 1
                        padded_state = new_state + [0] * (total_cells - len(new_state))
                        self.log_state(padded_state, step_count, "dfs", 
                                     row=row, col=col, value="#")
                    stack.append((new_state, idx + 1))
        return None

    def __bfs(self, board):
        n = len(board)
        total_cells = n * n
        queue = deque([([], 0)])
        step_count = 0
        
        # Initialize the log file
        self.log_state(board, 0, "bfs", mode='w')
        
        while queue:
            state, idx = queue.popleft()
            row = idx // n
            col = idx % n
            
            if idx == total_cells:
                reshape = lambda arr, n: [arr[i*n:(i+1)*n] for i in range(n)]
                res = reshape(state, n)
                if self.is_solved(board, res):
                    return res
            else:
                for value in [1, 0]:
                    self.steps += 1
                    new_state = state + [value]
                    if value == 1:  # Only log when placing a black cell
                        step_count += 1
                        padded_state = new_state + [0] * (total_cells - len(new_state))
                        self.log_state(padded_state, step_count, "bfs",
                                     row=row, col=col, value="#")
                    queue.append((new_state, idx + 1))
        return None

    def h(self, board, state):
        h = 0
        s = deepcopy(state)
        s += [0] * (len(board) * len(board[0]) - len(s))
        reshape = lambda arr, n: [arr[i*n:(i+1)*n] for i in range(n)]
        s = reshape(s, len(board))
        for r in range(len(board)):
            for c in range(len(board[0])):
                if self.board[r][c] is not None:  # It's a number cell
                    count_black = sum(
                        1
                        for nr in range(max(0, r - 1), min(len(board), r + 2))
                        for nc in range(max(0, c - 1), min(len(board[0]), c + 2))
                        if s[nr][nc] == 1
                    )
                    h += abs(count_black - self.board[r][c]) 
        return h
    
    def __a_star(self, board):
        n = len(board)
        total_cells = n * n
        pq = []
        initial_state = []
        initial_g = 0
        initial_f = initial_g + self.h(board, initial_state)
        heapq.heappush(pq, (initial_f, initial_g, initial_state, 0))
        step_count = 0
        
        # Initialize the log file
        self.log_state(board, 0, "a_star", mode='w')
        
        while pq:
            f, g, state, idx = heapq.heappop(pq)
            row = idx // n
            col = idx % n
            
            if idx == total_cells:
                reshape = lambda arr, n: [arr[i*n:(i+1)*n] for i in range(n)]
                res = reshape(state, n)
                if self.is_solved(board, res):
                    return res
            else:
                for value in [1, 0]:
                    self.steps += 1
                    new_state = state + [value]
                    if value == 1:  # Only log when placing a black cell
                        step_count += 1
                        padded_state = new_state + [0] * (total_cells - len(new_state))
                        self.log_state(padded_state, step_count, "a_star",
                                     row=row, col=col, value="#")
                    new_idx = idx + 1
                    new_g = g + 1
                    new_f = new_g + self.h(board, new_state)
                    heapq.heappush(pq, (new_f, new_g, new_state, new_idx))
        return None
    

                                                                                                     
# Example Usage
# Save a text file "mosaic_input.txt" with the puzzle.
# mosaic = Mosaic("mosaic_input.txt")
# mosaic.solve_by_DFS()



# Example Usage
# Save a text file "mosaic_input.txt" with the puzzle.
if __name__ == '__main__':
    mosaic = Mosaic("testcases/tc2.txt")
    # mosaic.solve_by_DFS()
    mosaic.solve_by_BFS()
    # mosaic.solve_by_A_star()
