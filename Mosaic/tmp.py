def solve_by_A_star(self, output_file: str) -> bool:
        """Solve using A*."""
        def calculate_heuristic(state: List[List[int]]) -> int:
            h_value = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid[r][c] is not None:
                        count = self.count_blacks_around(r, c, state)
                        h_value += abs(count - self.grid[r][c])
            return h_value

        def is_solution_valid(state: List[List[int]]) -> bool:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid[r][c] is not None:
                        if self.count_blacks_around(r, c, state) != self.grid[r][c]:
                            return False
            return True

        # Priority queue: (f_score, g_score, state)
        start_state = [row[:] for row in self.solution]
        pq = [(calculate_heuristic(start_state), 0, start_state)]
        visited = set()

        while pq:
            f, g, current_state = heapq.heappop(pq)
            state_hash = tuple(tuple(row) for row in current_state)

            if state_hash in visited:
                continue

            self.solution = current_state
            self.steps += 1
            self.log_step(output_file)

            if is_solution_valid(current_state):
                return True

            visited.add(state_hash)

            # Generate next states
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid[r][c] is None and current_state[r][c] == 0:
                        for value in [1, 0]:
                            new_state = [row[:] for row in current_state]
                            new_state[r][c] = value
                            if self.is_valid_state(new_state):
                                heapq.heappush(pq, (g + 1 + calculate_heuristic(new_state), g + 1, new_state))

        return False

import heapq
import time
import tracemalloc
from typing import List, Tuple

class Mosaic:
    def __init__(self, input_file: str):
        self.grid = []
        with open(input_file, 'r') as f:
            for line in f:
                self.grid.append([int(x) if x.isdigit() else None for x in line.strip().split()])
        
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.grid else 0
        self.solution = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.steps = 0

    def count_blacks_around(self, r: int, c: int, state: List[List[int]]) -> int:
        """Count black cells around position (r, c)."""
        return sum(
            1 for dr in [-1, 0, 1]
            for dc in [-1, 0, 1]
            if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols and state[r + dr][c + dc] == 1
        )

    def is_valid_state(self, state: List[List[int]]) -> bool:
        """Check if the state is valid."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] is not None:
                    blacks = self.count_blacks_around(r, c, state)
                    remaining = sum(
                        1 for dr in [-1, 0, 1]
                        for dc in [-1, 0, 1]
                        if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols and state[r + dr][c + dc] == 0
                    )
                    if blacks > self.grid[r][c] or blacks + remaining < self.grid[r][c]:
                        return False
        return True

    def solve_by_DFS(self, output_file: str) -> bool:
        """Solve using DFS with heuristics."""
        # Collect all variable cells
        cells = [
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if self.grid[r][c] is None
        ]

        # Sort cells by constraints (e.g., neighbors with known values)
        cells.sort(key=lambda cell: sum(
            1 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
            if 0 <= cell[0] + dr < self.rows and 0 <= cell[1] + dc < self.cols and self.grid[cell[0] + dr][cell[1] + dc] is not None
        ), reverse=True)

        def dfs(idx: int) -> bool:
            # Base case: All cells are processed
            if idx == len(cells):
                return all(
                    self.count_blacks_around(r, c, self.solution) == self.grid[r][c]
                    for r in range(self.rows)
                    for c in range(self.cols)
                    if self.grid[r][c] is not None
                )

            r, c = cells[idx]

            # Try black (1)
            self.solution[r][c] = 1
            self.steps += 1
            self.log_step(output_file)
            if self.is_valid_state(self.solution) and dfs(idx + 1):
                return True

            # Try empty (0)
            self.solution[r][c] = 0
            self.steps += 1
            self.log_step(output_file)
            if self.is_valid_state(self.solution) and dfs(idx + 1):
                return True

            # Backtrack
            self.solution[r][c] = 0
            return False

        return dfs(0)




    def print_board(self):
        """Print the current state of the board."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] is not None:
                    print(f" {self.grid[r][c]}", end='')
                else:
                    print(f" {self.solution[r][c]}", end='')
            print()
        print()

    def log_step(self, output_file: str):
        """Log the current state of the solution to the output file."""
        with open(output_file, 'a') as f:
            f.write(f"Step {self.steps}:\n")
            for row in self.solution:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('\n')

    def print_solution(self):
        """Print the solved Mosaic grid."""
        for row in self.solution:
            print(' '.join(map(str, row)))

    def performance_metrics(self, start_time: float, memory_start: Tuple[int, int]):
        """Print performance metrics: steps, execution time, memory usage."""
        end_time = time.time()
        memory_end = tracemalloc.get_traced_memory()

        print(f"Number of steps: {self.steps}")
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        print(f"Memory usage: {(memory_end[1] - memory_start[1]) / 1024:.2f} KB")

if __name__ == "__main__":
    mosaic = Mosaic("testcases/tc2.txt")
    tracemalloc.start()
    start_time = time.time()
    memory_start = tracemalloc.get_traced_memory()

    output_file = "mosaic_output.txt"
    open(output_file, 'w').close()  # Clear the file before writing

    if mosaic.solve_by_DFS(output_file):
        print("Solved by DFS:")
        mosaic.print_solution()
    # if mosaic.solve_by_A_star(output_file):
    #     print("Solved by A*:")
    #     mosaic.print_solution()
    else:
        print("No solution found.")

    mosaic.performance_metrics(start_time, memory_start)
