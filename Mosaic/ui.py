import os
import sys
import pygame
from mosaic_solver import Mosaic

class Solution:
    def __init__(self, board, output_dir, step_delay=1000):
        self.board = board
        self.output_dir = output_dir
        self.step_delay = step_delay  # Delay between steps in milliseconds
        self.solution_steps = []
        self.current_step = 0
        self.total_steps = 0
        self.solving = False
        self.algorithm = None
        self.current_solution = None
        self.last_step_time = 0
        self.mosaic = Mosaic("testcases/tc4.txt")  # Load the test case

        # Pygame setup
        pygame.init()
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.screen = pygame.display.set_mode((400, 450))  # Increased height for title
        pygame.display.set_caption("Mosaic Puzzle Solver")

        # Buttons for interaction
        self.buttons = [
            {"rect": pygame.Rect(10, 400, 80, 30), "text": "DFS", "action": "solve_dfs", "color": self.BLUE},
            {"rect": pygame.Rect(100, 400, 80, 30), "text": "BFS", "action": "solve_bfs", "color": self.BLUE},
            {"rect": pygame.Rect(190, 400, 80, 30), "text": "A*", "action": "solve_a*", "color": self.BLUE},
            {"rect": pygame.Rect(280, 400, 80, 30), "text": "Reset", "action": "reset", "color": self.BLUE},
        ]

        # Load initial board from the test case
        self.load_initial_board()

    def load_initial_board(self):
        # Convert the test case board into a format suitable for visualization
        initial_board = []
        for row in self.board:
            board_row = []
            for cell in row:
                if cell == 1:
                    board_row.append(1)
                else:
                    board_row.append(0)
            initial_board.append(board_row)
        self.solution_steps.append({"description": "Initial Board", "board": initial_board})
        self.current_solution = self.solution_steps[0]
        self.total_steps = 1

    def load_steps_from_log(self, log_file_path):
        with open(log_file_path, 'r') as file:
            lines = file.readlines()

        step = None
        board = []
        for line in lines:
            line = line.strip()
            if line.startswith("Initial board:") or line.startswith("Step"):
                if step is not None:
                    self.solution_steps.append(step)
                step = {"description": line, "board": []}
            elif line:
                board_row = [int(cell) if cell.isdigit() else 0 for cell in line.split()]
                step["board"].append(board_row)

        if step is not None:
            self.solution_steps.append(step)

    def solve_puzzle(self, algorithm):
        self.algorithm = algorithm
        self.solving = True
        self.current_step = 0

        # # Clear previous solution file
        # filename = f"{self.output_dir}/solution_{algorithm}.txt"
        # if os.path.exists(filename):
        #     os.remove(filename)

        # Get solution based on algorithm
        if algorithm == 'solve_dfs':
            self.mosaic.solve_by_DFS()
        elif algorithm == 'solve_bfs':
            self.mosaic.solve_by_BFS()
        elif algorithm == 'solve_a*':
            self.mosaic.solve_by_A_star()

        # Load solution steps from the solver's log file
        if algorithm == "solve_a*":
            temp = "a_star"
        else:
            temp = algorithm[6:]
        self.load_steps_from_log("output/" + "output_" + temp + ".txt")

        self.total_steps = len(self.solution_steps)
        if self.total_steps > 0:
            self.current_solution = self.solution_steps[0]

    def update(self):
        current_time = pygame.time.get_ticks()
        if self.solving and self.current_step < self.total_steps - 1:
            if current_time - self.last_step_time >= self.step_delay:
                self.current_step += 1
                self.current_solution = self.solution_steps[self.current_step]
                self.last_step_time = current_time

    def reset(self):
        self.current_solution = None
        self.solution_steps = []
        self.current_step = 0
        self.total_steps = 0
        self.solving = False
        self.algorithm = None
        self.load_initial_board()

    def draw_board(self):
        if self.current_solution:
            cell_size = 40
            for row in range(len(self.current_solution["board"])):
                for col in range(len(self.current_solution["board"][row])):
                    cell_value = self.current_solution["board"][row][col]
                    color = self.GREEN if cell_value == 1 else self.WHITE
                    pygame.draw.rect(
                        self.screen,
                        color,
                        (col * cell_size, row * cell_size + 50, cell_size, cell_size),  # Offset for title
                    )
                    pygame.draw.rect(
                        self.screen,
                        self.BLACK,
                        (col * cell_size, row * cell_size + 50, cell_size, cell_size),
                        1,
                    )

    def draw_buttons(self):
        font = pygame.font.Font(None, 24)
        for button in self.buttons:
            pygame.draw.rect(self.screen, button["color"], button["rect"])
            text = font.render(button["text"], True, self.WHITE)
            text_rect = text.get_rect(center=button["rect"].center)
            self.screen.blit(text, text_rect)

    def draw_steps(self):
        font = pygame.font.Font(None, 24)
        step_text = f"Step: {self.current_step + 1}/{self.total_steps}"
        text = font.render(step_text, True, self.BLACK)
        self.screen.blit(text, (10, 370))

    def draw_title(self):
        font = pygame.font.Font(None, 36)
        title_text = "Mosaic Puzzle Solver"
        text = font.render(title_text, True, self.BLACK)
        self.screen.blit(text, (50, 10))

    def run(self):
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    for button in self.buttons:
                        if button["rect"].collidepoint(mouse_pos):
                            if button["action"] == 'reset':
                                self.reset()
                            else:
                                self.solve_puzzle(button["action"])

            # Update solution state
            self.update()

            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_title()
            self.draw_board()
            self.draw_buttons()
            self.draw_steps()
            pygame.display.flip()

            # Control frame rate
            clock.tick(60)

# Example usage
if __name__ == "__main__":
    # Initialize the board and output directory
    solver = Mosaic("testcases/tc4.txt")
    board = solver.board
    output_dir = "output"
    solution = Solution(board, output_dir)
    solution.run()