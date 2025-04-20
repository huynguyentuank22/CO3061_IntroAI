import pygame
import sys
import os
import time
from mosaic_solver import Mosaic

class MosaicUI:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Mosaic Puzzle Solver")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        
        # Button dimensions
        self.button_width = 120
        self.button_height = 40
        self.button_margin = 20
        
        # Initialize game state
        self.mosaic = Mosaic("testcases/tc4.txt")  # You can change the input file
        self.board = self.mosaic.board
        self.cell_size = min(400 // len(self.board), 400 // len(self.board[0]))
        self.grid_offset_x = (width - len(self.board[0]) * self.cell_size) // 2
        self.grid_offset_y = (height - len(self.board) * self.cell_size) // 2
        
        # Solution state
        self.current_solution = None
        self.solution_steps = []
        self.current_step = 0
        self.total_steps = 0
        self.solving = False
        self.algorithm = None
        self.step_delay = 500  # Delay between steps in milliseconds
        self.last_step_time = 0
        
        # Initialize buttons
        self.buttons = self.create_buttons()
        
        # Font
        self.font = pygame.font.Font(None, 36)
        
        # Create output directory if it doesn't exist
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def create_buttons(self):
        buttons = []
        algorithms = ["Solve DFS", "Solve BFS", "Solve A*", "Reset"]
        total_width = len(algorithms) * (self.button_width + self.button_margin) - self.button_margin
        start_x = (self.width - total_width) // 2
        
        for i, text in enumerate(algorithms):
            x = start_x + i * (self.button_width + self.button_margin)
            y = self.height - 80
            buttons.append({
                'rect': pygame.Rect(x, y, self.button_width, self.button_height),
                'text': text,
                'action': text.lower().replace(" ", "_")
            })
        return buttons

    def draw_board(self):
        # Draw grid
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                x = self.grid_offset_x + col * self.cell_size
                y = self.grid_offset_y + row * self.cell_size
                
                # Draw cell
                cell_color = self.WHITE
                if self.current_solution and self.current_solution[row][col] == 1:
                    cell_color = self.BLACK
                
                pygame.draw.rect(self.screen, cell_color, 
                               (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.GRAY, 
                               (x, y, self.cell_size, self.cell_size), 1)
                
                # Draw numbers
                if self.board[row][col] is not None:
                    text = self.font.render(str(self.board[row][col]), True, self.BLUE)
                    text_rect = text.get_rect(center=(x + self.cell_size//2, 
                                                    y + self.cell_size//2))
                    self.screen.blit(text, text_rect)

    def draw_buttons(self):
        for button in self.buttons:
            color = self.GREEN if button['text'] == "Solve DFS" and self.solving else self.GRAY
            pygame.draw.rect(self.screen, color, button['rect'])
            text = self.font.render(button['text'], True, self.BLACK)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)

    def draw_steps(self):
        if self.solving:
            text = self.font.render(f"Step: {self.current_step}/{self.total_steps}", True, self.BLACK)
        else:
            text = self.font.render("Click a solver button to start", True, self.BLACK)
        self.screen.blit(text, (20, 20))

    def log_state(self, step_num):
        filename = f"{self.output_dir}/step_{self.algorithm}_{step_num}.txt"
        with open(filename, 'w') as f:
            f.write(f"Step {step_num}/{self.total_steps}\n")
            f.write(f"Algorithm: {self.algorithm.upper()}\n\n")
            
            for row in range(len(self.current_solution)):
                for col in range(len(self.current_solution[0])):
                    if self.board[row][col] is not None:
                        f.write(f" {self.board[row][col]} ")
                    elif self.current_solution[row][col] == 1:
                        f.write(" # ")
                    else:
                        f.write(" . ")
                    if col % 3 == 2 and col < len(self.current_solution[0]) - 1:
                        f.write("|")
                f.write("\n")
                if row % 3 == 2 and row < len(self.current_solution) - 1:
                    f.write("-" * (len(self.current_solution[0]) * 3 + 2) + "\n")
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
                board_row = [int(cell) if cell.isdigit() else None for cell in line.split()]
                step["board"].append(board_row)

        if step is not None:
            self.solution_steps.append(step)
    def solve_puzzle(self, algorithm):
        self.algorithm = algorithm
        self.solving = True
        self.current_step = 0
        
        # Clear previous solution file
        filename = f"{self.output_dir}/solution_{algorithm}.txt"
        if os.path.exists(filename):
            os.remove(filename)
        
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
        self.solution_steps = self.load_steps_from_log("output/" + "output_" + temp + ".txt")
        
        if os.path.exists(filename):
            current_board = None
            with open(filename, 'r') as f:
                lines = f.readlines()
                i = 2  # Skip algorithm header
                while i < len(lines):
                    if lines[i].startswith("Step"):
                        board = [[0 for _ in range(len(self.board[0]))] 
                                for _ in range(len(self.board))]
                        i += 1  # Move to first row of the board
                        
                        row = 0
                        while i < len(lines) and not lines[i].isspace():
                            if '|' in lines[i] or '-' in lines[i]:  # Skip separator lines
                                i += 1
                                continue
                            
                            cells = lines[i].strip().split()
                            for col, char in enumerate(cells):
                                if char == '#':
                                    board[row][col] = 1
                            row += 1
                            i += 1
                        
                        self.solution_steps.append(board)
                        i += 1  # Skip blank line
                    else:
                        i += 1
        
        self.total_steps = len(self.solution_steps)
        if self.total_steps > 0:
            self.current_solution = self.solution_steps[0]

    def update(self):
        current_time = pygame.time.get_ticks()
        if self.solving and self.current_step < self.total_steps - 1:
            if current_time - self.last_step_time >= self.step_delay:
                self.current_step += 1
                self.current_solution = self.solution_steps[self.current_step]
                self.log_state(self.current_step)
                self.last_step_time = current_time

    def reset(self):
        self.current_solution = None
        self.solution_steps = []
        self.current_step = 0
        self.total_steps = 0
        self.solving = False
        self.algorithm = None

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
                        if button['rect'].collidepoint(mouse_pos):
                            if button['action'] == 'reset':
                                self.reset()
                            else:
                                self.solve_puzzle(button['action'])
            
            # Update solution state
            self.update()
            
            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_board()
            self.draw_buttons()
            self.draw_steps()
            pygame.display.flip()
            
            # Control frame rate
            clock.tick(60)

if __name__ == "__main__":
    game = MosaicUI()
    game.run() 