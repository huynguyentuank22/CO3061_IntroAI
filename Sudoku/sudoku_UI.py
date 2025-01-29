import pygame
from sudoku_solver import Sudoku
from copy import deepcopy
import time
import tracemalloc
import sys

class SudokuUI:
    def __init__(self, sudoku):
        # Initialize PyGame and UI properties
        pygame.init()
        self.sudoku = sudoku
        self.cell_size = 60
        self.grid_size = self.cell_size * 9
        self.margin = 40
        self.window_size = (self.grid_size + 2 * self.margin, self.grid_size + 100)
        
        # Setup display and fonts
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('Sudoku Solver Visualizer')
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Solving visualization properties
        self.solving_steps = []
        self.current_step = 0
        self.is_solving = False
        
        # Store original board for coloring
        self.original_board = [[num for num in row] for row in sudoku.board]
        
        # Add selected cell tracking
        self.selected_cell = None
        
        # Add game state
        self.player_mode = True  # True for player mode, False for solving mode
        
        # Add conflict highlighting
        self.conflicting_cells = []
        
        # Add animation delay for solving
        self.solving_delay = 100  # milliseconds
        self.last_step_time = 0
        
        # Add visualization timing
        self.step_delay = 50  # 0.5 seconds between steps
        
        # Add performance tracking
        self.start_time = None
        self.execution_time = None
        self.peak_memory = None
        self.step_count = 0
        
        # Add solution steps from file
        self.solution_steps = []
        self.load_solution_steps()
        
        # Add success notification
        self.solved_by_player = False

    def find_conflicts(self, num, pos):
        """Find all cells that conflict with the given number at position"""
        conflicts = []
        row, col = pos
        
        # Check row
        for j in range(9):
            if j != col and self.sudoku.board[row][j] == num:
                conflicts.append((row, j))
                
        # Check column
        for i in range(9):
            if i != row and self.sudoku.board[i][col] == num:
                conflicts.append((i, col))
                
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if (i != row or j != col) and self.sudoku.board[i][j] == num:
                    conflicts.append((i, j))
                    
        return conflicts

    def draw_board(self):
        """Draw the Sudoku board and UI elements"""
        self.screen.fill((255, 255, 255))  # White background
        
        # Draw step information above the board
        if self.is_solving and self.solution_steps:
            step_text = self.small_font.render(
                f'Step: {self.current_step}/{len(self.solution_steps)-1}', 
                True, (100, 100, 100))
            self.screen.blit(step_text, 
                            (self.margin, self.margin - 25))  # Position above the board

        # Draw grid lines
        for i in range(10):
            line_width = 3 if i % 3 == 0 else 1
            # Vertical lines
            pygame.draw.line(self.screen, (0, 0, 0),
                           (self.margin + i * self.cell_size, self.margin),
                           (self.margin + i * self.cell_size, self.margin + self.grid_size),
                           line_width)
            # Horizontal lines
            pygame.draw.line(self.screen, (0, 0, 0),
                           (self.margin, self.margin + i * self.cell_size),
                           (self.margin + self.grid_size, self.margin + i * self.cell_size),
                           line_width)

        # Highlight conflicting cells
        for row, col in self.conflicting_cells:
            pygame.draw.rect(self.screen, (255, 200, 200),  # Light red
                           (self.margin + col * self.cell_size,
                            self.margin + row * self.cell_size,
                            self.cell_size, self.cell_size))

        # Highlight selected cell
        if self.selected_cell and self.player_mode:
            row, col = self.selected_cell
            pygame.draw.rect(self.screen, (173, 216, 230),  # Light blue
                           (self.margin + col * self.cell_size,
                            self.margin + row * self.cell_size,
                            self.cell_size, self.cell_size))

        # Draw numbers
        for i in range(9):
            for j in range(9):
                if self.sudoku.board[i][j] != 0:
                    # Different colors for original vs player/solved numbers
                    if self.sudoku.board[i][j] == self.original_board[i][j]:
                        color = (0, 0, 139)  # Dark blue for original numbers
                    elif not self.is_solving:
                        color = (0, 0, 0)    # Black for player numbers
                    else:
                        color = (0, 100, 0)  # Green for solved numbers
                    num_surface = self.font.render(str(self.sudoku.board[i][j]), True, color)
                    x = self.margin + j * self.cell_size + (self.cell_size - num_surface.get_width()) // 2
                    y = self.margin + i * self.cell_size + (self.cell_size - num_surface.get_height()) // 2
                    self.screen.blit(num_surface, (x, y))

        # Draw buttons
        button_height = 40
        button_width = 120
        button_margin = 20
        
        # DFS Button
        self.dfs_button = pygame.draw.rect(self.screen, (70, 130, 180),
                                         (self.margin, self.margin + self.grid_size + button_margin, 
                                          button_width, button_height))
        pygame.draw.rect(self.screen, (30, 100, 150),
                        (self.margin, self.margin + self.grid_size + button_margin, 
                         button_width, button_height), 2)

        # A* Button
        self.astar_button = pygame.draw.rect(self.screen, (106, 170, 100),
                                           (self.margin + button_width + button_margin, 
                                            self.margin + self.grid_size + button_margin,
                                            button_width, button_height))
        pygame.draw.rect(self.screen, (76, 140, 70),
                        (self.margin + button_width + button_margin,
                         self.margin + self.grid_size + button_margin,
                         button_width, button_height), 2)

        # Draw button text
        dfs_text = self.small_font.render('Solve DFS', True, (255, 255, 255))
        astar_text = self.small_font.render('Solve A*', True, (255, 255, 255))
        
        # Center text on buttons
        dfs_x = self.margin + (button_width - dfs_text.get_width()) // 2
        astar_x = self.margin + button_width + button_margin + (button_width - astar_text.get_width()) // 2
        text_y = self.margin + self.grid_size + button_margin + (button_height - dfs_text.get_height()) // 2
        
        self.screen.blit(dfs_text, (dfs_x, text_y))
        self.screen.blit(astar_text, (astar_x, text_y))

        # Update instructions based on mode
        if self.player_mode:
            instruction_text1 = self.small_font.render('Click cell and press 1-9 to fill.', 
                                                    True, (100, 100, 100))
            instruction_text2 = self.small_font.render('Press DEL/BACKSPACE to clear.', 
                                                    True, (100, 100, 100))
            
            # Position both instruction text lines
            self.screen.blit(instruction_text1, 
                            (self.margin + 2 * button_width + 2 * button_margin, 
                             self.margin + self.grid_size + button_margin))
            self.screen.blit(instruction_text2, 
                            (self.margin + 2 * button_width + 2 * button_margin, 
                             self.margin + self.grid_size + button_margin + 20))
        else:
            instruction_text1 = self.small_font.render('Wait for automatic progression', 
                                                    True, (100, 100, 100))
            # Position single instruction text
            self.screen.blit(instruction_text1, 
                            (self.margin + 2 * button_width + 2 * button_margin, 
                             self.margin + self.grid_size + button_margin + 10))

        # Draw performance metrics during/after solving
        if self.is_solving or self.execution_time:
            metrics_y = self.margin + self.grid_size + 70
            if self.step_count > 0:
                step_text = self.small_font.render(f'Steps: {self.step_count}', True, (100, 100, 100))
                self.screen.blit(step_text, (self.margin + 2 * button_width + 2 * button_margin, metrics_y))
            
            if self.execution_time:
                time_text = self.small_font.render(
                    f'Time: {self.execution_time:.4f}s', True, (100, 100, 100))
                self.screen.blit(time_text, (self.margin + 2 * button_width + 2 * button_margin, metrics_y + 20))
            
            if self.peak_memory:
                memory_text = self.small_font.render(
                    f'Memory: {self.peak_memory / 1024:.2f} KB', True, (100, 100, 100))
                self.screen.blit(memory_text, (self.margin + 2 * button_width + 2 * button_margin, metrics_y + 40))

        # Draw success notification if solved by player
        if self.solved_by_player:
            success_text = self.font.render('Puzzle Solved!', True, (0, 128, 0))  # Green color
            text_rect = success_text.get_rect(center=(self.window_size[0] // 2, self.margin - 25))
            self.screen.blit(success_text, text_rect)

        pygame.display.flip()

    def check_solution(self):
        """Check if the current board state is a valid solution"""
        # Check if board is filled
        if any(0 in row for row in self.sudoku.board):
            return False
            
        # Check each row, column and box
        for i in range(9):
            # Check rows
            if len(set(self.sudoku.board[i])) != 9:
                return False
            # Check columns
            if len(set(self.sudoku.board[j][i] for j in range(9))) != 9:
                return False
            # Check 3x3 boxes
            box_row, box_col = 3 * (i // 3), 3 * (i % 3)
            box = []
            for r in range(box_row, box_row + 3):
                for c in range(box_col, box_col + 3):
                    box.append(self.sudoku.board[r][c])
            if len(set(box)) != 9:
                return False
        return True

    def handle_player_input(self, event):
        """Handle player input for filling numbers"""
        if not self.player_mode or not self.selected_cell:
            return
            
        row, col = self.selected_cell
        
        # Handle arrow keys
        if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
            new_row, new_col = row, col
            if event.key == pygame.K_LEFT:
                new_col = (col - 1) % 9
            elif event.key == pygame.K_RIGHT:
                new_col = (col + 1) % 9
            elif event.key == pygame.K_UP:
                new_row = (row - 1) % 9
            elif event.key == pygame.K_DOWN:
                new_row = (row + 1) % 9
            self.selected_cell = (new_row, new_col)
            self.conflicting_cells = []
            return
            
        if event.key in range(pygame.K_1, pygame.K_9 + 1):
            num = event.key - pygame.K_0
            if self.original_board[row][col] == 0:  # Only allow editing empty cells
                # Clear previous conflicts
                self.conflicting_cells = []
                
                # Check for conflicts before placing number
                conflicts = self.find_conflicts(num, (row, col))
                if conflicts:
                    self.conflicting_cells = conflicts + [self.selected_cell]
                else:
                    self.sudoku.board[row][col] = num
                    # Check if puzzle is solved
                    if self.check_solution():
                        self.solved_by_player = True
                    
        elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
            if self.original_board[row][col] == 0:  # Only allow clearing player-added numbers
                self.sudoku.board[row][col] = 0
                self.conflicting_cells = []
                self.solved_by_player = False

    def handle_cell_click(self, mouse_pos):
        """Handle mouse click on cells"""
        if not self.player_mode:
            return
            
        x, y = mouse_pos
        if (self.margin <= x <= self.margin + self.grid_size and
            self.margin <= y <= self.margin + self.grid_size):
            row = (y - self.margin) // self.cell_size
            col = (x - self.margin) // self.cell_size
            self.selected_cell = (row, col)

    def load_solution_steps(self):
        """Load solution steps from output file"""
        try:
            with open(self.sudoku.output_file, "r") as file:
                current_board = []
                seen_boards = set()  # Track unique board states
                
                for line in file:
                    if line.startswith("Step"):
                        if current_board:
                            # Convert board to string for comparison
                            board_str = str(current_board)
                            if board_str not in seen_boards:
                                self.solution_steps.append([row[:] for row in current_board])
                                seen_boards.add(board_str)
                        current_board = []
                    elif "|" in line and "-" not in line:  # Board row, excluding separator lines
                        # Extract numbers from the line, converting dots to zeros
                        row = []
                        for part in line.strip().split("|"):
                            numbers = [int(n) if n.strip() != "." else 0 
                                     for n in part.strip().split()]
                            row.extend(numbers)
                        if len(row) == 9:  # Valid row
                            current_board.append(row)
                        
                # Add final board if it exists and is unique
                if current_board and str(current_board) not in seen_boards:
                    self.solution_steps.append([row[:] for row in current_board])
                
            # Ensure we have the initial and final states
            if len(self.solution_steps) > 0:
                print(f"Loaded {len(self.solution_steps)} unique solution steps")
            
        except FileNotFoundError:
            print(f"Output file not found: {self.sudoku.output_file}")

    def start_solving(self, algorithm):
        """Initialize solving visualization"""
        # Reset board to initial state
        self.sudoku.board = [row[:] for row in self.original_board]
        
        # Reset all solving-related variables
        self.player_mode = False
        self.is_solving = True
        self.current_step = 0
        self.step_count = 0
        self.last_step_time = pygame.time.get_ticks()
        
        # Clear previous solution steps
        self.solution_steps = []
        
        # Clear the output file before solving
        with open(self.sudoku.output_file, "w") as file:
            file.write("")
        
        # Solve with selected algorithm and collect steps
        if algorithm == 'DFS':
            self.sudoku.solver('DFS')
        else:
            self.sudoku.solver('A*')
        
        # Load the new solution steps
        self.load_solution_steps()
        
        # Start visualization from initial state
        if self.solution_steps:
            self.sudoku.board = deepcopy(self.solution_steps[0])
        else:
            print("No solution steps loaded")

    def show_next_step(self):
        """Show the next step in the solution"""
        current_time = pygame.time.get_ticks()
        if (self.is_solving and 
            self.current_step < len(self.solution_steps) - 1 and 
            current_time - self.last_step_time >= self.step_delay):
            
            self.current_step += 1
            self.step_count += 1
            self.sudoku.board = deepcopy(self.solution_steps[self.current_step])
            self.last_step_time = current_time

    def run(self):
        """Main game loop"""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if self.dfs_button.collidepoint(mouse_pos):
                        self.start_solving('DFS')
                    elif self.astar_button.collidepoint(mouse_pos):
                        self.start_solving('A*')
                    else:
                        self.handle_cell_click(mouse_pos)
                        self.conflicting_cells = []  # Clear conflicts when selecting new cell
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.is_solving:
                        self.show_next_step()
                    else:
                        self.handle_player_input(event)
            
            # Automatic step progression during solving
            if self.is_solving:
                self.show_next_step()
            
            self.draw_board()
            clock.tick(60)
        
        pygame.quit()

if __name__ == '__main__':
    # Get test case from command line argument, default to "tc1" if not provided
    testcase = sys.argv[1] if len(sys.argv) > 1 else "tc1"
    
    # Create Sudoku instance with test case
    sudoku = Sudoku(testcase)
    
    # Create and run UI
    ui = SudokuUI(sudoku)
    ui.run() 