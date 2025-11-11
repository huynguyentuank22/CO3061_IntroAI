import pygame
import sys
from ultimate_board import UltimateBoard

class GameUI:
    def __init__(self, width=900, height=970):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Ultimate Tic Tac Toe")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.LIGHT_BLUE = (173, 216, 230)
        
        # Board dimensions
        self.cell_size = width // 9
        self.board_size = self.cell_size * 3
        
        # Font
        self.font = pygame.font.SysFont('Arial', 30)
        self.small_font = pygame.font.SysFont('Arial', 20)

        # Top UI bar and Pause button (outside board area)
        self.ui_bar_height = 70
        self.board_origin_y = self.ui_bar_height
        self.pause_button = pygame.Rect(self.width - 140, 15, 120, 40)

        # Pause menu overlay
        self.is_showing_pause_menu = False
        self.pause_menu_buttons = {
            'continue': pygame.Rect(self.width // 2 - 150, self.height // 2 - 100, 300, 60),
            'quit': pygame.Rect(self.width // 2 - 150, self.height // 2 + 60, 300, 60)
        }
        # Restart is optional (hidden in network mode)
        self.pause_button_restart_rect = pygame.Rect(self.width // 2 - 150, self.height // 2 - 20, 300, 60)
        self.show_restart_in_pause = True
        # Optional countdown seconds display in pause menu (network mode)
        self.pause_countdown_seconds = None
        
    def draw_board(self, board):
        """Draw the entire Ultimate Tic Tac Toe board"""
        self.screen.fill(self.WHITE)
        
        # Top UI bar
        pygame.draw.rect(self.screen, (245, 245, 245), (0, 0, self.width, self.ui_bar_height))
        pygame.draw.line(self.screen, self.GRAY, (0, self.ui_bar_height), (self.width, self.ui_bar_height), 2)
        self.draw_pause_button()

        # Draw small boards
        for board_row in range(3):
            for board_col in range(3):
                # Calculate the position of this small board
                x0 = board_col * self.board_size
                y0 = self.board_origin_y + board_row * self.board_size
                
                # Highlight active board
                if board.active_board == (board_row, board_col) or board.active_board is None:
                    pygame.draw.rect(self.screen, self.LIGHT_BLUE, 
                                    (x0, y0, self.board_size, self.board_size))
                
                # Draw this small board
                self._draw_small_board(board.boards[board_row][board_col], x0, y0)
                
                # Mark if this small board is won
                if board.boards[board_row][board_col].winner:
                    self._draw_board_winner(board.boards[board_row][board_col].winner, x0, y0)
        
        # Draw grid lines for the large board
        for i in range(1, 3):
            # Vertical lines
            pygame.draw.line(self.screen, self.BLACK, 
                           (i * self.board_size, self.board_origin_y), 
                           (i * self.board_size, self.board_origin_y + 3 * self.board_size), 5)
            # Horizontal lines
            pygame.draw.line(self.screen, self.BLACK, 
                           (0, self.board_origin_y + i * self.board_size), 
                           (self.width, self.board_origin_y + i * self.board_size), 5)
        
        # Draw game status
        if board.winner:
            status_text = f"Player {board.winner} wins!"
            text = self.font.render(status_text, True, self.GREEN)
            self.screen.blit(text, (20, 18))
        elif board.is_draw:
            status_text = "Game is a draw!"
            text = self.font.render(status_text, True, self.BLUE)
            self.screen.blit(text, (20, 18))
        else:
            status_text = f"Player {board.current_player}'s turn"
            text = self.font.render(status_text, True, self.BLACK)
            self.screen.blit(text, (20, 18))
            
            if board.active_board:
                hint_text = f"Play in board ({board.active_board[0]+1},{board.active_board[1]+1})"
                hint = self.small_font.render(hint_text, True, self.BLACK)
                self.screen.blit(hint, (20, 45))
            else:
                hint_text = "Play in any board"
                hint = self.small_font.render(hint_text, True, self.BLACK)
                self.screen.blit(hint, (20, 45))

        # Draw pause overlay if active
        if self.is_showing_pause_menu:
            self.draw_pause_menu()
        
        pygame.display.flip()
    
    def draw_pause_button(self):
        """Draw the pause button in the top UI bar"""
        mouse_pos = pygame.mouse.get_pos()
        if self.pause_button.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, self.LIGHT_BLUE, self.pause_button, border_radius=6)
        else:
            pygame.draw.rect(self.screen, self.GRAY, self.pause_button, border_radius=6)
        pygame.draw.rect(self.screen, self.BLACK, self.pause_button, 2, border_radius=6)

        text = self.small_font.render("Pause", True, self.BLACK)
        self.screen.blit(text, (self.pause_button.centerx - text.get_width() // 2, 
                                self.pause_button.centery - text.get_height() // 2))
    
    def is_pause_button_clicked(self, pos):
        """Check if the pause button was clicked"""
        return self.pause_button.collidepoint(pos)

    def draw_pause_menu(self):
        """Draw a semi-transparent overlay with pause menu options"""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        panel_rect = pygame.Rect(self.width // 2 - 220, self.height // 2 - 160, 440, 320)
        pygame.draw.rect(self.screen, (250, 250, 250), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.BLACK, panel_rect, 2, border_radius=8)

        title = self.font.render("Paused", True, self.BLACK)
        self.screen.blit(title, (panel_rect.centerx - title.get_width() // 2, panel_rect.top + 20))

        mouse_pos = pygame.mouse.get_pos()
        for key, rect in self.pause_menu_buttons.items():
            color = self.LIGHT_BLUE if rect.collidepoint(mouse_pos) else self.GRAY
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            pygame.draw.rect(self.screen, self.BLACK, rect, 2, border_radius=6)
            label = {"continue": "Continue", "restart": "Restart Game", "quit": "Quit Game"}[key]
            text = self.small_font.render(label, True, self.BLACK)
            self.screen.blit(text, (rect.centerx - text.get_width() // 2, 
                                    rect.centery - text.get_height() // 2))

        # Optional Restart in the middle (for non-network games)
        if self.show_restart_in_pause:
            color = self.LIGHT_BLUE if self.pause_button_restart_rect.collidepoint(mouse_pos) else self.GRAY
            pygame.draw.rect(self.screen, color, self.pause_button_restart_rect, border_radius=6)
            pygame.draw.rect(self.screen, self.BLACK, self.pause_button_restart_rect, 2, border_radius=6)
            text = self.small_font.render("Restart Game", True, self.BLACK)
            self.screen.blit(text, (self.pause_button_restart_rect.centerx - text.get_width() // 2, 
                                    self.pause_button_restart_rect.centery - text.get_height() // 2))

        # Optional countdown
        if self.pause_countdown_seconds is not None:
            countdown_text = self.small_font.render(f"Resuming in: {int(self.pause_countdown_seconds)}s", True, self.BLACK)
            self.screen.blit(countdown_text, (panel_rect.centerx - countdown_text.get_width() // 2,
                                              panel_rect.top + 70))

    def pause_menu_button_from_pos(self, pos):
        """Return which pause menu button is clicked, or None"""
        for key, rect in self.pause_menu_buttons.items():
            if rect.collidepoint(pos):
                return key
        if self.show_restart_in_pause and self.pause_button_restart_rect.collidepoint(pos):
            return 'restart'
        return None
    
    def _draw_small_board(self, small_board, x0, y0):
        """Draw a single small board"""
        # Draw grid lines for the small board
        for i in range(1, 3):
            # Vertical lines
            pygame.draw.line(self.screen, self.GRAY, 
                           (x0 + i * self.cell_size, y0), 
                           (x0 + i * self.cell_size, y0 + self.board_size), 2)
            # Horizontal lines
            pygame.draw.line(self.screen, self.GRAY, 
                           (x0, y0 + i * self.cell_size), 
                           (x0 + self.board_size, y0 + i * self.cell_size), 2)
        
        # Draw X's and O's
        for row in range(3):
            for col in range(3):
                cell_x = x0 + col * self.cell_size
                cell_y = y0 + row * self.cell_size
                
                if small_board.board[row][col] == 'X':
                    self._draw_x(cell_x, cell_y)
                elif small_board.board[row][col] == 'O':
                    self._draw_o(cell_x, cell_y)
    
    def _draw_x(self, x, y):
        """Draw an X in the specified cell"""
        margin = self.cell_size // 4
        pygame.draw.line(self.screen, self.RED, 
                       (x + margin, y + margin), 
                       (x + self.cell_size - margin, y + self.cell_size - margin), 3)
        pygame.draw.line(self.screen, self.RED, 
                       (x + self.cell_size - margin, y + margin), 
                       (x + margin, y + self.cell_size - margin), 3)
    
    def _draw_o(self, x, y):
        """Draw an O in the specified cell"""
        margin = self.cell_size // 4
        center_x = x + self.cell_size // 2
        center_y = y + self.cell_size // 2
        radius = self.cell_size // 2 - margin
        pygame.draw.circle(self.screen, self.BLUE, (center_x, center_y), radius, 3)
    
    def _draw_board_winner(self, winner, x, y):
        """Mark a small board that has been won"""
        color = self.RED if winner == 'X' else self.BLUE
        
        # Draw semi-transparent overlay
        overlay = pygame.Surface((self.board_size, self.board_size), pygame.SRCALPHA)
        overlay.fill((color[0], color[1], color[2], 100))  # Semi-transparent
        self.screen.blit(overlay, (x, y))
        
        # Draw winner mark
        text = self.font.render(winner, True, color)
        self.screen.blit(text, (x + self.board_size // 2 - text.get_width() // 2, 
                              y + self.board_size // 2 - text.get_height() // 2))
    
    def get_cell_from_click(self, pos):
        """Convert mouse position to board coordinates"""
        x, y = pos
        
        # Get the small board coordinates
        board_col = x // self.board_size
        if y < self.board_origin_y:
            return None
        board_row = (y - self.board_origin_y) // self.board_size
        
        # Get the cell coordinates within the small board
        x_offset = x % self.board_size
        y_offset = (y - self.board_origin_y) % self.board_size
        col = x_offset // self.cell_size
        row = y_offset // self.cell_size
        
        return board_row, board_col, row, col
