import pygame
import sys
import time
import random
from ultimate_board import UltimateBoard
from game_ui import GameUI
from player import HumanPlayer, RandomPlayer, MinimaxPlayer, MCTSPlayer, ModelPlayer
from battle_logger import BattleLogger
from training_logger import TrainingDataLogger

class Game:
    def __init__(self):
        self.board = UltimateBoard()
        self.ui = GameUI()
        self.clock = pygame.time.Clock()
        
        # Default players
        self.players = {
            'X': HumanPlayer('X'),
            'O': None  # Will be set when game starts
        }
        
        self.ai_options = {
            'Random': RandomPlayer,
            'Easy': lambda mark: MinimaxPlayer(mark, depth=1),
            'Medium': lambda mark: MinimaxPlayer(mark, depth=3),
            'Hard': lambda mark: MinimaxPlayer(mark, depth=5),
            'MCTS': lambda mark: MCTSPlayer(mark, simulation_time=1.0),
            'Model': lambda mark: ModelPlayer(mark, model_path='model.pt', temperature=0.5)
        }
        
        self.running = False
        self.agent_mode = False
        self.move_delay = 0.5  # Delay in seconds between AI moves (for visualization)
        
        # Store player types for restart
        self.player_x_type = 'Human'
        self.player_o_type = 'Random'
        
        # For logging
        self.logger = BattleLogger()
        self.training_logger = TrainingDataLogger()
        self.start_time = None
        self.move_count = 0
        
    def start_game(self, player_o_type='Random'):
        """Start the game with human player vs selected AI opponent"""
        self.board = UltimateBoard()  # Reset board
        
        # Set players
        self.players['X'] = HumanPlayer('X')
        self.players['O'] = self.ai_options[player_o_type]('O')
        
        # Store player types
        self.player_x_type = 'Human'
        self.player_o_type = player_o_type
        
        # Initialize game statistics
        self.start_time = time.time()
        self.move_count = 0
        
        self.agent_mode = False
        self.running = True
        self.main_loop()
    
    def start_game_agents(self, player_x_type, player_o_type):
        """Start the game with two AI agents playing against each other"""
        self.board = UltimateBoard()  # Reset board
        
        # Set both players as AI agents
        self.players['X'] = self.ai_options[player_x_type]('X')
        self.players['O'] = self.ai_options[player_o_type]('O')
        
        # Store player types
        self.player_x_type = player_x_type
        self.player_o_type = player_o_type
        
        # Initialize game statistics
        self.start_time = time.time()
        self.move_count = 0
        
        self.agent_mode = True
        self.running = True
        self.main_loop()
    
    def restart_game(self):
        """Restart the current game with the same player types"""
        # Reset the board
        self.board = UltimateBoard()
        
        # Reinitialize players with the same settings
        if self.player_x_type == 'Human':
            self.players['X'] = HumanPlayer('X')
        else:
            self.players['X'] = self.ai_options[self.player_x_type]('X')
            
        self.players['O'] = self.ai_options[self.player_o_type]('O')
        
        # Reset game statistics
        self.start_time = time.time()
        self.move_count = 0
    
    def main_loop(self):
        """Main game loop"""
        while self.running:
            # Draw the board
            self.ui.draw_board(self.board)
            
            # Handle current player's move
            current_mark = self.board.current_player
            current_player = self.players[current_mark]
            
            # Special handling for agent vs agent mode
            if self.agent_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:  # Restart game
                            self.restart_game()
                            continue
                        if event.key == pygame.K_q:  # Quit
                            pygame.quit()
                            sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        pos = pygame.mouse.get_pos()
                        if self.ui.is_restart_button_clicked(pos):
                            self.restart_game()
                            continue
                
                # Show which agent is thinking
                pygame.display.set_caption(f"Ultimate Tic Tac Toe - {current_mark} ({type(current_player).__name__}) thinking...")
                
                # Get AI move
                move = current_player.get_move(self.board)
                pygame.display.set_caption("Ultimate Tic Tac Toe - Agent vs Agent")
                
                if move:
                    board_row, board_col, row, col = move
                    self.board.make_move(board_row, board_col, row, col)
                    self.move_count += 1
                    
                    # Add delay to visualize the moves
                    time.sleep(self.move_delay)
            
            # If human player
            elif isinstance(current_player, HumanPlayer):
                self._handle_human_move()
            
            # If AI player
            else:
                pygame.display.set_caption(f"Ultimate Tic Tac Toe - AI thinking...")
                move = current_player.get_move(self.board)
                pygame.display.set_caption("Ultimate Tic Tac Toe")
                
                if move:
                    board_row, board_col, row, col = move
                    self.board.make_move(board_row, board_col, row, col)
                    self.move_count += 1
                    time.sleep(self.move_delay)  # Small delay to see AI moves
            
            # Check for game end
            if self.board.winner is not None or self.board.is_draw:
                self.ui.draw_board(self.board)
                
                # Log the result if agent vs agent mode
                if self.agent_mode:
                    game_duration = time.time() - self.start_time
                    self.logger.log_battle(
                        self.player_x_type, 
                        self.player_o_type, 
                        self.board.winner, 
                        self.move_count, 
                        game_duration, 
                        self.board
                    )
                
                self._handle_game_over()
                
            # Cap the frame rate
            self.clock.tick(30)
    
    def _handle_human_move(self):
        """Handle human player input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                pos = pygame.mouse.get_pos()
                
                # Check if restart button was clicked
                if self.ui.is_restart_button_clicked(pos):
                    self.restart_game()
                    return
                
                # Handle game board click
                board_row, board_col, row, col = self.ui.get_cell_from_click(pos)
                
                # Try to make the move
                self.board.make_move(board_row, board_col, row, col)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Restart game
                    self.restart_game()
                    return
                if event.key == pygame.K_q:  # Quit
                    pygame.quit()
                    sys.exit()
    
    def _handle_game_over(self):
        """Handle game over state"""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Restart
                        self.restart_game()
                        waiting = False
                    if event.key == pygame.K_q:  # Quit
                        pygame.quit()
                        sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = pygame.mouse.get_pos()
                    if self.ui.is_restart_button_clicked(pos):
                        self.restart_game()
                        waiting = False
            
            # Display game over message
            font = pygame.font.SysFont('Arial', 30)
            if self.board.winner:
                text = font.render(f"Game over! Player {self.board.winner} wins! Press R to restart or Q to quit", True, (0, 255, 0))
            else:
                text = font.render("Game over! It's a draw! Press R to restart or Q to quit", True, (0, 0, 255))
                
            text_rect = text.get_rect(center=(self.ui.width // 2, self.ui.height - 50))
            self.ui.screen.blit(text, text_rect)
            
            # Make sure restart button is visible
            self.ui.draw_restart_button()
            
            pygame.display.flip()
    
    def run_batch_simulation(self, num_battles=100, display_progress=True, collect_training_data=False):
        """Run a batch of agent vs agent battles with random agent types"""
        # Get list of available agent types (excluding Human)
        agent_types = list(self.ai_options.keys())
        
        # Create pygame window for displaying progress if needed
        if display_progress:
            pygame.init()
            width, height = 600, 300
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Batch Simulation Progress")
            font = pygame.font.SysFont('Arial', 24)
            small_font = pygame.font.SysFont('Arial', 18)
            WHITE = (255, 255, 255)
            BLACK = (0, 0, 0)
            GREEN = (0, 255, 0)
        
        completed_battles = 0
        
        print(f"Starting batch simulation of {num_battles} battles...")
        
        simulation_start_time = time.time()
        
        for i in range(num_battles):
            # Handle pygame events to prevent "not responding"
            if display_progress:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return completed_battles
            
            # Select random agent types
            agent_x_type = random.choice(agent_types)
            agent_o_type = random.choice(agent_types)
            
            print(f"Battle {i+1}/{num_battles}: {agent_x_type} vs {agent_o_type}")
            
            # Reset the board
            self.board = UltimateBoard()
            
            # Set both players as AI agents
            self.players['X'] = self.ai_options[agent_x_type]('X')
            self.players['O'] = self.ai_options[agent_o_type]('O')
            
            # Store player types
            self.player_x_type = agent_x_type
            self.player_o_type = agent_o_type
            
            # Initialize game statistics
            self.start_time = time.time()
            self.move_count = 0
            
            # Run the game without UI until completion
            while self.board.winner is None and not self.board.is_draw:
                current_mark = self.board.current_player
                current_player = self.players[current_mark]
                current_player_type = self.player_x_type if current_mark == 'X' else self.player_o_type
                
                # Get AI move
                move = current_player.get_move(self.board)
                
                if move:
                    board_row, board_col, row, col = move
                    
                    # For training data collection, log move before it's made
                    if collect_training_data:
                        # Convert the move to a 0-80 index
                        move_index = (board_row * 3 + board_col) * 9 + (row * 3 + col)
                        self.training_logger.log_move(self.board, move_index, current_player_type, self.move_count)
                    
                    # Make the move
                    self.board.make_move(board_row, board_col, row, col)
                    self.move_count += 1
            
            # Game over, update training data with results
            if collect_training_data:
                self.training_logger.update_game_results(self.board.winner)
            
            # Log the battle result
            game_duration = time.time() - self.start_time
            self.logger.log_battle(
                self.player_x_type, 
                self.player_o_type, 
                self.board.winner, 
                self.move_count, 
                game_duration, 
                self.board
            )
            
            completed_battles += 1
            
            # Update progress display
            if display_progress and i % 5 == 0:  # Update every 5 battles to avoid slowdowns
                screen.fill(WHITE)
                
                # Progress percentage
                progress = (i + 1) / num_battles * 100
                
                # Draw progress bar
                bar_width = width - 40
                bar_height = 30
                outline_rect = pygame.Rect(20, 70, bar_width, bar_height)
                pygame.draw.rect(screen, BLACK, outline_rect, 2)
                
                fill_width = int(bar_width * (i + 1) / num_battles)
                fill_rect = pygame.Rect(20, 70, fill_width, bar_height)
                pygame.draw.rect(screen, GREEN, fill_rect)
                
                # Draw text
                progress_text = font.render(f"Progress: {progress:.1f}%", True, BLACK)
                screen.blit(progress_text, (20, 20))
                
                battle_text = small_font.render(f"Battle {i+1}/{num_battles}: {agent_x_type} vs {agent_o_type}", True, BLACK)
                screen.blit(battle_text, (20, 120))
                
                if collect_training_data:
                    training_text = small_font.render("Collecting training data for neural network", True, BLACK)
                    screen.blit(training_text, (20, 150))
                
                if i > 0:
                    elapsed = time.time() - simulation_start_time
                    estimated_total = elapsed / (i + 1) * num_battles
                    remaining = estimated_total - elapsed
                    
                    time_text = small_font.render(
                        f"Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | " +
                        f"Remaining: {int(remaining//60)}m {int(remaining%60)}s", 
                        True, BLACK)
                    screen.blit(time_text, (20, 180))
                
                pygame.display.flip()
        
        # Ensure all remaining training data is written to disk
        if collect_training_data:
            self.training_logger.flush()
            
        if display_progress:
            # Show completion message
            screen.fill(WHITE)
            complete_text = font.render(f"Simulation complete! {num_battles} battles logged.", True, BLACK)
            screen.blit(complete_text, (20, 50))
            
            file_text = small_font.render(f"Results saved to {self.logger.log_file}", True, BLACK)
            screen.blit(file_text, (20, 100))
            
            if collect_training_data:
                training_text = small_font.render(f"Training data saved to {self.training_logger.log_file}", True, BLACK)
                screen.blit(training_text, (20, 130))
            
            continue_text = small_font.render("Click anywhere or press any key to continue...", True, BLACK)
            screen.blit(continue_text, (20, 170))
            
            pygame.display.flip()
            
            # Wait for user input
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return completed_battles
                    if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
            
            pygame.quit()
        
        print(f"Batch simulation complete. {completed_battles} battles logged to {self.logger.log_file}")
        if collect_training_data:
            print(f"Neural network training data saved to {self.training_logger.log_file}")
            
        return completed_battles
