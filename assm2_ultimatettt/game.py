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
        self.network_mode = False
        self.peer = None
        self.network_username = None
        self.is_my_turn = False
        self.my_network_pause_count = 0
        
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
            'Model': lambda mark: ModelPlayer(mark, model_path='model.pt', temperature=0.5, use_mcts=False),
            'Model+MCTS': lambda mark: ModelPlayer(mark, model_path='model.pt', temperature=0.1, 
                                                  use_mcts=True, simulation_time=1.0, num_simulations=800, 
                                                  c_puct=1.5, top_k_moves=3)
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
        self.paused = False
        self.network_pause_deadline = None
        self.disconnected = False
        self.disconnect_reason = None
        # Rematch state
        self.rematch_requested = False
        self.showing_rematch_popup = False
        
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
    
    def start_game_network(self, peer, username, i_start_first: bool):
        """Start network human vs human game using PeerNetwork"""
        self.board = UltimateBoard()
        self.players['X'] = HumanPlayer('X')  # UI only; moves go via network
        self.players['O'] = HumanPlayer('O')
        self.player_x_type = 'Human'
        self.player_o_type = 'Human'
        self.start_time = time.time()
        self.move_count = 0
        self.agent_mode = False
        self.network_mode = True
        self.peer = peer
        self.network_username = username
        self.is_my_turn = i_start_first
        # Hide restart in pause menu for network mode
        self.ui.show_restart_in_pause = False
        self.ui.pause_countdown_seconds = None
        self.disconnected = False
        self.disconnect_reason = None
        self.rematch_requested = False
        self.showing_rematch_popup = False
        self.running = True
        
        # Set up disconnect callback
        def on_disconnect(reason):
            print(f"[DISCONNECT] Disconnect detected: {reason}")
            self.disconnected = True
            self.disconnect_reason = reason
            self.running = False
        
        if self.peer:
            self.peer.on_disconnect = on_disconnect
            # Reset rematch state
            self.peer.reset_rematch_state()
        
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
    
    def _restart_network_game(self):
        """Restart network game after rematch acceptance"""
        print(f"[REMATCH] Restarting network game")
        # Reset board
        self.board = UltimateBoard()
        
        # Reset game statistics
        self.start_time = time.time()
        self.move_count = 0
        
        # Reset rematch state
        self.rematch_requested = False
        self.ui.show_rematch_popup = False
        if self.peer:
            self.peer.reset_rematch_state()
        
        # Alternate who goes first (opposite of current)
        self.is_my_turn = not self.is_my_turn
        
        # Continue main loop (game will resume)
    
    def main_loop(self):
        """Main game loop"""
        while self.running:
            # Draw the board
            self.ui.draw_board(self.board)

            # If paused, handle pause menu and skip game updates
            if self.paused:
                # Update countdown if network mode
                if self.network_mode and self.network_pause_deadline:
                    remaining = max(0, int(self.network_pause_deadline - time.time()))
                    self.ui.pause_countdown_seconds = remaining
                    # Continuously check if timer expired or both players are ready
                    if self.peer:
                        # Update opponent ready indicator
                        if self.peer.opponent_ack:
                            self.ui.opponent_ready_text = "Opponent is ready to continue"
                        else:
                            self.ui.opponent_ready_text = None
                        
                        # Check if resume was triggered (either by both ready or timer)
                        # This handles the case where we trigger the resume
                        resume_triggered = self.peer.try_resume()
                        if resume_triggered:
                            # Resume was successful (we triggered it), unpause the game
                            print(f"[PAUSE] Resume triggered by this player")
                            self.paused = False
                            self.ui.is_showing_pause_menu = False
                            self.network_pause_deadline = None
                            self.ui.pause_countdown_seconds = None
                            self.ui.opponent_ready_text = None
                        # Also check if opponent sent RESUME message (pause_active becomes False)
                        # This handles the case where opponent triggered the resume
                        elif not self.peer.pause_active:
                            # Opponent sent RESUME, unpause the game
                            print(f"[PAUSE] Resume received from opponent")
                            self.paused = False
                            self.ui.is_showing_pause_menu = False
                            self.network_pause_deadline = None
                            self.ui.pause_countdown_seconds = None
                            self.ui.opponent_ready_text = None
                self._handle_pause_events()
                self.clock.tick(30)
                continue
            
            # Handle current player's move
            current_mark = self.board.current_player
            current_player = self.players[current_mark]
            
            # Special handling for network human vs human
            if self.network_mode:
                # Check for disconnection first
                if self.disconnected:
                    # Show disconnect message briefly before returning
                    self._show_disconnect_message()
                    return
                
                # Check if peer connection is lost
                if self.peer and not self.peer.is_connected:
                    self.disconnected = True
                    self.disconnect_reason = self.peer.game_status if isinstance(self.peer.game_status, str) else "Connection lost"
                    self._show_disconnect_message()
                    return
                
                # Poll peer status for incoming events
                if self.peer:
                    # If opponent requested pause, show pause overlay
                    if self.peer.pause_active and not self.paused:
                        self.paused = True
                        self.ui.is_showing_pause_menu = True
                        self.network_pause_deadline = self.peer.pause_deadline_ts
                        self.ui.opponent_ready_text = None
                    # Note: Resume check is now handled in the paused block above
                    # This check here is redundant but kept for safety
                    if not self.peer.pause_active and self.paused:
                        # Resume was received, unpause the game
                        self.paused = False
                        self.ui.is_showing_pause_menu = False
                        self.network_pause_deadline = None
                        self.ui.pause_countdown_seconds = None
                        self.ui.opponent_ready_text = None
                # Check for rematch acceptance
                # Both players restart when both have accepted (either by accepting opponent's request or having their request accepted)
                if self.peer and self.peer.rematch_accepted and (self.rematch_requested or self.peer.opponent_rematch_request):
                    # Both players ready, restart game
                    self._restart_network_game()
                    continue
                
                status = self.peer.get_game_status() if self.peer else None
                if isinstance(status, dict) and status.get('type') == 'MOVE':
                    # Apply opponent move
                    try:
                        self.board.make_move(status['main_row'], status['main_col'], status['sub_row'], status['sub_col'])
                        self.move_count += 1
                        self.is_my_turn = True
                        print(f"Applied opponent move: {status['main_row']},{status['main_col']},{status['sub_row']},{status['sub_col']}")
                        # Clear status after successful application
                        self.peer.game_status = None
                    except Exception as e:
                        print(f"Error applying opponent move: {e}")
                        import traceback
                        traceback.print_exc()
                        # Clear status so we don't retry invalid moves
                        self.peer.game_status = None
                # Pause updates are handled via callback updating flags/UI
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if self.peer:
                            self.peer.send_quit()
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.ui.is_pause_button_clicked(event.pos):
                            # Request network pause (max twice)
                            if self.peer and self.peer.request_pause(30):
                                self.paused = True
                                self.ui.is_showing_pause_menu = True
                                self.network_pause_deadline = self.peer.pause_deadline_ts
                            continue
                        pos = pygame.mouse.get_pos()
                        if not self.is_my_turn:
                            continue
                        cell = self.ui.get_cell_from_click(pos)
                        if cell is None:
                            continue
                        board_row, board_col, row, col = cell
                        # Try move legality via board
                        try:
                            self.board.make_move(board_row, board_col, row, col)
                            self.move_count += 1
                            self.is_my_turn = False
                            if self.peer:
                                self.peer.send_move(board_row, board_col, row, col)
                        except Exception:
                            pass
                # No AI thinking caption in network mode

            # Special handling for agent vs agent mode (local)
            elif self.agent_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.ui.is_pause_button_clicked(event.pos):
                            self.paused = True
                            self.ui.is_showing_pause_menu = True
                            continue
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:  # Restart game
                            self.restart_game()
                            continue
                        if event.key == pygame.K_q:  # Quit
                            pygame.quit()
                            sys.exit()
                
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
                
                # Handle game over (will return if rematch accepted in network mode)
                self._handle_game_over()
                
                # If we're here and it's network mode, check if we should continue (rematch)
                # Both players restart when both have accepted
                if self.network_mode and self.peer and self.peer.rematch_accepted and (self.rematch_requested or self.peer.opponent_rematch_request):
                    # Rematch was accepted, game already restarted in _handle_game_over
                    continue
                
            # Cap the frame rate
            self.clock.tick(30)
    
    def _handle_human_move(self):
        """Handle human player input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.network_mode and self.peer:
                    self.peer.send_quit()
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.ui.is_pause_button_clicked(event.pos):
                    if self.network_mode:
                        # Request network pause
                        if self.peer and self.peer.request_pause(30):
                            self.paused = True
                            self.ui.is_showing_pause_menu = True
                            self.network_pause_deadline = self.peer.pause_deadline_ts
                    else:
                        self.paused = True
                        self.ui.is_showing_pause_menu = True
                    return
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                pos = pygame.mouse.get_pos()

                # Handle game board click
                cell = self.ui.get_cell_from_click(pos)
                if cell is None:
                    return
                board_row, board_col, row, col = cell
                
                # Try to make the move
                self.board.make_move(board_row, board_col, row, col)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Restart game
                    if not self.network_mode:
                        self.restart_game()
                    return
                if event.key == pygame.K_q:  # Quit
                    if self.network_mode and self.peer:
                        self.peer.send_quit()
                    pygame.quit()
                    sys.exit()
    
    def _handle_game_over(self):
        """Handle game over state"""
        waiting = True
        # Cache the game over text surface to avoid re-rendering every frame
        game_over_text_surface = None
        game_over_text_rect = None
        last_text_state = None
        
        while waiting:
            # Check for network rematch requests
            if self.network_mode and self.peer:
                # Check if opponent requested rematch
                if self.peer.opponent_rematch_request and not self.ui.show_rematch_popup:
                    self.ui.show_rematch_popup = True
                    self.ui.rematch_opponent_name = self.peer.opponent_username or "Opponent"
                    # Invalidate cached text since state changed
                    game_over_text_surface = None
                
                # Check if rematch was accepted by both
                # Both players restart when both have accepted (either by accepting opponent's request or having their request accepted)
                if self.peer.rematch_accepted and (self.rematch_requested or self.peer.opponent_rematch_request):
                    # Both players ready, restart game
                    self._restart_network_game()
                    return
                
                # Check for disconnection
                if not self.peer.is_connected:
                    self.disconnected = True
                    self.disconnect_reason = self.peer.game_status if isinstance(self.peer.game_status, str) else "Connection lost"
                    waiting = False
                    return
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.network_mode and self.peer:
                        self.peer.send_quit()
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.ui.is_pause_button_clicked(event.pos):
                        # ignore pause during game-over; no-op
                        pass
                    # Handle rematch popup clicks
                    if self.ui.show_rematch_popup:
                        choice = self.ui.rematch_popup_button_from_pos(event.pos)
                        if choice == 'yes':
                            if self.peer:
                                self.peer.accept_rematch()
                                # Mark that we've accepted (either our request or opponent's)
                                # This ensures we restart when both are ready
                                self.rematch_requested = True  # Treat accepting as being ready for rematch
                                # Invalidate cached text since state changed
                                game_over_text_surface = None
                            self.ui.show_rematch_popup = False
                        elif choice == 'no':
                            if self.peer:
                                self.peer.decline_rematch()
                            waiting = False
                            return

                if event.type == pygame.KEYDOWN:
                    if self.network_mode:
                        # Network mode: R requests rematch, Q quits
                        if event.key == pygame.K_r:  # Request rematch
                            if self.peer and not self.rematch_requested:
                                self.peer.request_rematch()
                                self.rematch_requested = True
                                # Invalidate cached text since state changed
                                game_over_text_surface = None
                        elif event.key == pygame.K_q:  # Quit
                            if self.peer:
                                self.peer.send_quit()
                            waiting = False
                            return
                    else:
                        # Local mode: R restarts, Q quits
                        if event.key == pygame.K_r:  # Restart
                            self.restart_game()
                            waiting = False
                        elif event.key == pygame.K_q:  # Quit
                            pygame.quit()
                            sys.exit()
            
            # Draw the board (only once, cached)
            self.ui.draw_board(self.board)
            
            # Only re-render text if state changed
            current_text_state = (
                self.network_mode,
                self.rematch_requested,
                self.board.winner,
                self.board.is_draw
            )
            
            if game_over_text_surface is None or current_text_state != last_text_state:
                font = pygame.font.SysFont('Arial', 30)
                if self.network_mode:
                    if self.rematch_requested:
                        text_str = "Rematch requested. Waiting for opponent... Press Q to quit"
                        text_color = (255, 165, 0)
                    else:
                        if self.board.winner:
                            text_str = f"Game over! Player {self.board.winner} wins! Press R to request rematch or Q to quit"
                            text_color = (0, 255, 0)
                        else:
                            text_str = "Game over! It's a draw! Press R to request rematch or Q to quit"
                            text_color = (0, 0, 255)
                else:
                    if self.board.winner:
                        text_str = f"Game over! Player {self.board.winner} wins! Press R to restart or Q to quit"
                        text_color = (0, 255, 0)
                    else:
                        text_str = "Game over! It's a draw! Press R to restart or Q to quit"
                        text_color = (0, 0, 255)
                
                game_over_text_surface = font.render(text_str, True, text_color)
                game_over_text_rect = game_over_text_surface.get_rect(center=(self.ui.width // 2, self.ui.height - 30))
                last_text_state = current_text_state
            
            # Blit the cached text surface
            self.ui.screen.blit(game_over_text_surface, game_over_text_rect)
            
            pygame.display.flip()
            self.clock.tick(30)

    def _handle_pause_events(self):
        """Handle events while the pause menu is active"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.network_mode and self.peer:
                    self.peer.send_quit()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_p):
                    # Continue
                    if self.network_mode and self.peer:
                        self.peer.send_pause_ack()
                        # Resume might wait for both; try immediate resume if possible
                        self.peer.try_resume()
                    else:
                        self.paused = False
                        self.ui.is_showing_pause_menu = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                choice = self.ui.pause_menu_button_from_pos(event.pos)
                if choice == 'continue':
                    if self.network_mode and self.peer:
                        self.peer.send_pause_ack()
                        self.peer.try_resume()
                    else:
                        self.paused = False
                        self.ui.is_showing_pause_menu = False
                elif choice == 'restart':
                    if not self.network_mode:
                        self.restart_game()
                        self.paused = False
                        self.ui.is_showing_pause_menu = False
                elif choice == 'quit':
                    # Return to main menu
                    if self.network_mode and self.peer:
                        self.peer.send_quit()
                    self.paused = False
                    self.ui.is_showing_pause_menu = False
                    self.running = False
    
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
    
    def run_benchmark_test(self, num_games=20):
        """Run a benchmark comparing standard Model vs Model+MCTS"""
        print("\nRunning benchmark: Model vs Model+MCTS")
        print("====================================")
        
        # Create players
        model_player = ModelPlayer('X', model_path='model.pt', temperature=0.1, use_mcts=False)
        model_mcts_player = ModelPlayer('O', model_path='model.pt', temperature=0.1, 
                                     use_mcts=True, simulation_time=1.0, num_simulations=800)
        
        results = {'Model': 0, 'Model+MCTS': 0, 'Draw': 0}
        move_times = {'Model': [], 'Model+MCTS': []}
        
        # Run games
        for i in range(num_games):
            print(f"Game {i+1}/{num_games}...")
            
            # Reset board
            board = UltimateBoard()
            turn_count = 0
            
            # Alternate who goes first
            if i % 2 == 0:
                players = {'X': model_player, 'O': model_mcts_player}
                player_types = {'X': 'Model', 'O': 'Model+MCTS'}
            else:
                players = {'X': model_mcts_player, 'O': model_player}
                player_types = {'X': 'Model+MCTS', 'O': 'Model'}
                
                # Update player marks for correct evaluation
                model_player.mark = 'O'
                model_mcts_player.mark = 'X'
            
            # Play the game
            while board.winner is None and not board.is_draw:
                current_mark = board.current_player
                current_player = players[current_mark]
                current_type = player_types[current_mark]
                
                # Time the move decision
                start_time = time.time()
                move = current_player.get_move(board)
                end_time = time.time()
                
                # Record move time
                move_times[current_type].append(end_time - start_time)
                
                # Make the move
                if move:
                    board_row, board_col, row, col = move
                    board.make_move(board_row, board_col, row, col)
                    turn_count += 1
            
            # Record result
            if board.winner:
                winner_type = player_types[board.winner]
                results[winner_type] += 1
                print(f"  Winner: {winner_type} in {turn_count} moves")
            else:
                results['Draw'] += 1
                print(f"  Draw after {turn_count} moves")
            
            # Reset player marks for next game
            model_player.mark = 'X' 
            model_mcts_player.mark = 'O'
                
        # Display results
        print("\nBenchmark Results:")
        print(f"Total games: {num_games}")
        print(f"Model wins: {results['Model']} ({results['Model']/num_games*100:.1f}%)")
        print(f"Model+MCTS wins: {results['Model+MCTS']} ({results['Model+MCTS']/num_games*100:.1f}%)")
        print(f"Draws: {results['Draw']} ({results['Draw']/num_games*100:.1f}%)")
        
        print("\nAverage move time:")
        print(f"Model: {sum(move_times['Model'])/len(move_times['Model']):.3f} seconds")
        print(f"Model+MCTS: {sum(move_times['Model+MCTS'])/len(move_times['Model+MCTS']):.3f} seconds")
        
        return results
    
    def _show_disconnect_message(self):
        """Show disconnect message briefly before returning to menu"""
        if not self.disconnect_reason:
            self.disconnect_reason = "Connection lost"
        
        # Show message for 2 seconds
        message_duration = 2.0
        start_time = time.time()
        
        font = pygame.font.SysFont('Arial', 36)
        small_font = pygame.font.SysFont('Arial', 24)
        
        while time.time() - start_time < message_duration:
            self.ui.screen.fill(self.ui.WHITE)
            
            # Draw message
            main_text = font.render(self.disconnect_reason, True, self.ui.RED)
            sub_text = small_font.render("Returning to main menu...", True, self.ui.BLACK)
            
            main_rect = main_text.get_rect(center=(self.ui.width // 2, self.ui.height // 2 - 30))
            sub_rect = sub_text.get_rect(center=(self.ui.width // 2, self.ui.height // 2 + 30))
            
            self.ui.screen.blit(main_text, main_rect)
            self.ui.screen.blit(sub_text, sub_rect)
            
            pygame.display.flip()
            
            # Handle events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            self.clock.tick(30)
