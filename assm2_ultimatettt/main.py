import pygame
import sys
import time
from game import Game
from peer import PeerNetwork

def main():
    """Main function to start the game"""
    # Initialize Pygame
    pygame.init()
    
    # Create menu screen
    width, height = 900, 900
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Ultimate Tic Tac Toe - Menu")
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    LIGHT_BLUE = (173, 216, 230)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    
    # Fonts
    title_font = pygame.font.SysFont('Arial', 48)
    button_font = pygame.font.SysFont('Arial', 36)
    small_font = pygame.font.SysFont('Arial', 24)
    
    # AI options - added Model option
    ai_options = ['Random', 'Easy', 'Medium', 'Hard', 'MCTS', 'Model']
    
    # Create game instance
    game = Game()
    
    # Main menu options - add Network Human vs Human and Generate Training Data
    main_options = ['Human vs AI', 'Agent vs Agent', 'Human vs Human (Network)', 'Batch Simulation', 'Generate Training Data']
    
    # Menu state
    menu_state = 'main'  # 'main', 'human_vs_ai', 'agent_vs_agent', 'network_name', 'network_lobby', 'batch_simulation', 'training_data', 'disconnect_message'
    first_agent = None  # For agent vs agent mode
    batch_size = 100    # Default batch simulation size
    training_size = 50 # Default training data generation size
    disconnect_message = None
    disconnect_start_time = None
    disconnect_display_duration = 3.0

    # Network UI state
    username_input = ""
    username_active = False
    peer = None
    is_broadcasting = False
    network_buttons = []  # Accept buttons
    
    # Menu loop
    running = True
    while running:
        screen.fill(WHITE)
        
        # Draw title
        title = title_font.render("Ultimate Tic Tac Toe", True, BLACK)
        screen.blit(title, (width // 2 - title.get_width() // 2, 100))
        
        # Handle different menu states
        if menu_state == 'main':
            subtitle = button_font.render("Select Game Mode:", True, BLACK)
            screen.blit(subtitle, (width // 2 - subtitle.get_width() // 2, 200))
            
            # Draw buttons for main options
            button_height = 60
            button_width = 400
            button_margin = 20
            button_y = 300
            
            buttons = []
            for option in main_options:
                button_rect = pygame.Rect(width // 2 - button_width // 2, 
                                         button_y, 
                                         button_width, 
                                         button_height)
                
                # Check if mouse is over button
                mouse_pos = pygame.mouse.get_pos()
                if button_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, LIGHT_BLUE, button_rect)
                else:
                    pygame.draw.rect(screen, GRAY, button_rect)
                
                pygame.draw.rect(screen, BLACK, button_rect, 2)  # Button border
                
                # Button text
                text = button_font.render(option, True, BLACK)
                screen.blit(text, (button_rect.centerx - text.get_width() // 2, 
                                 button_rect.centery - text.get_height() // 2))
                
                buttons.append((button_rect, option))
                button_y += button_height + button_margin
                
        elif menu_state == 'human_vs_ai':
            subtitle = button_font.render("Select AI Opponent:", True, BLACK)
            screen.blit(subtitle, (width // 2 - subtitle.get_width() // 2, 200))
            
            # Draw buttons for AI options
            button_height = 60
            button_width = 300
            button_margin = 20
            button_y = 300
            
            buttons = []
            for option in ai_options:
                button_rect = pygame.Rect(width // 2 - button_width // 2, 
                                         button_y, 
                                         button_width, 
                                         button_height)
                
                # Check if mouse is over button
                mouse_pos = pygame.mouse.get_pos()
                if button_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, LIGHT_BLUE, button_rect)
                else:
                    pygame.draw.rect(screen, GRAY, button_rect)
                
                pygame.draw.rect(screen, BLACK, button_rect, 2)  # Button border
                
                # Button text
                text = button_font.render(option, True, BLACK)
                screen.blit(text, (button_rect.centerx - text.get_width() // 2, 
                                 button_rect.centery - text.get_height() // 2))
                
                buttons.append((button_rect, option))
                button_y += button_height + button_margin
                
        elif menu_state == 'agent_vs_agent':
            if first_agent is None:
                subtitle = button_font.render("Select Agent X (First Player):", True, BLACK)
            else:
                subtitle = button_font.render("Select Agent O (Second Player):", True, BLACK)
            
            screen.blit(subtitle, (width // 2 - subtitle.get_width() // 2, 200))
            
            # Draw buttons for AI options
            button_height = 60
            button_width = 300
            button_margin = 20
            button_y = 300
            
            buttons = []
            for option in ai_options:
                button_rect = pygame.Rect(width // 2 - button_width // 2, 
                                         button_y, 
                                         button_width, 
                                         button_height)
                
                # Check if mouse is over button
                mouse_pos = pygame.mouse.get_pos()
                if button_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, LIGHT_BLUE, button_rect)
                else:
                    pygame.draw.rect(screen, GRAY, button_rect)
                
                pygame.draw.rect(screen, BLACK, button_rect, 2)  # Button border
                
                # Button text
                text = button_font.render(option, True, BLACK)
                screen.blit(text, (button_rect.centerx - text.get_width() // 2, 
                                 button_rect.centery - text.get_height() // 2))
                
                buttons.append((button_rect, option))
                button_y += button_height + button_margin
                
        elif menu_state == 'network_name':
            subtitle = button_font.render("Enter your name:", True, BLACK)
            screen.blit(subtitle, (width // 2 - subtitle.get_width() // 2, 220))

            # Input box
            input_rect = pygame.Rect(width // 2 - 200, 280, 400, 50)
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.rect(screen, LIGHT_BLUE if input_rect.collidepoint(mouse_pos) or username_active else GRAY, input_rect)
            pygame.draw.rect(screen, BLACK, input_rect, 2)

            name_surface = button_font.render(username_input or "Player", True, BLACK)
            screen.blit(name_surface, (input_rect.x + 10, input_rect.y + 10))

            # Continue and Back buttons
            cont_rect = pygame.Rect(width // 2 - 150, 360, 300, 60)
            back_rect = pygame.Rect(width // 2 - 150, 430, 300, 60)
            for rect, label in [(cont_rect, "Continue"), (back_rect, "Back")]:
                if rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, LIGHT_BLUE, rect)
                else:
                    pygame.draw.rect(screen, GRAY, rect)
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = button_font.render(label, True, BLACK)
                screen.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))

            buttons = [(cont_rect, "network_continue"), (back_rect, "back")]

        elif menu_state == 'network_lobby':
            subtitle = button_font.render(f"Network Lobby - {username_input}", True, BLACK)
            screen.blit(subtitle, (width // 2 - subtitle.get_width() // 2, 180))

            # Ensure peer network created and UDP listener started
            if peer is None:
                try:
                    peer = PeerNetwork(username_input)
                    if not peer.initialize_udp_socket():
                        error_text = small_font.render("Failed to initialize network. Check firewall settings.", True, (255, 0, 0))
                        screen.blit(error_text, (width // 2 - error_text.get_width() // 2, 750))
                    else:
                        # Start UDP listener thread
                        if not peer.start_udp_listener():
                            error_text = small_font.render("Failed to start network listener.", True, (255, 0, 0))
                            screen.blit(error_text, (width // 2 - error_text.get_width() // 2, 750))
                    # Initialize TCP socket for accepting connections
                    peer.initialize_tcp_socket()
                except Exception as e:
                    print(f"Error creating peer network: {e}")
                    import traceback
                    traceback.print_exc()
                    peer = None
                    error_text = small_font.render(f"Network error: {str(e)}", True, (255, 0, 0))
                    screen.blit(error_text, (width // 2 - error_text.get_width() // 2, 750))

            # Search/Stop button
            button_height = 50
            search_rect = pygame.Rect(width // 2 - 320, 230, 260, button_height)
            listen_rect = pygame.Rect(width // 2 + 60, 230, 260, button_height)
            mouse_pos = pygame.mouse.get_pos()
            for rect, label in [(search_rect, "Stop Searching" if is_broadcasting else "Search for Opponent"),
                                (listen_rect, "Back to Menu")]:
                if rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, LIGHT_BLUE, rect)
                else:
                    pygame.draw.rect(screen, GRAY, rect)
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = small_font.render(label, True, BLACK)
                screen.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))

            # Status text
            if peer:
                if is_broadcasting:
                    status_text = small_font.render("Status: Searching for opponents...", True, GREEN)
                elif peer.is_connected:
                    status_text = small_font.render("Status: Connected! Starting game...", True, GREEN)
                else:
                    status_text = small_font.render("Status: Listening for requests...", True, BLACK)
                screen.blit(status_text, (width // 2 - status_text.get_width() // 2, 290))

            # Pending requests list
            list_title = small_font.render("Incoming Requests:", True, BLACK)
            screen.blit(list_title, (width // 2 - 150, 310))
            network_buttons = []
            y = 340
            requests = peer.get_pending_requests() if peer else []
            for req in requests[:6]:
                row_rect = pygame.Rect(width // 2 - 300, y, 600, 40)
                pygame.draw.rect(screen, (245, 245, 245), row_rect)
                pygame.draw.rect(screen, BLACK, row_rect, 1)
                label = small_font.render(f"{req['username']}  (signal {req['strength']})", True, BLACK)
                screen.blit(label, (row_rect.x + 10, row_rect.y + 8))
                accept_rect = pygame.Rect(row_rect.right - 110, row_rect.y + 5, 100, 30)
                if accept_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, LIGHT_BLUE, accept_rect)
                else:
                    pygame.draw.rect(screen, GRAY, accept_rect)
                pygame.draw.rect(screen, BLACK, accept_rect, 1)
                accept_text = small_font.render("Accept", True, BLACK)
                screen.blit(accept_text, (accept_rect.centerx - accept_text.get_width() // 2,
                                          accept_rect.centery - accept_text.get_height() // 2))
                network_buttons.append((accept_rect, ('accept', req['username'])))
                y += 50

            # Check for game start when connected
            if peer and peer.is_connected:
                # Check for GAME_START message
                status = peer.get_game_status()
                if isinstance(status, dict) and status.get('type') == 'GAME_START':
                    first_player = status.get('first_player')
                    i_start_first = (first_player == username_input)
                    print(f"Starting network game, first player: {first_player}, I start first: {i_start_first}")
                    # Clear the game status so we don't start multiple times
                    peer.game_status = None
                    pygame.display.set_caption("Ultimate Tic Tac Toe - Network Game")
                    game.start_game_network(peer, username_input, i_start_first)
                    pygame.display.set_caption("Ultimate Tic Tac Toe - Menu")
                    # After game returns
                    is_broadcasting = False
                    
                    # Check if game ended due to disconnection
                    disconnect_message = None
                    if game.disconnected:
                        disconnect_message = game.disconnect_reason or "Opponent has left the game"
                        print(f"Game ended due to disconnection: {disconnect_message}")
                    
                    # Clean up peer
                    if peer:
                        peer.stop_broadcasting()
                        if peer.peer_connection:
                            try:
                                peer.peer_connection.close()
                            except:
                                pass
                        # Clean up TCP socket
                        if peer.tcp_socket:
                            try:
                                peer.tcp_socket.close()
                            except:
                                pass
                        # Clean up UDP socket
                        if peer.udp_socket:
                            try:
                                peer.udp_socket.close()
                            except:
                                pass
                    peer = None
                    
                    # Show disconnect message if applicable
                    if disconnect_message:
                        menu_state = 'disconnect_message'
                        disconnect_start_time = time.time()
                        disconnect_display_duration = 3.0  # Show for 3 seconds
                    else:
                        menu_state = 'main'
                # Debug: show connection status
                elif peer.is_connected:
                    # Show waiting message if connected but no GAME_START yet
                    if not hasattr(peer, '_waiting_shown'):
                        print(f"Connected to {peer.opponent_username}, waiting for GAME_START...")
                        peer._waiting_shown = True

            buttons = [(search_rect, 'toggle_search'), (listen_rect, 'back')]

        elif menu_state == 'disconnect_message':
            # Show disconnect message screen
            if disconnect_message:
                # Check if display duration has passed
                if disconnect_start_time and time.time() - disconnect_start_time >= disconnect_display_duration:
                    menu_state = 'main'
                    disconnect_message = None
                    disconnect_start_time = None
                else:
                    # Draw disconnect message
                    message_text = button_font.render(disconnect_message, True, RED)
                    screen.blit(message_text, (width // 2 - message_text.get_width() // 2, height // 2 - 50))
                    
                    return_text = small_font.render("Returning to main menu...", True, BLACK)
                    screen.blit(return_text, (width // 2 - return_text.get_width() // 2, height // 2 + 20))
            else:
                menu_state = 'main'

        elif menu_state == 'batch_simulation':
            subtitle = button_font.render("Batch Simulation Settings", True, BLACK)
            screen.blit(subtitle, (width // 2 - subtitle.get_width() // 2, 200))
            
            # Show current batch size
            size_text = button_font.render(f"Number of battles: {batch_size}", True, BLACK)
            screen.blit(size_text, (width // 2 - size_text.get_width() // 2, 280))
            
            # Buttons to adjust batch size
            button_height = 50
            button_width = 50
            button_margin = 20
            
            # Decrease button
            decrease_rect = pygame.Rect(width // 2 - 100, 350, button_width, button_height)
            mouse_pos = pygame.mouse.get_pos()
            if decrease_rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, LIGHT_BLUE, decrease_rect)
            else:
                pygame.draw.rect(screen, GRAY, decrease_rect)
            
            pygame.draw.rect(screen, BLACK, decrease_rect, 2)
            decrease_text = button_font.render("-", True, BLACK)
            screen.blit(decrease_text, (decrease_rect.centerx - decrease_text.get_width() // 2, 
                                      decrease_rect.centery - decrease_text.get_height() // 2))
            
            # Increase button
            increase_rect = pygame.Rect(width // 2 + 50, 350, button_width, button_height)
            if increase_rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, LIGHT_BLUE, increase_rect)
            else:
                pygame.draw.rect(screen, GRAY, increase_rect)
            
            pygame.draw.rect(screen, BLACK, increase_rect, 2)
            increase_text = button_font.render("+", True, BLACK)
            screen.blit(increase_text, (increase_rect.centerx - increase_text.get_width() // 2, 
                                      increase_rect.centery - increase_text.get_height() // 2))
            
            # Start and back buttons
            start_rect = pygame.Rect(width // 2 - 150, 450, 300, button_height)
            if start_rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, LIGHT_BLUE, start_rect)
            else:
                pygame.draw.rect(screen, GRAY, start_rect)
            
            pygame.draw.rect(screen, BLACK, start_rect, 2)
            start_text = button_font.render("Start Simulation", True, BLACK)
            screen.blit(start_text, (start_rect.centerx - start_text.get_width() // 2, 
                                   start_rect.centery - start_text.get_height() // 2))
            
            back_rect = pygame.Rect(width // 2 - 150, 520, 300, button_height)
            if back_rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, LIGHT_BLUE, back_rect)
            else:
                pygame.draw.rect(screen, GRAY, back_rect)
            
            pygame.draw.rect(screen, BLACK, back_rect, 2)
            back_text = button_font.render("Back", True, BLACK)
            screen.blit(back_text, (back_rect.centerx - back_text.get_width() // 2, 
                                  back_rect.centery - back_text.get_height() // 2))
            
            buttons = [(decrease_rect, "decrease"), (increase_rect, "increase"), 
                      (start_rect, "start"), (back_rect, "back")]
                
        elif menu_state == 'training_data':
            subtitle = button_font.render("Neural Network Training Data", True, BLACK)
            screen.blit(subtitle, (width // 2 - subtitle.get_width() // 2, 200))
            
            # Show current training size
            size_text = button_font.render(f"Number of games: {training_size}", True, BLACK)
            screen.blit(size_text, (width // 2 - size_text.get_width() // 2, 280))
            
            # Buttons to adjust training size
            button_height = 50
            button_width = 50
            button_margin = 20
            
            # Decrease button
            decrease_rect = pygame.Rect(width // 2 - 100, 350, button_width, button_height)
            mouse_pos = pygame.mouse.get_pos()
            if decrease_rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, LIGHT_BLUE, decrease_rect)
            else:
                pygame.draw.rect(screen, GRAY, decrease_rect)
            
            pygame.draw.rect(screen, BLACK, decrease_rect, 2)
            decrease_text = button_font.render("-", True, BLACK)
            screen.blit(decrease_text, (decrease_rect.centerx - decrease_text.get_width() // 2, 
                                      decrease_rect.centery - decrease_text.get_height() // 2))
            
            # Increase button
            increase_rect = pygame.Rect(width // 2 + 50, 350, button_width, button_height)
            if increase_rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, LIGHT_BLUE, increase_rect)
            else:
                pygame.draw.rect(screen, GRAY, increase_rect)
            
            pygame.draw.rect(screen, BLACK, increase_rect, 2)
            increase_text = button_font.render("+", True, BLACK)
            screen.blit(increase_text, (increase_rect.centerx - increase_text.get_width() // 2, 
                                      increase_rect.centery - increase_text.get_height() // 2))
            
            # Info text
            info_text = small_font.render("Each game generates multiple training examples", True, BLACK)
            screen.blit(info_text, (width // 2 - info_text.get_width() // 2, 420))
            
            # Start and back buttons
            start_rect = pygame.Rect(width // 2 - 150, 470, 300, button_height)
            if start_rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, LIGHT_BLUE, start_rect)
            else:
                pygame.draw.rect(screen, GRAY, start_rect)
            
            pygame.draw.rect(screen, BLACK, start_rect, 2)
            start_text = button_font.render("Generate Data", True, BLACK)
            screen.blit(start_text, (start_rect.centerx - start_text.get_width() // 2, 
                                   start_rect.centery - start_text.get_height() // 2))
            
            back_rect = pygame.Rect(width // 2 - 150, 540, 300, button_height)
            if back_rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, LIGHT_BLUE, back_rect)
            else:
                pygame.draw.rect(screen, GRAY, back_rect)
            
            pygame.draw.rect(screen, BLACK, back_rect, 2)
            back_text = button_font.render("Back", True, BLACK)
            screen.blit(back_text, (back_rect.centerx - back_text.get_width() // 2, 
                                  back_rect.centery - back_text.get_height() // 2))
            
            # Output file info
            output_text = small_font.render("Output will be saved to: train.csv", True, BLACK)
            screen.blit(output_text, (width // 2 - output_text.get_width() // 2, 610))
            
            buttons = [(decrease_rect, "decrease"), (increase_rect, "increase"), 
                      (start_rect, "start"), (back_rect, "back")]
        
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if menu_state == 'network_name':
                    # Activate input box
                    input_rect = pygame.Rect(width // 2 - 200, 280, 400, 50)
                    username_active = input_rect.collidepoint(event.pos)
                
                # Handle Accept button clicks FIRST (before main button loop)
                if menu_state == 'network_lobby':
                    for arect, payload in network_buttons:
                        if arect.collidepoint(event.pos):
                            kind, uname = payload
                            if kind == 'accept' and peer and not peer.is_connected:
                                print(f"Accepting connection from {uname}")
                                if peer.accept_connection(uname):
                                    print(f"Successfully accepted connection from {uname}")
                                else:
                                    print(f"Failed to accept connection from {uname}")
                                # Break to avoid checking other buttons
                                break
                
                # Main button handling
                for button, option in buttons:
                    if button.collidepoint(event.pos):
                        if menu_state == 'main':
                            if option == 'Human vs AI':
                                menu_state = 'human_vs_ai'
                            elif option == 'Agent vs Agent':
                                menu_state = 'agent_vs_agent'
                                first_agent = None
                            elif option == 'Human vs Human (Network)':
                                menu_state = 'network_name'
                            elif option == 'Batch Simulation':
                                menu_state = 'batch_simulation'
                            elif option == 'Generate Training Data':
                                menu_state = 'training_data'
                                
                        elif menu_state == 'network_name':
                            if option == 'network_continue':
                                if not username_input.strip():
                                    username_input = "Player"
                                # Go to lobby
                                menu_state = 'network_lobby'
                            elif option == 'back':
                                menu_state = 'main'
                                username_input = ""
                                username_active = False

                        elif menu_state == 'network_lobby':
                            if option == 'toggle_search':
                                if peer:
                                    if is_broadcasting:
                                        peer.stop_broadcasting()
                                        is_broadcasting = False
                                    else:
                                        peer.broadcast_connect_request()
                                        is_broadcasting = True
                            elif option == 'back':
                                if peer:
                                    peer.stop_broadcasting()
                                peer = None
                                is_broadcasting = False
                                menu_state = 'main'

                        elif menu_state == 'human_vs_ai':
                            # Start game with human vs selected AI
                            game.start_game(option)
                            # After game ends, we'll return to menu
                            pygame.display.set_caption("Ultimate Tic Tac Toe - Menu")
                            menu_state = 'main'
                            
                        elif menu_state == 'agent_vs_agent':
                            if first_agent is None:
                                first_agent = option
                            else:
                                # Start game with two AI agents
                                game.start_game_agents(first_agent, option)
                                # After game ends, return to menu
                                pygame.display.set_caption("Ultimate Tic Tac Toe - Menu")
                                menu_state = 'main'
                                first_agent = None
                                
                        elif menu_state == 'batch_simulation':
                            if option == "decrease":
                                batch_size = max(10, batch_size - 10)  # Minimum 10 battles
                            elif option == "increase":
                                batch_size = min(500, batch_size + 10)  # Maximum 500 battles
                            elif option == "start":
                                # Close pygame window temporarily
                                pygame.quit()
                                
                                # Run simulation
                                game.run_batch_simulation(batch_size)
                                
                                # Re-initialize pygame for menu
                                pygame.init()
                                screen = pygame.display.set_mode((width, height))
                                pygame.display.set_caption("Ultimate Tic Tac Toe - Menu")
                                menu_state = 'main'
                            elif option == "back":
                                menu_state = 'main'
                                
                        elif menu_state == 'training_data':
                            if option == "decrease":
                                training_size = max(10, training_size - 10)  # Minimum 100 games
                            elif option == "increase":
                                training_size = min(10000, training_size + 10)  # Maximum 10000 games
                            elif option == "start":
                                # Close pygame window temporarily
                                pygame.quit()
                                
                                # Run simulation with training data collection
                                game.run_batch_simulation(training_size, collect_training_data=True)
                                
                                # Re-initialize pygame for menu
                                pygame.init()
                                screen = pygame.display.set_mode((width, height))
                                pygame.display.set_caption("Ultimate Tic Tac Toe - Menu")
                                menu_state = 'main'
                            elif option == "back":
                                menu_state = 'main'
                            
            if event.type == pygame.KEYDOWN:
                if menu_state == 'network_name' and username_active:
                    if event.key == pygame.K_RETURN:
                        username_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        username_input = username_input[:-1]
                    else:
                        # Basic text input
                        ch = event.unicode
                        if ch.isprintable() and len(username_input) < 20:
                            username_input += ch

        pygame.time.Clock().tick(30)

if __name__ == "__main__":
    main()
