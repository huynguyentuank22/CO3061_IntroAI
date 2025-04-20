import pygame
import sys
from game import Game

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
    
    # Fonts
    title_font = pygame.font.SysFont('Arial', 48)
    button_font = pygame.font.SysFont('Arial', 36)
    small_font = pygame.font.SysFont('Arial', 24)
    
    # AI options - added Model option
    ai_options = ['Random', 'Easy', 'Medium', 'Hard', 'MCTS', 'Model']
    
    # Create game instance
    game = Game()
    
    # Main menu options - add Generate Training Data
    main_options = ['Human vs AI', 'Agent vs Agent', 'Batch Simulation', 'Generate Training Data']
    
    # Menu state
    menu_state = 'main'  # 'main', 'human_vs_ai', 'agent_vs_agent', 'batch_simulation', 'training_data'
    first_agent = None  # For agent vs agent mode
    batch_size = 100    # Default batch simulation size
    training_size = 50 # Default training data generation size
    
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
            button_width = 300
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
                for button, option in buttons:
                    if button.collidepoint(event.pos):
                        if menu_state == 'main':
                            if option == 'Human vs AI':
                                menu_state = 'human_vs_ai'
                            elif option == 'Agent vs Agent':
                                menu_state = 'agent_vs_agent'
                                first_agent = None
                            elif option == 'Batch Simulation':
                                menu_state = 'batch_simulation'
                            elif option == 'Generate Training Data':
                                menu_state = 'training_data'
                                
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
                            
        pygame.time.Clock().tick(30)

if __name__ == "__main__":
    main()
