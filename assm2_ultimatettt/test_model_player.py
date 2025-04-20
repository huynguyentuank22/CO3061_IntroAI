from player import RandomPlayer, MinimaxPlayer, MCTSPlayer, ModelPlayer
from ultimate_board import UltimateTicTacToeBoard  # Assuming you have this class
import time
import random

def play_game(board, player_x, player_o, verbose=False):
    """Play a single game between two players and return the winner"""
    players = {'X': player_x, 'O': player_o}
    current_mark = 'X'
    
    while board.winner is None and not board.is_draw:
        if verbose:
            print(f"\n{current_mark}'s turn:")
            print(board)
        
        start_time = time.time()
        move = players[current_mark].get_move(board)
        end_time = time.time()
        
        if move is None:
            break
            
        if verbose:
            print(f"{current_mark} plays at board ({move[0]},{move[1]}) cell ({move[2]},{move[3]}) in {end_time-start_time:.3f}s")
            
        board_row, board_col, row, col = move
        board.make_move(board_row, board_col, row, col)
        
        # Switch player
        current_mark = 'O' if current_mark == 'X' else 'X'
    
    if verbose:
        print("\nFinal board:")
        print(board)
        if board.winner:
            print(f"{board.winner} wins!")
        else:
            print("It's a draw!")
            
    return board.winner

def play_tournament(players_x, players_o, num_games=100, verbose_last=False):
    """Play a tournament between multiple players and display statistics"""
    results = {player_x.mark + '_' + player_x.__class__.__name__: {
                player_o.mark + '_' + player_o.__class__.__name__: {'wins': 0, 'losses': 0, 'draws': 0}
                for player_o in players_o} 
              for player_x in players_x}
    
    game_count = 0
    total_games = len(players_x) * len(players_o) * num_games
    
    for i, player_x in enumerate(players_x):
        for j, player_o in enumerate(players_o):
            player_x_name = player_x.mark + '_' + player_x.__class__.__name__
            player_o_name = player_o.mark + '_' + player_o.__class__.__name__
            
            print(f"Playing {num_games} games: {player_x_name} vs {player_o_name}")
            
            for game in range(num_games):
                game_count += 1
                verbose = verbose_last and game == num_games - 1
                
                # Create a new board for each game
                board = UltimateTicTacToeBoard()
                
                # Play the game
                winner = play_game(board, player_x, player_o, verbose)
                
                # Record results
                if winner == 'X':
                    results[player_x_name][player_o_name]['wins'] += 1
                    print('.', end='', flush=True)
                elif winner == 'O':
                    results[player_x_name][player_o_name]['losses'] += 1
                    print('x', end='', flush=True)
                else:  # Draw
                    results[player_x_name][player_o_name]['draws'] += 1
                    print('d', end='', flush=True)
                
                # Print progress
                if game % 10 == 9:
                    print(f" [{game_count}/{total_games}]")
            
            print()  # New line after each player pair
    
    # Print full results
    print("\nTournament Results:")
    for player_x_name in results:
        print(f"\n{player_x_name} as X:")
        for player_o_name, stats in results[player_x_name].items():
            total = stats['wins'] + stats['losses'] + stats['draws']
            if total > 0:
                win_rate = stats['wins'] / total * 100
                print(f"  vs {player_o_name}: W: {stats['wins']}, L: {stats['losses']}, D: {stats['draws']} - Win rate: {win_rate:.1f}%")
    
    return results

if __name__ == "__main__":
    # Create players
    model_player = ModelPlayer('X', model_path='model.pt', temperature=0.5)
    random_player = RandomPlayer('O')
    minimax_player = MinimaxPlayer('O', depth=2)  # Lower depth for faster play
    mcts_player = MCTSPlayer('O', simulation_time=1.0)
    
    # Play tournament
    players_x = [model_player]
    players_o = [random_player, minimax_player, mcts_player]
    
    results = play_tournament(players_x, players_o, num_games=10, verbose_last=True)
    
    # You could also test your model as O player
    model_player_o = ModelPlayer('O', model_path='model.pt', temperature=0.5)
    random_player_x = RandomPlayer('X')
    minimax_player_x = MinimaxPlayer('X', depth=2)
    
    players_x = [random_player_x, minimax_player_x]
    players_o = [model_player_o]
    
    results_as_o = play_tournament(players_x, players_o, num_games=10, verbose_last=True)
