import time
import random
from copy import deepcopy
import sys
from ultimate_board import UltimateBoard
from player import ModelPlayer, RandomPlayer, MinimaxPlayer, MCTSPlayer

def test_model_vs_model_mcts(num_games=5):
    """Test the performance of Model vs Model+MCTS"""
    print("Testing Model vs Model+MCTS")
    print("==========================")
    
    # Create players
    model_player = ModelPlayer('X', model_path='model.pt', temperature=0.1, use_mcts=False)
    model_mcts_player = ModelPlayer('O', model_path='model.pt', temperature=0.1, 
                                 use_mcts=True, simulation_time=1.0, num_simulations=400)
    
    results = {'Model': 0, 'Model+MCTS': 0, 'Draw': 0}
    move_counts = []
    
    for i in range(num_games):
        print(f"\nGame {i+1}/{num_games}")
        
        # Reset board
        board = UltimateBoard()
        moves_made = 0
        
        # Alternate who goes first
        if i % 2 == 0:
            players = {'X': model_player, 'O': model_mcts_player}
            labels = {'X': 'Model', 'O': 'Model+MCTS'}
            model_player.mark = 'X'
            model_mcts_player.mark = 'O'
        else:
            players = {'X': model_mcts_player, 'O': model_player}
            labels = {'X': 'Model+MCTS', 'O': 'Model'}
            model_player.mark = 'O'
            model_mcts_player.mark = 'X'
        
        print(f"{labels['X']} (X) vs {labels['O']} (O)")
        
        # Play the game
        while board.winner is None and not board.is_draw:
            current_mark = board.current_player
            current_player = players[current_mark]
            
            print(f"\nPlayer {current_mark} ({labels[current_mark]}) thinking...")
            start_time = time.time()
            move = current_player.get_move(board)
            end_time = time.time()
            
            print(f"Player {current_mark} ({labels[current_mark]}) chose move: {move} (took {end_time-start_time:.2f}s)")
            
            # Make move
            if move:
                board_row, board_col, row, col = move
                board.make_move(board_row, board_col, row, col)
                moves_made += 1
            
            # Print board state (compact representation)
            print_compact_board(board)
        
        # Record result
        if board.winner:
            winner_type = labels[board.winner]
            results[winner_type] += 1
            print(f"\nGame {i+1} Result: {winner_type} wins in {moves_made} moves")
        else:
            results['Draw'] += 1
            print(f"\nGame {i+1} Result: Draw after {moves_made} moves")
        
        move_counts.append(moves_made)
    
    # Print overall results
    print("\nFinal Results:")
    print(f"Model wins: {results['Model']} ({results['Model']/num_games*100:.1f}%)")
    print(f"Model+MCTS wins: {results['Model+MCTS']} ({results['Model+MCTS']/num_games*100:.1f}%)")
    print(f"Draws: {results['Draw']} ({results['Draw']/num_games*100:.1f}%)")
    print(f"Average game length: {sum(move_counts)/len(move_counts):.1f} moves")
    
    return results

def test_model_mcts_vs_simple_opponents(num_games=5):
    """Test the performance of Model+MCTS against Random and Easy opponents"""
    
    print("\nTesting Model+MCTS vs Simple Opponents")
    print("=====================================")
    
    # Create the Model+MCTS player
    model_mcts_player = ModelPlayer('X', model_path='model.pt', temperature=0.1, 
                                   use_mcts=True, simulation_time=1.0, 
                                   num_simulations=400, c_puct=1.5, top_k_moves=5)
    
    # List of opponents to test against
    opponents = {
        # 'Random': lambda mark: RandomPlayer(mark),
        'Easy': lambda mark: MinimaxPlayer(mark, depth=3),
        # 'MCTS': lambda mark: MCTSPlayer(mark, simulation_time=1.0)
    }
    
    # Test against each opponent type
    for opponent_name, opponent_creator in opponents.items():
        print(f"\n--- Testing against {opponent_name} ---")
        
        # Track results
        results = {'Model+MCTS': 0, opponent_name: 0, 'Draw': 0}
        move_counts = []
        
        for i in range(num_games):
            print(f"Game {i+1}/{num_games}")
            
            # Reset board
            board = UltimateBoard()
            moves_made = 0
            
            # Alternate who goes first
            if i % 2 == 0:
                # Model+MCTS goes first (X)
                model_mcts_player.mark = 'X'
                opponent = opponent_creator('O')
                first_player = 'Model+MCTS'
                second_player = opponent_name
            else:
                # Opponent goes first (X)
                model_mcts_player.mark = 'O'
                opponent = opponent_creator('X')
                first_player = opponent_name
                second_player = 'Model+MCTS'
            
            print(f"{first_player} (X) vs {second_player} (O)")
            
            # Play the game
            while board.winner is None and not board.is_draw:
                current_mark = board.current_player
                
                start_time = time.time()
                if (current_mark == 'X' and first_player == 'Model+MCTS') or \
                   (current_mark == 'O' and second_player == 'Model+MCTS'):
                    # Model+MCTS's turn
                    move = model_mcts_player.get_move(board)
                    player_type = 'Model+MCTS'
                else:
                    # Opponent's turn
                    move = opponent.get_move(board)
                    player_type = opponent_name
                end_time = time.time()
                
                print(f"Player {current_mark} ({player_type}) chose move: {move} (took {end_time-start_time:.2f}s)")
                
                # Make move
                if move:
                    board_row, board_col, row, col = move
                    board.make_move(board_row, board_col, row, col)
                    moves_made += 1
                    
                    # Print board after every few moves
                    if moves_made % 4 == 0:
                        print_compact_board(board)
            
            # Record result
            if board.winner:
                winner = 'Model+MCTS' if ((board.winner == 'X' and first_player == 'Model+MCTS') or 
                                         (board.winner == 'O' and second_player == 'Model+MCTS')) else opponent_name
                results[winner] += 1
                print(f"  Winner: {winner} in {moves_made} moves")
            else:
                results['Draw'] += 1
                print(f"  Draw after {moves_made} moves")
            
            # Always show the final board
            print_compact_board(board)
            
            move_counts.append(moves_made)
        
        # Print summary for this opponent
        print(f"\nSummary against {opponent_name}:")
        print(f"Model+MCTS wins: {results['Model+MCTS']} ({results['Model+MCTS']/num_games*100:.1f}%)")
        print(f"{opponent_name} wins: {results[opponent_name]} ({results[opponent_name]/num_games*100:.1f}%)")
        print(f"Draws: {results['Draw']} ({results['Draw']/num_games*100:.1f}%)")
        print(f"Average game length: {sum(move_counts)/len(move_counts):.1f} moves")

def test_mcts_parameter_comparison():
    """Test different parameters for the MCTS algorithm"""
    print("\nTesting MCTS Parameters")
    print("=====================")
    
    # Create a test board with some initial moves
    board = UltimateBoard()
    moves = [
        (1, 1, 0, 0),  # X plays top-left of center board
        (0, 0, 1, 1),  # O plays center of top-left board
        (1, 1, 1, 1),  # X plays center of center board
    ]
    for move in moves:
        board_row, board_col, row, col = move
        board.make_move(board_row, board_col, row, col)
    
    print("Test board:")
    print_compact_board(board)
    
    print("\nTesting different top-k values:")
    for top_k in [3, 5, 10]:
        # Create player with specific parameter
        player = ModelPlayer('X', model_path='model.pt', temperature=0.1, 
                          use_mcts=True, simulation_time=1.0, 
                          num_simulations=400, top_k_moves=top_k)
        
        # Measure time
        start_time = time.time()
        move = player.get_move(deepcopy(board))
        elapsed = time.time() - start_time
        
        print(f"top_k={top_k}: Best move={move}, Time={elapsed:.3f}s")
    
    print("\nTesting different c_puct values:")
    for c_puct in [0.5, 1.0, 1.5, 3.0]:
        # Create player with specific parameter
        player = ModelPlayer('X', model_path='model.pt', temperature=0.1, 
                          use_mcts=True, simulation_time=1.0, 
                          num_simulations=400, c_puct=c_puct)
        
        # Get move
        start_time = time.time()
        move = player.get_move(deepcopy(board))
        elapsed = time.time() - start_time
        
        print(f"c_puct={c_puct}: Best move={move}, Time={elapsed:.3f}s")

def print_compact_board(board):
    """Print a compact representation of the ultimate tic tac toe board"""
    # Convert board to a string representation
    board_str = str(board).split('\n')
    
    # Print only the essential lines
    essential_lines = []
    for i, line in enumerate(board_str):
        if i < 2:  # Skip header
            continue
        if line.strip() == "" or "---" in line:
            continue
        essential_lines.append(line)
    
    print('\n'.join(essential_lines))
    print()

def main():
    """Main function to run tests"""
    # Check command line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "simple":
            test_model_mcts_vs_simple_opponents(num_games=3)
        elif test_type == "params":
            test_mcts_parameter_comparison()
        elif test_type == "model":
            test_model_vs_model_mcts(num_games=3)
        else:
            print(f"Unknown test type: {test_type}")
            print("Available tests: simple, params, model")
    else:
        # # Run all tests by default
        # print("Running Model vs Model+MCTS test...")
        # test_model_vs_model_mcts(num_games=10)
        
        # print("\nRunning parameter comparison test...")
        # test_mcts_parameter_comparison()
        
        print("\nRunning Model+MCTS vs simple opponents test...")
        test_model_mcts_vs_simple_opponents(num_games=10)

if __name__ == "__main__":
    main()
