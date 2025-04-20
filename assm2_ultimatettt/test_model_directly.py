import time
import argparse
from ultimate_board import UltimateBoard
from player import RandomPlayer, MinimaxPlayer, MCTSPlayer, ModelPlayer

def play_match(player_x, player_o, verbose=True):
    """Play a single match between two players and return the result."""
    board = UltimateBoard()
    move_count = 0
    
    start_time = time.time()
    
    while board.winner is None and not board.is_draw and move_count < 100:  # Safety limit
        current_player = player_x if board.current_player == 'X' else player_o
        
        # Get and make move
        move = current_player.get_move(board)
        if move is None:
            break
            
        board_row, board_col, row, col = move
        board.make_move(board_row, board_col, row, col)
        move_count += 1
        
        if verbose:
            print(f"Move {move_count}: Player {board.current_player} plays at board ({board_row},{board_col}), cell ({row},{col})")
            # Print a simplified board representation
            print_board(board)
    
    duration = time.time() - start_time
    
    # Display result
    result = "Draw"
    if board.winner == 'X':
        result = "X wins"
    elif board.winner == 'O':
        result = "O wins"
    
    if verbose:
        print(f"\nGame over after {move_count} moves ({duration:.2f} seconds)")
        print(f"Result: {result}")
    
    return board.winner, move_count, duration

def print_board(board):
    """Print a simplified text representation of the board."""
    symbols = {None: '.', 'X': 'X', 'O': 'O'}
    
    # Print each small board row by row
    for big_row in range(3):
        for small_row in range(3):
            line = ""
            for big_col in range(3):
                for small_col in range(3):
                    line += symbols[board.boards[big_row][big_col].board[small_row][small_col]] + " "
                if big_col < 2:
                    line += "| "
            print(line)
        if big_row < 2:
            print("-" * 29)
    print()

def run_multiple_matches(num_matches, player_x, player_o, verbose_last=True):
    """Run multiple matches between the same players and summarize results."""
    x_wins = 0
    o_wins = 0
    draws = 0
    total_moves = 0
    total_time = 0
    
    for i in range(num_matches):
        verbose = verbose_last and (i == num_matches - 1)
        if not verbose and i % 10 == 0:
            print(f"Playing match {i+1}/{num_matches}...")
            
        result, moves, duration = play_match(player_x, player_o, verbose)
        
        if result == 'X':
            x_wins += 1
            print('.', end='', flush=True)
        elif result == 'O':
            o_wins += 1
            print('x', end='', flush=True)
        else:
            draws += 1
            print('d', end='', flush=True)
            
        total_moves += moves
        total_time += duration
        
        if (i+1) % 10 == 0:
            print()  # New line every 10 matches
    
    print(f"\n\nResults after {num_matches} matches:")
    print(f"Player X ({player_x.__class__.__name__}): {x_wins} wins ({x_wins/num_matches*100:.1f}%)")
    print(f"Player O ({player_o.__class__.__name__}): {o_wins} wins ({o_wins/num_matches*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_matches*100:.1f}%)")
    print(f"Average moves per game: {total_moves/num_matches:.1f}")
    print(f"Average time per game: {total_time/num_matches:.2f} seconds")
    
    return x_wins, o_wins, draws

def main():
    parser = argparse.ArgumentParser(description="Test the trained model against other agents")
    parser.add_argument("--model_path", default="model.pt", help="Path to the trained model")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for model sampling (0=greedy)")
    parser.add_argument("--opponent", default="random", choices=["random", "easy", "medium", "hard", "mcts"], 
                        help="Opponent type")
    parser.add_argument("--matches", type=int, default=10, help="Number of matches to play")
    parser.add_argument("--as_o", action="store_true", help="Play model as O (second player)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output for all matches")
    
    args = parser.parse_args()
    
    # Create model player
    model_player = ModelPlayer('X' if not args.as_o else 'O', 
                               model_path=args.model_path, 
                               temperature=args.temperature)
    
    # Create opponent
    opponent_mark = 'O' if not args.as_o else 'X'
    if args.opponent == "random":
        opponent = RandomPlayer(opponent_mark)
    elif args.opponent == "easy":
        opponent = MinimaxPlayer(opponent_mark, depth=1)
    elif args.opponent == "medium":
        opponent = MinimaxPlayer(opponent_mark, depth=3)
    elif args.opponent == "hard":
        opponent = MinimaxPlayer(opponent_mark, depth=5)
    else:  # mcts
        opponent = MCTSPlayer(opponent_mark, simulation_time=1.0)
    
    print(f"Testing {model_player.__class__.__name__} against {opponent.__class__.__name__}")
    print(f"Model temperature: {args.temperature}, Number of matches: {args.matches}")
    print(f"Model playing as: {'X (first player)' if not args.as_o else 'O (second player)'}")
    
    # Run matches
    if args.as_o:
        run_multiple_matches(args.matches, opponent, model_player, not args.verbose)
    else:
        run_multiple_matches(args.matches, model_player, opponent, not args.verbose)

if __name__ == "__main__":
    main()
