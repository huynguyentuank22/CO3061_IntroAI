from player import RandomPlayer, MinimaxPlayer, MCTSPlayer, ModelPlayer
from ultimate_board import UltimateBoard
import argparse

def play_game(player_x, player_o, verbose=True):
    """Play a single game between two players."""
    board = UltimateBoard()
    players = {'X': player_x, 'O': player_o}
    current_player = 'X'
    
    move_count = 0
    while board.winner is None and not board.is_draw and move_count < 81:
        if verbose:
            print(f"\nMove {move_count + 1}, {current_player}'s turn")
        
        move = players[current_player].get_move(board)
        if move is None:
            break
            
        board_row, board_col, row, col = move
        if verbose:
            print(f"Player {current_player} plays at board ({board_row},{board_col}), cell ({row},{col})")
        
        board.make_move(board_row, board_col, row, col)
        move_count += 1
        
        if verbose:
            print(board)
        
        current_player = 'O' if current_player == 'X' else 'X'
    
    if verbose:
        print("\nGame Over!")
        if board.winner:
            print(f"Player {board.winner} wins!")
        elif board.is_draw:
            print("It's a draw!")
    
    return board.winner

def run_matches(num_games=5):
    """Run a series of matches between your model and other players."""
    # Create players
    model_player_x = ModelPlayer('X', model_path='model.pt', temperature=0.5)
    model_player_o = ModelPlayer('O', model_path='model.pt', temperature=0.5)
    
    random_player_x = RandomPlayer('X')
    random_player_o = RandomPlayer('O')
    
    minimax_player_x = MinimaxPlayer('X', depth=2)  # Lower depth for faster play
    minimax_player_o = MinimaxPlayer('O', depth=2)
    
    mcts_player_x = MCTSPlayer('X', simulation_time=1.0)
    mcts_player_o = MCTSPlayer('O', simulation_time=1.0)
    
    # Define matchups to test
    matchups = [
        (model_player_x, random_player_o, "Model (X) vs Random (O)"),
        (random_player_x, model_player_o, "Random (X) vs Model (O)"),
        (model_player_x, minimax_player_o, "Model (X) vs Minimax (O)"),
        (minimax_player_x, model_player_o, "Minimax (X) vs Model (O)"),
        (model_player_x, mcts_player_o, "Model (X) vs MCTS (O)"),
        (mcts_player_x, model_player_o, "MCTS (X) vs Model (O)")
    ]
    
    # Run matches
    for player_x, player_o, description in matchups:
        wins_x = 0
        wins_o = 0
        draws = 0
        
        print(f"\n===== {description} =====")
        for i in range(num_games):
            print(f"\nGame {i+1}/{num_games}")
            result = play_game(player_x, player_o, verbose=True)
            
            if result == 'X':
                wins_x += 1
            elif result == 'O':
                wins_o += 1
            else:
                draws += 1
        
        print(f"\n{description} Results:")
        print(f"X wins: {wins_x} ({wins_x/num_games*100:.1f}%)")
        print(f"O wins: {wins_o} ({wins_o/num_games*100:.1f}%)")
        print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Ultimate Tic Tac Toe model")
    parser.add_argument("--games", type=int, default=5, help="Number of games per matchup")
    args = parser.parse_args()
    
    run_matches(args.games)
