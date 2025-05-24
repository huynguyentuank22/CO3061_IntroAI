import argparse
import torch
import random
import time
import numpy as np
from tqdm import tqdm

from ultimate_board import UltimateBoard
from rl_player import RLPlayer

def evaluate(model1_path, model2_path, num_games=100, use_mcts=False, verbose=False):
    """Evaluate two models against each other"""
    print(f"Evaluating {model1_path} vs {model2_path} over {num_games} games")
    
    # Initialize players
    player1 = RLPlayer('X', model_path=model1_path, temperature=0.5, use_mcts=use_mcts)
    player2 = RLPlayer('O', model_path=model2_path, temperature=0.5, use_mcts=use_mcts)
    
    # Track results
    results = {"model1_wins": 0, "model2_wins": 0, "draws": 0}
    game_lengths = []
    
    for game_num in tqdm(range(num_games)):
        # Create a new game
        game = UltimateBoard()
        current_player = player1
        other_player = player2
        
        if game_num >= num_games // 2:
            # Switch sides halfway through to ensure fairness
            current_player = player2
            other_player = player1
        
        # Track moves in this game
        moves_count = 0
        
        # Play the game
        while game.winner is None and not game.is_draw:
            move = current_player.get_move(game)
            if move:
                board_row, board_col, row, col = move
                game.make_move(board_row, board_col, row, col)
                moves_count += 1
                
                if verbose and game_num == 0:  # Only print the first game in detail
                    print(f"Player {current_player.mark} plays {move}")
                    print(game)
            
            # Switch players
            current_player, other_player = other_player, current_player
        
        # Record game result
        if game.winner == 'X':
            if game_num < num_games // 2:
                results["model1_wins"] += 1
            else:
                results["model2_wins"] += 1
        elif game.winner == 'O':
            if game_num < num_games // 2:
                results["model2_wins"] += 1
            else:
                results["model1_wins"] += 1
        else:
            results["draws"] += 1
        
        # Record game length
        game_lengths.append(moves_count)
    
    # Calculate statistics
    win_rate_model1 = results["model1_wins"] / num_games
    win_rate_model2 = results["model2_wins"] / num_games
    draw_rate = results["draws"] / num_games
    avg_game_length = sum(game_lengths) / len(game_lengths)
    
    print(f"\nEvaluation Results ({num_games} games):")
    print(f"Model 1 ({model1_path}) wins: {results['model1_wins']} ({win_rate_model1:.2%})")
    print(f"Model 2 ({model2_path}) wins: {results['model2_wins']} ({win_rate_model2:.2%})")
    print(f"Draws: {results['draws']} ({draw_rate:.2%})")
    print(f"Average game length: {avg_game_length:.1f} moves")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Ultimate TTT models')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model')
    parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    parser.add_argument('--use_mcts', action='store_true', help='Use MCTS for move selection')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    evaluate(
        model1_path=args.model1,
        model2_path=args.model2,
        num_games=args.games,
        use_mcts=args.use_mcts,
        verbose=args.verbose
    )
