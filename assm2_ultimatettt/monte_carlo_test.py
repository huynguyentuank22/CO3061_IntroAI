from copy import deepcopy
import random
import time
from ultimate_board import UltimateBoard
from player import MCTSPlayer

def format_move(move):
    """Format a move as a readable string: board(r,c) -> cell(r,c)"""
    board_row, board_col, row, col = move
    return f"b({board_row},{board_col})->c({row},{col})"

def print_ultimate_board_state(board):
    """Print the current state of the Ultimate Tic-Tac-Toe board in a readable format."""
    lines = []
    lines.append("=== ULTIMATE BOARD STATE ===")
    lines.append(f"Current player: {board.current_player}")
    lines.append(f"Active board: {board.active_board}")
    lines.append("")
    
    # Print the entire 9x9 grid with proper formatting
    for br in range(3):
        # Print three rows of small boards side by side
        for r in range(3):
            row_str = ""
            for bc in range(3):
                # For each cell in the current small board
                for c in range(3):
                    cell = board.boards[br][bc].board[r][c]
                    row_str += f" {cell if cell else '.'} "
                # Add separator between small boards
                if bc < 2:
                    row_str += "|"
            lines.append(row_str)
        # Add separator between rows of small boards
        if br < 2:
            lines.append("-" * 29)
    
    # Print which small boards have been won
    lines.append("\nSmall boards won:")
    for br in range(3):
        row_str = ""
        for bc in range(3):
            winner = board.boards[br][bc].winner
            row_str += f" {winner if winner else '-'} "
        lines.append(row_str)
    lines.append("===========================")
    return "\n".join(lines)

def mcts_with_logging(board, player_mark, num_simulations=1000):
    """
    Run MCTS algorithm with a fixed number of simulations per move and detailed logging.
    This helps visualize how MCTSPlayer makes decisions.
    """
    opponent_mark = 'O' if player_mark == 'X' else 'X'
    moves = board.get_available_moves()
    
    log_lines = []
    log_lines.append("# MONTE CARLO TREE SEARCH DECISION PROCESS")
    log_lines.append(f"Current player: {player_mark}")
    log_lines.append(f"Evaluating {len(moves)} possible moves with {num_simulations} simulations per move\n")
    
    # Initialize statistics
    wins = {move: 0 for move in moves}
    draws = {move: 0 for move in moves}
    losses = {move: 0 for move in moves}
    
    # Run simulations for each move
    total_time_start = time.time()
    
    for move in moves:
        move_time_start = time.time()
        log_lines.append(f"Testing move: {format_move(move)}")
        
        # Make a copy of the board and apply the move
        board_copy = deepcopy(board)
        board_row, board_col, row, col = move
        board_copy.make_move(board_row, board_col, row, col)
        
        # Check if this is an immediate win
        if board_copy.winner == player_mark:
            log_lines.append(f"  → This is a winning move!")
            wins[move] = num_simulations
            continue
            
        # Log the resulting position
        log_lines.append(f"  → Running {num_simulations} random simulations from this position")
        
        # Track results for this move
        move_wins = 0
        move_draws = 0
        move_losses = 0
        
        # Run simulations
        for _ in range(num_simulations):
            result = simulate_random_game(board_copy, opponent_mark, player_mark)
            
            if result == player_mark:
                move_wins += 1
            elif result == opponent_mark:
                move_losses += 1
            else:  # Draw or None
                move_draws += 1
                
        # Store results
        wins[move] = move_wins
        draws[move] = move_draws
        losses[move] = move_losses
        
        # Calculate win rate
        win_rate = move_wins / num_simulations * 100
        draw_rate = move_draws / num_simulations * 100
        loss_rate = move_losses / num_simulations * 100
        
        move_time = time.time() - move_time_start
        log_lines.append(f"  → Results: {move_wins} wins, {move_draws} draws, {move_losses} losses")
        log_lines.append(f"  → Win rate: {win_rate:.1f}%, Draw rate: {draw_rate:.1f}%, Loss rate: {loss_rate:.1f}%")
        log_lines.append(f"  → Simulation time: {move_time:.3f} seconds\n")
    
    # Find best move by win rate
    best_move = None
    best_win_rate = -1
    
    for move in moves:
        win_rate = wins[move] / num_simulations
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_move = move
    
    total_time = time.time() - total_time_start
    
    # Final decision and statistics
    log_lines.append(f"FINAL DECISION: {format_move(best_move)} with win rate {best_win_rate*100:.1f}%")
    log_lines.append(f"Total simulation time: {total_time:.3f} seconds")
    
    # Create a visual comparison of move win rates
    log_lines.append("\n# MOVE WIN RATE COMPARISON")
    log_lines.append("Move               | Win %  | Draw % | Loss % | Simulations")
    log_lines.append("--------------------|--------|--------|--------|------------")
    
    # Sort moves by win rate for better visualization
    sorted_moves = sorted(moves, key=lambda m: wins[m]/num_simulations, reverse=True)
    
    for move in sorted_moves:
        win_percent = wins[move] / num_simulations * 100
        draw_percent = draws[move] / num_simulations * 100
        loss_percent = losses[move] / num_simulations * 100
        best_marker = " ←" if move == best_move else ""
        
        log_lines.append(f"{format_move(move).ljust(18)} | {win_percent:5.1f}% | {draw_percent:5.1f}% | {loss_percent:5.1f}% | {num_simulations}{best_marker}")
    
    return best_move, "\n".join(log_lines)

def simulate_random_game(board, starting_player_mark, player_mark):
    """
    Simulate a random game from the current position.
    Returns the winner or None if draw.
    """
    board_copy = deepcopy(board)
    current_mark = starting_player_mark
    
    # Play random moves until the game is over
    while board_copy.winner is None and not board_copy.is_draw:
        moves = board_copy.get_available_moves()
        if not moves:
            break
            
        # Make a random move
        board_row, board_col, row, col = random.choice(moves)
        board_copy.make_move(board_row, board_col, row, col)
        current_mark = 'O' if current_mark == 'X' else 'X'
    
    return board_copy.winner

def create_mcts_visualization(board, move_stats, best_move, num_simulations):
    """Create a visual representation of the MCTS decision process"""
    tree_lines = []
    tree_lines.append("\n# MCTS DECISION TREE VISUALIZATION")
    tree_lines.append("Root (Current Position)")
    tree_lines.append("│")
    
    # Get moves sorted by win rate
    moves = list(move_stats.keys())
    moves.sort(key=lambda m: move_stats[m][0]/num_simulations, reverse=True)
    
    # Show each move branch
    for i, move in enumerate(moves):
        wins, draws, losses = move_stats[move]
        win_rate = wins / num_simulations * 100
        
        # Determine the branch connector
        prefix = "├── " if i < len(moves) - 1 else "└── "
        is_best = move == best_move
        
        # Mark the best move
        marker = " ★" if is_best else ""
        
        tree_lines.append(f"{prefix}{format_move(move)} [Win: {win_rate:.1f}%]{marker}")
        
        # Show some random outcomes as children
        child_prefix = "│   " if i < len(moves) - 1 else "    "
        
        # Only show child nodes for first few moves to keep visualization manageable
        if i < 3:
            # Win outcome
            win_prefix = child_prefix + "├── "
            tree_lines.append(f"{win_prefix}Random playout → {player_mark} wins")
            
            # Draw outcome
            draw_prefix = child_prefix + "├── "
            tree_lines.append(f"{draw_prefix}Random playout → Draw")
            
            # Loss outcome
            loss_prefix = child_prefix + "└── "
            tree_lines.append(f"{loss_prefix}Random playout → {opponent_mark} wins")
        else:
            tree_lines.append(f"{child_prefix}└── ...")
    
    return "\n".join(tree_lines)

# Example usage
if __name__ == "__main__":
    # Create a board with some moves already made
    board = UltimateBoard()
    
    # Make some example moves to create an interesting position
    board.make_move(0, 0, 1, 1)  # X plays in top-left board, center cell
    board.make_move(1, 1, 0, 0)  # O plays in center board, top-left cell
    board.make_move(0, 0, 2, 2)  # X plays in top-left board, bottom-right cell
    board.make_move(2, 2, 0, 0)  # O plays in bottom-right board, top-left cell
    
    # Print the initial state
    board_state = print_ultimate_board_state(board)
    print(board_state)
    
    # Set player mark and number of simulations
    player_mark = board.current_player  # Should be X
    opponent_mark = 'O' if player_mark == 'X' else 'X'
    num_simulations = 100  # Reduce for testing, use 1000+ for report
    
    # Run MCTS with logging
    print("Running MCTS simulations...")
    
    # Track move statistics for visualization
    moves = board.get_available_moves()
    move_stats = {move: [0, 0, 0] for move in moves}  # [wins, draws, losses]
    
    # Manual simulation for each move to gather statistics
    for move in moves:
        board_copy = deepcopy(board)
        board_row, board_col, row, col = move
        board_copy.make_move(board_row, board_col, row, col)
        
        # Skip simulation if winning move
        if board_copy.winner == player_mark:
            move_stats[move] = [num_simulations, 0, 0]  # All wins
            continue
            
        # Run simulations
        for _ in range(num_simulations):
            result = simulate_random_game(board_copy, opponent_mark, player_mark)
            
            if result == player_mark:
                move_stats[move][0] += 1  # Win
            elif result == opponent_mark:
                move_stats[move][2] += 1  # Loss
            else:
                move_stats[move][1] += 1  # Draw
    
    # Find best move
    best_move = max(moves, key=lambda m: move_stats[m][0])
    
    # Run the full logging version
    best_move, log_text = mcts_with_logging(board, player_mark, num_simulations)
    
    # Create visualization
    tree_viz = create_mcts_visualization(board, move_stats, best_move, num_simulations)
    
    # Print and save results
    print(log_text)
    print(tree_viz)
    
    # Save to file for report
    with open("mcts_decision_log.txt", "w", encoding='utf-8') as f:
        f.write(board_state + "\n\n")
        f.write(log_text)
        f.write(tree_viz)
        f.write("\n\n# EXPLANATION OF MCTS DECISION MAKING\n")
        f.write("Monte Carlo Tree Search chose move " + format_move(best_move) + " because:\n")
        f.write("1. It had the highest win rate in random playouts\n")
        f.write("2. MCTS evaluates moves by simulating random games to their conclusion\n")
        f.write("3. This statistical approach helps in positions where traditional evaluation is difficult\n")
        f.write("4. It balances short-term tactics with long-term position quality\n")
        f.write("5. The win percentage represents the probability of winning from that position\n")
    
    print("\nMCTS analysis saved to 'mcts_decision_log.txt'")