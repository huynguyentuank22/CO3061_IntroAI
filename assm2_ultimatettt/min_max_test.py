from copy import deepcopy
from ultimate_board import UltimateBoard

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

def evaluate_board(board, player_mark):
    """Simple heuristic evaluation function"""
    score = 0
    opponent_mark = 'O' if player_mark == 'X' else 'X'
    
    # Count small boards won by each player
    for row in range(3):
        for col in range(3):
            if board.boards[row][col].winner == player_mark:
                score += 3
            elif board.boards[row][col].winner == opponent_mark:
                score -= 3
    
    return score

def minimax_with_logging(board, player_mark, depth=1):
    """
    Run minimax algorithm with depth=1 and log the evaluation process.
    This simplified version helps visualize how MinimaxPlayer makes decisions.
    """
    opponent_mark = 'O' if player_mark == 'X' else 'X'
    moves = board.get_available_moves()
    
    log_lines = []
    log_lines.append("# MINIMAX DECISION PROCESS (DEPTH=1)")
    log_lines.append(f"Current player: {player_mark}")
    log_lines.append(f"Evaluating {len(moves)} possible moves\n")
    
    # Track best move and score
    best_score = -float('inf')
    best_move = None
    
    # First level - player's moves
    for move in moves:
        log_lines.append(f"Testing move: {format_move(move)}")
        
        # Make a copy of the board and apply the move
        board_copy = deepcopy(board)
        board_row, board_col, row, col = move
        board_copy.make_move(board_row, board_col, row, col)
        
        # Check if this is a winning move
        if board_copy.winner == player_mark:
            log_lines.append(f"  → This is a winning move! Score: 100")
            score = 100
        else:
            # Start with the static evaluation of the position
            static_score = evaluate_board(board_copy, player_mark)
            log_lines.append(f"  → Initial position value: {static_score}")
            
            # Since depth=1, we examine opponent's responses
            opponent_moves = board_copy.get_available_moves()
            log_lines.append(f"  → Opponent has {len(opponent_moves)} possible responses")
            
            # Track worst possible outcome (from player's perspective)
            worst_score = float('inf')
            worst_response = None
            
            # Second level - opponent's responses
            for opp_move in opponent_moves:
                opp_board_copy = deepcopy(board_copy)
                board_row, board_col, row, col = opp_move
                opp_board_copy.make_move(board_row, board_col, row, col)
                
                # Check if opponent's move results in a win
                if opp_board_copy.winner == opponent_mark:
                    opp_score = -100
                    log_lines.append(f"    • Response {format_move(opp_move)}: Opponent wins! Score: {opp_score}")
                else:
                    # Evaluate the resulting position
                    opp_score = evaluate_board(opp_board_copy, player_mark)
                    log_lines.append(f"    • Response {format_move(opp_move)}: Score: {opp_score}")
                
                # Update worst response if needed
                if opp_score < worst_score:
                    worst_score = opp_score
                    worst_response = opp_move
            
            # Minimax principle: player assumes opponent picks the response minimizing player's score
            log_lines.append(f"  → Opponent's best response: {format_move(worst_response) if worst_response else 'None'}")
            log_lines.append(f"  → Final move score: {worst_score}")
            score = worst_score
        
        # Update best move if needed
        if score > best_score:
            best_score = score
            best_move = move
            log_lines.append(f"  → New best move found! Score: {best_score}\n")
        else:
            log_lines.append(f"  → Not better than current best ({best_score})\n")
    
    # Final decision
    log_lines.append(f"FINAL DECISION: {format_move(best_move)} with score {best_score}")
    
    return best_move, "\n".join(log_lines)

def create_decision_tree_visualization(board, best_move, moves_to_show=5):
    """Create a simplified decision tree visualization for the report"""
    tree_lines = []
    tree_lines.append("\n# MINIMAX DECISION TREE VISUALIZATION")
    tree_lines.append("Root (X to move)")
    tree_lines.append("│")
    
    # Sort moves by their score (best first)
    # For this simplified version, we'll just use the best move and a few others
    moves = board.get_available_moves()
    if len(moves) > moves_to_show:
        # Keep the best move and some others
        other_moves = [m for m in moves if m != best_move][:moves_to_show-1]
        moves = [best_move] + other_moves
    
    # Show the moves
    for i, move in enumerate(moves):
        prefix = "├── " if i < len(moves) - 1 else "└── "
        is_best = move == best_move
        
        # Make a copy and apply the move
        board_copy = deepcopy(board)
        board_row, board_col, row, col = move
        board_copy.make_move(board_row, board_col, row, col)
        score = evaluate_board(board_copy, board.current_player)
        
        # Mark the best move
        marker = " (BEST)" if is_best else ""
        tree_lines.append(f"{prefix}{format_move(move)} [score: {score}]{marker}")
        
        # Child nodes (opponent's responses)
        child_prefix = "│   " if i < len(moves) - 1 else "    "
        opponent_moves = board_copy.get_available_moves()
        
        # Only show a few opponent responses
        opp_moves_to_show = min(3, len(opponent_moves))
        for j in range(opp_moves_to_show):
            opp_move = opponent_moves[j]
            opp_prefix = child_prefix + ("├── " if j < opp_moves_to_show - 1 else "└── ")
            
            # Evaluate opponent move
            opp_board = deepcopy(board_copy)
            board_row, board_col, row, col = opp_move
            opp_board.make_move(board_row, board_col, row, col)
            opp_score = evaluate_board(opp_board, board.current_player)
            
            tree_lines.append(f"{opp_prefix}{format_move(opp_move)} [score: {opp_score}]")
        
        # Show ellipsis if there are more opponent moves
        if len(opponent_moves) > opp_moves_to_show:
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
    
    # Print the initial state before running minimax
    board_state = print_ultimate_board_state(board)
    print(board_state)
    
    # Now it's X's turn to play in the top-left board
    player_mark = board.current_player
    
    # Run minimax with logging
    best_move, log_text = minimax_with_logging(board, player_mark)
    
    # Create a visual decision tree
    tree_viz = create_decision_tree_visualization(board, best_move)
    
    # Print the log
    print(log_text)
    print(tree_viz)
    
    # Save to a file for the report
    with open("minimax_decision_log.txt", "w", encoding='utf-8') as f:
        f.write(board_state + "\n\n")
        f.write(log_text)
        f.write(tree_viz)
        f.write("\n\n# EXPLANATION OF MOVE SELECTION\n")
        f.write("The minimax algorithm chose move b(0,0)->c(0,0) with score 3 because:\n")
        f.write("1. This move completes a diagonal line in the top-left small board (0,0)\n")
        f.write("2. Winning a small board gives a score of +3 points\n") 
        f.write("3. No opponent response can undo this advantage\n")
        f.write("4. This creates a permanent strategic advantage\n")
        
    print("\nDetailed minimax analysis saved to 'minimax_decision_log.txt'")