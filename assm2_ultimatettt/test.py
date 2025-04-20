import random
import copy

def get_valid_moves(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == ' ']

def make_move(board, move, player):
    new_board = copy.deepcopy(board)
    new_board[move[0]][move[1]] = player
    return new_board

def check_winner(board):
    lines = []
    for i in range(3):
        lines.append(board[i])  
        lines.append([board[0][i], board[1][i], board[2][i]])  
    lines.append([board[0][0], board[1][1], board[2][2]])
    lines.append([board[0][2], board[1][1], board[2][0]])
    for line in lines:
        if line == ['X'] * 3: return 'X'
        if line == ['O'] * 3: return 'O'
    if all(cell != ' ' for row in board for cell in row):
        return 'Draw'
    return None  

def simulate_random_game(board, player):
    current_player = player
    while True:
        winner = check_winner(board)
        if winner:
            return winner
        moves = get_valid_moves(board)
        move = random.choice(moves)
        board = make_move(board, move, current_player)
        current_player = 'O' if current_player == 'X' else 'X'

def monte_carlo_search(board, player, simulations_per_move=1000):
    valid_moves = get_valid_moves(board)
    best_move = None
    best_win_rate = -1

    for move in valid_moves:
        wins = 0
        for _ in range(simulations_per_move):
            simulated_board = make_move(board, move, player)
            result = simulate_random_game(simulated_board, 'O' if player == 'X' else 'X')
            if result == player:
                wins += 1
        win_rate = wins / simulations_per_move
        print(f"Move {move}: Win rate = {win_rate}")
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_move = move

    return best_move

initial_board = [
    ['X', 'O', ' '],
    [' ', 'X', ' '],
    ['O', ' ', ' ']
]

best = monte_carlo_search(initial_board, 'X', simulations_per_move=1000)
print(f"Best move for X is: {best}")
