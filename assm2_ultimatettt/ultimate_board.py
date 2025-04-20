from small_board import SmallBoard

class UltimateBoard:
    def __init__(self):
        # Create 3x3 grid of small boards
        self.boards = [[SmallBoard() for _ in range(3)] for _ in range(3)]
        self.active_board = None  # (row, col) or None if any board is valid
        self.current_player = 'X'  # X goes first
        self.winner = None
        self.is_draw = False
    
    def make_move(self, board_row, board_col, row, col):
        """Make a move on the specified small board at the given position"""
        # Check if move is valid
        if self.winner is not None or self.is_draw:
            return False
            
        # Check if this is a valid board to play on
        if self.active_board is not None and (board_row, board_col) != self.active_board:
            return False
        
        # Make the move on the small board
        if not self.boards[board_row][board_col].make_move(row, col, self.current_player):
            return False
            
        # Set the active board for the next player
        target_board = (row, col)
        if (self.boards[target_board[0]][target_board[1]].winner is not None or 
            self.boards[target_board[0]][target_board[1]].is_full):
            # If the target board is complete, allow next player to play anywhere
            self.active_board = None
        else:
            self.active_board = target_board
        
        # Check if the overall game is won or drawn
        self._check_winner()
        self._check_draw()
        
        # Switch player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        return True
    
    def _check_winner(self):
        """Check if there is an overall winner"""
        # Create a 3x3 grid representing the winners of each small board
        winners = [[self.boards[row][col].winner for col in range(3)] for row in range(3)]
        
        # Check rows
        for row in range(3):
            if winners[row][0] == winners[row][1] == winners[row][2] and winners[row][0] is not None:
                self.winner = winners[row][0]
                return
        
        # Check columns
        for col in range(3):
            if winners[0][col] == winners[1][col] == winners[2][col] and winners[0][col] is not None:
                self.winner = winners[0][col]
                return
        
        # Check diagonals
        if winners[0][0] == winners[1][1] == winners[2][2] and winners[0][0] is not None:
            self.winner = winners[0][0]
            return
        if winners[0][2] == winners[1][1] == winners[2][0] and winners[0][2] is not None:
            self.winner = winners[0][2]
            return
    
    def _check_draw(self):
        """Check if the game is a draw (all small boards are either won or full)"""
        if self.winner is not None:
            return
            
        for row in range(3):
            for col in range(3):
                if self.boards[row][col].winner is None and not self.boards[row][col].is_full:
                    return
        self.is_draw = True
    
    def get_available_moves(self):
        """Get all valid moves as (board_row, board_col, row, col)"""
        moves = []
        
        # If an active board is specified, only return moves for that board
        if self.active_board is not None:
            board_row, board_col = self.active_board
            for row, col in self.boards[board_row][board_col].get_available_moves():
                moves.append((board_row, board_col, row, col))
            return moves
        
        # Otherwise, return moves for all boards that aren't complete
        for board_row in range(3):
            for board_col in range(3):
                if self.boards[board_row][board_col].winner is None and not self.boards[board_row][board_col].is_full:
                    for row, col in self.boards[board_row][board_col].get_available_moves():
                        moves.append((board_row, board_col, row, col))
        
        return moves
