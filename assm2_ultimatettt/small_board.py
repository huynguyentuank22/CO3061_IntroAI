class SmallBoard:
    def __init__(self):
        # Initialize empty 3x3 board
        self.board = [[None for _ in range(3)] for _ in range(3)]
        self.winner = None
        self.is_full = False
    
    def make_move(self, row, col, player):
        """Make a move on the small board if valid"""
        if self.winner is not None or self.is_full:
            return False
        if row < 0 or row > 2 or col < 0 or col > 2:
            return False
        if self.board[row][col] is not None:
            return False
        
        self.board[row][col] = player
        self._check_winner()
        self._check_full()
        return True
    
    def _check_winner(self):
        """Check if there is a winner on this board"""
        # Check rows
        for row in range(3):
            if self.board[row][0] == self.board[row][1] == self.board[row][2] and self.board[row][0] is not None:
                self.winner = self.board[row][0]
                return
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and self.board[0][col] is not None:
                self.winner = self.board[0][col]
                return
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] is not None:
            self.winner = self.board[0][0]
            return
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] is not None:
            self.winner = self.board[0][2]
            return
    
    def _check_full(self):
        """Check if the board is full (no empty cells)"""
        for row in range(3):
            for col in range(3):
                if self.board[row][col] is None:
                    return
        self.is_full = True
    
    def get_available_moves(self):
        """Return list of available moves as (row, col) tuples"""
        if self.winner is not None or self.is_full:
            return []
        
        moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] is None:
                    moves.append((row, col))
        return moves
