import numpy as np
import jogador_random as p1 # Importa o módulo do jogador
import neural_network as nn

# Configurações do tabuleiro
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

def create_board():
    return np.zeros((ROWS, COLS), dtype=int)

def is_valid_location(board, col):
    return board[ROWS - 1][col] == EMPTY

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r

def check_win(board, piece):
    for c in range(COLS - 3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True
    
    for c in range(COLS):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True
    
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True
    
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True
    
    return False, []

def blocks_opponent_win(board, col):
    row = get_next_open_row(board, col)

    if row is None:
        return False  # Column is full
    
    # Temporarily drop a piece in the given column for PLAYER1
    drop_piece(board, row, col, PLAYER1) 

    # Check if this move would create a winning condition for PLAYER1
    opponent_win_blocked = check_win(board, PLAYER1)

    # Undo the move by resetting the cell
    board[row][col] = EMPTY

    return opponent_win_blocked

def check_sequence(board, col):
    piece = PLAYER2  # Assume PLAYER2 is making the move (the AI)
    row = get_next_open_row(board, col)  # Get the row where the piece will be dropped
    
    if row is None:
        return False, False, None  # If the column is full, return False for both connections and None for direction
    
    # Drop the piece temporarily
    drop_piece(board, row, col, piece)
    
    # Initialize flags for connections and directions
    two_in_a_row = False
    three_in_a_row = False

    # Check horizontal connections
    horizontal_count = 0
    for c in range(COLS):
        if board[row][c] == piece:
            horizontal_count += 1
        else:
            if horizontal_count == 2:
                two_in_a_row = True
            elif horizontal_count == 3:
                three_in_a_row = True
            horizontal_count = 0

    # Check vertical connections
    vertical_count = 0
    for r in range(ROWS):
        if board[r][col] == piece:
            vertical_count += 1
        else:
            if vertical_count == 2:
                two_in_a_row = True
            elif vertical_count == 3:
                three_in_a_row = True
            vertical_count = 0

    # Remove the piece (undo move)
    board[row][col] = EMPTY

    return two_in_a_row, three_in_a_row

def is_board_full(board):
    # Check if the top row of any column is empty
    for col in range(board.shape[1]):  
        if board[ROWS - 1][col] == EMPTY:  
            return False  
    return True

def is_center_column_move(col):
    center_columns = [COLS // 2 - 1, COLS // 2, COLS // 2 + 1]  # Considering columns 3, 4, 5 as center
    return col in center_columns

def is_isolated_move(board, row, col):
    # Check for adjacency in the same row
    row_check = (col > 0 and board[row][col - 1] != EMPTY) or (col < COLS - 1 and board[row][col + 1] != EMPTY)

    # Check for adjacency in the same column
    col_check = (row > 0 and board[row - 1][col] != EMPTY) or (row < ROWS - 1 and board[row + 1][col] != EMPTY)

    return not (row_check or col_check)  

def play_game(model):
    board = create_board()
    game_over = False
    turn = np.random.randint(0, 2)
    score = 0

    while not game_over:
        current_player = PLAYER1 if turn % 2 == 0 else PLAYER2
        
        if current_player == PLAYER1:
            # Predict move and check validity
            output = nn.predict_move(model, board, current_player)
            col = np.argmax(output)

            while not is_valid_location(board, col):
                # Check if the board is full
                if is_board_full(board):
                    return score  # Return the score if the board is full
                
                # Modify the output probabilities
                output = nn.predict_move(model, board, current_player)
                output[col] = 0  # Mask out the invalid column

                col = np.argmax(output)
        else:
            # Predict move and check validity
            output = nn.predict_move(model, board, current_player)
            col = np.argmax(output)

            if not is_valid_location(board, col):
                # Penalize for invalid move and modify probability distribution to re-select
                score -= 144
                return score

        row = get_next_open_row(board, col)

        if current_player == PLAYER2:
            # Diminui o score se a jogada for isolada
            if is_isolated_move(board, row, col):
                score -= 13

            # Aumenta o score se a jogada bloquear uma possível vitória do adversário
            if blocks_opponent_win(board, col):
                score += 21

            # Aumenta o score se a jogada for no centro
            if is_center_column_move(col):
                score += 5

            # Aumenta o score se a jogada fizer uma sequência
            two_in_a_row, three_in_a_row = check_sequence(board, col)

            if three_in_a_row:
                score += 13
            elif two_in_a_row:
                score += 5
        
        drop_piece(board, row, col, current_player)

        if check_win(board, current_player):
            if current_player == PLAYER2:
                score += 34
            else:
                score -= 34
            return score 

        turn += 1
