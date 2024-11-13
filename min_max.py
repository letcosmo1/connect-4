import numpy as np
import math

NAME = "minimax"

# Definições de parâmetros
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2
WINDOW_LENGTH = 4
DEPTH = 3  # Profundidade de busca da IA

# Função para verificar se uma coluna ainda tem espaço para jogada
def is_valid_location(board, col):
    return board[ROWS-1][col] == EMPTY

# Função para obter a próxima linha disponível em uma coluna
def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r

# Função para fazer uma jogada no tabuleiro
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# Função para avaliar o tabuleiro atual
def score_position(board, piece):
    score = 0

    # Pontuação central (favorece o centro do tabuleiro)
    center_array = [int(i) for i in list(board[:, COLS//2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Verificar linhas horizontais
    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLS-3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Verificar colunas verticais
    for c in range(COLS):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROWS-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Verificar diagonais ascendentes
    for r in range(ROWS-3):
        for c in range(COLS-3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Verificar diagonais descendentes
    for r in range(ROWS-3):
        for c in range(COLS-3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

# Função para avaliar uma "janela" de quatro espaços no tabuleiro
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER1 if piece == PLAYER2 else PLAYER2

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

# Função para verificar se uma jogada vence o jogo
def check_win(board, piece):
    # Horizontal
    for c in range(COLS-3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Vertical
    for c in range(COLS):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Diagonal ascendente
    for c in range(COLS-3):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Diagonal descendente
    for c in range(COLS-3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

    return False

# Função para obter a lista de colunas válidas
def get_valid_locations(board):
    valid_locations = []
    for col in range(COLS):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

# Função Minimax com poda alfa-beta
def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = check_win(board, PLAYER1) or check_win(board, PLAYER2) or len(valid_locations) == 0

    if depth == 0 or is_terminal:
        if is_terminal:
            if check_win(board, PLAYER2):
                return (None, 100000000000)
            elif check_win(board, PLAYER1):
                return (None, -100000000000)
            else:
                return (None, 0)  # Empate
        else:
            return (None, score_position(board, PLAYER2))

    if maximizingPlayer:
        value = -math.inf
        best_col = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER2)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value

    else:  # Minimizing player
        value = math.inf
        best_col = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER1)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

# Função para retornar um vetor de pontuações para cada coluna
def minmax_score(board, piece):
    valid_locations = get_valid_locations(board)
    pontuacoes = []

    # Calcula a pontuação de cada coluna válida usando o minimax
    for col in range(COLS):
        if col in valid_locations:
            row = get_next_open_row(board, col)
            board_copy = board.copy()
            drop_piece(board_copy, row, col, piece)
            _, score = minimax(board_copy, DEPTH, -math.inf, math.inf, piece == PLAYER2)
        else:
            score = -10000  # Atribui um valor baixo se a coluna não é válida
        pontuacoes.append(score)

    return pontuacoes

# Função principal para fazer a jogada
def jogada(board, piece):
    scores = minmax_score(board, piece)
    print(scores)
    best_col = scores.index(max(scores))
    return best_col

# # Função principal para fazer a jogada
# def jogada(board, piece):
#     col, minimax_score = minimax(board, DEPTH, -math.inf, math.inf, True)
#     return col
