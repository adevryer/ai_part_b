# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, BOARD_N

LOW_THRESHOLD = 3
HIGH_THRESHOLD = 7


def find_num_pieces(board: dict[Coord, PlayerColor], our_player: PlayerColor):
    our_pieces = 0
    their_pieces = 0

    # Find all player pieces currently on the board
    for coord, color in board.items():
        if color == our_player:
            our_pieces += 1
        else:
            their_pieces += 1

    return our_pieces, their_pieces


def line_lengths(board: dict[Coord, PlayerColor]):
    row_len = {}
    col_len = {}

    for i in range(BOARD_N):
        curr_len = 0
        for j in range(BOARD_N):
            if Coord(i, j) in board.keys():
                curr_len += 1

        row_len[i] = curr_len

    for i in range(BOARD_N):
        curr_len = 0
        for j in range(BOARD_N):
            if Coord(j, i) in board.keys():
                curr_len += 1

        col_len[i] = curr_len

    return row_len, col_len


def line_length_weight(length_dict):
    num_high_len = 0
    num_low_len = 0

    for element in length_dict.values():
        if element > HIGH_THRESHOLD:
            num_high_len += 1
        elif element < LOW_THRESHOLD:
            num_low_len += 1

    return num_high_len, num_low_len
