# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, BOARD_N

NUM_PLAYERS = 2
GOOD_THRESHOLD = 5
BAD_THRESHOLD = 7

MOVE_WEIGHT = 2
PIECE_WEIGHT = 5
LINE_WEIGHT = 5
HOLE_WEIGHT = 3


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
    num_good_len = 0
    num_bad_len = 0

    for element in length_dict.values():
        if element > BAD_THRESHOLD:
            num_bad_len += 1
        elif element < GOOD_THRESHOLD:
            num_good_len += 1

    return num_good_len, num_bad_len
