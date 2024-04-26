# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

from agent.game import Game
from agent.placement_algorithms import find_starting_positions
from referee.game import PlayerColor, Coord, BOARD_N

NUM_PLAYERS = 2
GOOD_THRESHOLD = 5
BAD_THRESHOLD = 8


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
        elif 0 < element < GOOD_THRESHOLD:
            num_good_len += 1

    return num_good_len, num_bad_len


def utility_value(game: Game, state: dict[Coord, PlayerColor]):
    # finding available squares for player moves
    our_positions = find_starting_positions(state, game.our_player)
    their_positions = find_num_pieces(state, game.other_player)
    our_options = len(our_positions)
    their_options = len(their_positions)

    # finding number of squares on board
    our_pieces, their_pieces = find_num_pieces(state, game.our_player)

    # finding length of lines on board and if they are too long or not
    row_len, col_len = line_lengths(state)
    num_good_len = 0
    num_bad_len = 0

    row_weights = line_length_weight(row_len)
    num_good_len += row_weights[0]
    num_bad_len += row_weights[1]

    col_weights = line_length_weight(col_len)
    num_good_len += col_weights[0]
    num_bad_len += col_weights[1]

    pass
