# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent
import random

from referee.game import PlayerColor, Direction, Coord, BOARD_N, Board, PlaceAction

NUM_PLAYERS = 2


def find_starting_positions(board: dict[Coord, PlayerColor], playerColor: PlayerColor):
    """ Finds the possible locations where pieces can be placed given the current board state """
    player_pieces = []
    starting_positions = []

    # Find all player pieces currently on the board
    for coord, color in board.items():
        if color == playerColor:
            player_pieces.append(coord)

    for coord in player_pieces:
        # We can place new squares in these four directions
        possible_positions = [coord + Direction.Up, coord + Direction.Down, coord + Direction.Left,
                              coord + Direction.Right]

        for element in possible_positions:
            if element not in board:
                starting_positions.append(element)

    return starting_positions


def init_board():
    hash_table = [[[random.randint(0, pow(2, 16)) for k in range(NUM_PLAYERS)] for j in range(BOARD_N)]
                  for i in range(BOARD_N)]
    return hash_table


def hash_index(player):
    if player == PlayerColor.RED:
        return 0
    elif player == PlayerColor.BLUE:
        return 1
    else:
        return -1


def board_hash(state: dict[Coord, PlayerColor], hash_table):
    h = 0
    for i in range(BOARD_N):
        for j in range(BOARD_N):
            if Coord(i, j) in state.keys():
                player_index = hash_index(state[Coord(i, j)])
                h ^= hash_table[i][j][player_index]
    return h