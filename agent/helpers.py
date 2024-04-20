# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import numpy as np

from referee.game import PlayerColor, Direction, Coord, BOARD_N, Board


def find_starting_positions(board: dict[Coord, PlayerColor], playerColor: PlayerColor):
    """ Finds the possible locations where pieces can be placed given the current board state """
    player_pieces = []
    starting_positions = []

    # Find all player pieces currently on the board
    for coord, color in board.items():
        if color == playerColor:
            player_pieces.append(coord)

    # First placement of the game, generate a random coordinate to place on
    if not player_pieces:
        def find_rand_coord():
            while True:
                random_coord = Coord(np.random.randint(BOARD_N), np.random.randint(BOARD_N))
                if random_coord not in board:
                    return random_coord

        starting_positions.append(find_rand_coord())

    else:
        for coord in player_pieces:
            # We can place new squares in these four directions
            possible_positions = [coord + Direction.Up, coord + Direction.Down, coord + Direction.Left,
                                  coord + Direction.Right]

            for element in possible_positions:
                if element not in board:
                    starting_positions.append(element)

    return starting_positions
