# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Direction, Coord, BOARD_N, Board, PlaceAction


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
