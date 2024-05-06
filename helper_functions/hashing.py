# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import random

from referee.game import PlayerColor, Coord, BOARD_N

NUM_PLAYERS = 2


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
