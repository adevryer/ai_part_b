# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import itertools

from .hashing_algorithms import init_board, board_hash
from .utility_calculators import find_num_pieces, line_lengths, line_length_weight
from .search_algorithms import PlacementProblem, find_all_placements, find_starting_positions
from referee.game import PlayerColor, PlaceAction, Coord, BOARD_N

# We will always be able to place one of these two pieces on our first go
FIRST_PIECES = [PlaceAction(Coord(2, 2), Coord(2, 3), Coord(2, 4), Coord(2, 5)),
                PlaceAction(Coord(7, 6), Coord(7, 7), Coord(7, 8), Coord(7, 9))]

# move number threshold to start looking two moves ahead to search for death moves
EVAL_THRESHOLD = 400

# evaluation function feature weights
MOVE_WEIGHT = 10
PIECE_WEIGHT = 125
CHANGE_WEIGHT = 300
LINE_WEIGHT = 5


class Game:
    """ Class for the Game problem. Contains methods to find possible actions, results of actions and a heuristic
    value calculator for given game states. Adapted from AIMA's Python code repository for gameplay."""

    def __init__(self, colour: PlayerColor):
        """ The constructor specifies the initial board as well as other game elements (e.g. transposition table,
        Zobrist hashing table etc). """
        self.our_player = colour
        self.other_player = (PlayerColor.BLUE if colour == PlayerColor.RED else PlayerColor.RED)
        self.first = True
        self.hash_table = init_board()
        self.utility_values = {}

    def actions(self, state, colour):
        """ Return a list of the allowable moves at this point. """
        if not self.first:
            possible_actions = []
            place_actions = []

            # Find squares which are adjacent to red blocks
            placement_positions = find_starting_positions(state, colour)

            for position in placement_positions:
                current_actions = find_all_placements(PlacementProblem(position, state))

                for element in current_actions:
                    possible_actions.append(element)

            # Remove any duplicate states from the list
            possible_actions = list(possible_actions for possible_actions, _ in itertools.groupby(possible_actions))

            # Turn these combinations into PlaceActions and return
            for element in possible_actions:
                place_actions.append(PlaceAction(element[0], element[1], element[2], element[3]))

            return place_actions

        else:
            # Pick one of the two starting coordinates
            self.first = False
            possible_placement = [Coord(FIRST_PIECES[0].c1.r, FIRST_PIECES[0].c1.c),
                                  Coord(FIRST_PIECES[0].c2.r, FIRST_PIECES[0].c2.c),
                                  Coord(FIRST_PIECES[0].c3.r, FIRST_PIECES[0].c3.c),
                                  Coord(FIRST_PIECES[0].c4.r, FIRST_PIECES[0].c4.c)]

            for element in possible_placement:
                if element in state.keys():
                    return FIRST_PIECES[1]

            return FIRST_PIECES[0]

    def result(self, state, action: PlaceAction, colour: PlayerColor):
        """ Return the state that results from making a move from a state. """
        new_state = state.copy()
        new_state[action.c1] = colour
        new_state[action.c2] = colour
        new_state[action.c3] = colour
        new_state[action.c4] = colour

        # We need to check these rows and columns for any full lines
        rows = {action.c1.r, action.c2.r, action.c3.r, action.c4.r}
        cols = {action.c1.c, action.c2.c, action.c3.c, action.c4.c}

        row_duplicate = set()
        col_duplicate = set()

        for element in rows:
            duplicates = True

            for i in range(0, BOARD_N):
                if Coord(element, i) not in new_state.keys():
                    duplicates = False

            if duplicates:
                row_duplicate.add(element)

        for element in cols:
            duplicates = True

            for i in range(0, BOARD_N):
                if Coord(i, element) not in new_state.keys():
                    duplicates = False

            if duplicates:
                col_duplicate.add(element)

        # Remove the squares if we found duplicates
        for element in row_duplicate:
            for i in range(0, BOARD_N):
                if Coord(element, i) in new_state.keys():
                    new_state.pop(Coord(element, i))

        for element in col_duplicate:
            for i in range(0, BOARD_N):
                if Coord(i, element) in new_state.keys():
                    new_state.pop(Coord(i, element))

        return new_state

    def heuristic(self, state, prev_state):
        """ Return the value of this given state to player. """
        # calculate the hash table value for the current and previous state
        current = board_hash(state, self.hash_table)
        prev = board_hash(prev_state, self.hash_table)
        hash_str = str(current) + str(prev)
        hash_val = int(hash_str)

        if hash_val not in self.utility_values.keys():
            # haven't calculated before, cache and return
            value = utility_value(self, state, prev_state)
            self.utility_values[hash_val] = value
            return value
        else:
            # calculated previously, just return cached value
            return self.utility_values[hash_val]


def play_action(state, game):
    """ Determine the best action to play using our greedy agent and return it. """
    actions = game.actions(state, game.our_player)
    size = len(actions)
    # print(f'{size} moves available')

    if size > 1:
        # if we have more than one move, just play as a greedy agent and check two moves ahead for death states
        return select_best(state, game, actions, game.our_player)[0]
    elif size == 1:
        # only one move possible, return it immediately
        return actions[0]


def select_best(state, game, actions, player, select_amount=1):
    """ Returns the specified number of moves with the highest (or lowest) heuristic value. Also checks if the
    agent can play a "death state" in the next move (i.e. a move which will end the game). It will return a move
    which does not lead to a death state (if one exists) even if it is not the highest heuristic value. Returns as
    a list in case we want to return more than one highest move later on. """
    scores = {}
    currently_selected = 0
    final_selection = []

    # get the heuristic scores for all the moves
    for move in actions:
        scores[move] = game.heuristic(game.result(state, move, player), state)

    keys = sorted(scores, key=scores.get, reverse=True)

    # if we have more than 400 actions, just return the highest valued state
    # very unlikely to lead to a death state above this threshold
    if len(actions) > EVAL_THRESHOLD:
        return keys[0:select_amount]

    else:
        for action in keys:
            # check if this action leads to death
            death_action = is_state_good(game, state, action)

            # it does not, record this action
            if not death_action:
                final_selection.append(action)
                currently_selected += 1

            if currently_selected == select_amount:
                break

        if len(final_selection) == 0:
            # print("We give up XD")
            return keys[0:select_amount]
        else:
            # return the state which does not have any death moves
            return final_selection


def is_state_good(game, state, action):
    """ Lookahead two moves (i.e. next player moves, then we move) to check if the next player can end the game.
    Prune this state if this is true. """

    # simulate all moves the next player can do
    our_new_state = game.result(state, action, game.our_player)
    their_next_moves = game.actions(our_new_state, game.other_player)

    for their_move in their_next_moves:
        # simulate our next move which we can do
        their_new_state = game.result(our_new_state, their_move, game.other_player)
        our_next_moves = game.actions(their_new_state, game.our_player)

        # if we have no moves in this state, prune it from the list of available moves to play
        if len(our_next_moves) == 0:
            # print("Pruned")
            return True

    # not a death state, we can play this move with confidence
    return False


def utility_value(game: Game, state: dict[Coord, PlayerColor], old_state: dict[Coord, PlayerColor]):
    """ Calculates the heuristic value of a game state. """
    weight = 0

    # FEATURE 1
    # find the number of squares where we could possibly make a move
    our_moves = len(find_starting_positions(state, game.our_player))
    their_moves = len(find_starting_positions(state, game.other_player))
    weight += MOVE_WEIGHT * (our_moves - their_moves)

    # FEATURE 2
    # finding number of squares on board
    our_pieces, their_pieces = find_num_pieces(state, game.our_player)
    weight += PIECE_WEIGHT * (our_pieces - their_pieces)

    # FEATURE 3
    # finding change in piece counts since the last state
    our_prev_pieces, their_prev_pieces = find_num_pieces(old_state, game.our_player)
    our_loss = 0
    their_loss = 0

    our_player_change = our_pieces - our_prev_pieces
    other_player_change = their_pieces - their_prev_pieces

    if our_player_change < 0:
        our_loss = abs(our_player_change)

    if other_player_change < 0:
        their_loss = abs(other_player_change)

    weight += CHANGE_WEIGHT * (their_loss - our_loss)

    # FEATURE 4
    # finding length of lines on board and if they are too long or not
    row_len, col_len = line_lengths(state)
    num_high_len = 0
    num_low_len = 0

    row_weights = line_length_weight(row_len)
    num_high_len += row_weights[0]
    num_low_len += row_weights[1]

    col_weights = line_length_weight(col_len)
    num_high_len += col_weights[0]
    num_low_len += col_weights[1]

    weight += LINE_WEIGHT * (num_high_len - num_low_len)

    return weight
