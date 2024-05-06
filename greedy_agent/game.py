# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import itertools

from agent.hashing import init_board, board_hash
from agent.utility_calculators import find_num_pieces, line_lengths, line_length_weight, PIECE_WEIGHT, LINE_WEIGHT, \
    CHANGE_WEIGHT, MOVE_WEIGHT
from agent.search_algorithms import PlacementProblem, find_all_placements, find_starting_positions
from referee.game import PlayerColor, PlaceAction, Coord, BOARD_N

# We will always be able to place one of these two pieces on our first go
FIRST_PIECES = [PlaceAction(Coord(2, 3), Coord(2, 4), Coord(2, 5), Coord(1, 4)),
                PlaceAction(Coord(7, 6), Coord(7, 7), Coord(7, 8), Coord(8, 7))]


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def __init__(self, colour: PlayerColor):
        """The constructor specifies the initial board."""
        self.our_player = colour
        self.other_player = (PlayerColor.BLUE if colour == PlayerColor.RED else PlayerColor.RED)
        self.first = True
        self.hash_table = init_board()
        self.utility_values = {}

    def actions(self, state, colour):
        """Return a list of the allowable moves at this point."""
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
        """Return the state that results from making a move from a state."""
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
        """Return the value of this final state to player."""
        current = board_hash(state, self.hash_table)
        prev = board_hash(prev_state, self.hash_table)
        hash_str = str(current) + str(prev)
        hash_val = int(hash_str)

        if hash_val not in self.utility_values.keys():
            value = utility_value(self, state, prev_state)
            self.utility_values[hash_val] = value
            return value
        else:
            return self.utility_values[hash_val]


def greedy_agent(state, game):
    return select_best(state, game, game.our_player, False, 1)[0]


def select_best(state, game, player, is_min, select_amount):
    actions = game.actions(state, player)
    scores = {}

    for move in actions:
        scores[move] = game.heuristic(game.result(state, move, player), state)

    if is_min:
        keys = sorted(scores, key=scores.get)
    else:
        keys = sorted(scores, key=scores.get, reverse=True)

    return keys[0:select_amount]


def utility_value(game: Game, state: dict[Coord, PlayerColor], old_state: dict[Coord, PlayerColor]):
    weight = 0

    our_moves = len(find_starting_positions(state, game.our_player))
    their_moves = len(find_starting_positions(state, game.other_player))
    weight += MOVE_WEIGHT * (our_moves - their_moves)

    """
    hash_val = board_hash(state, game.hash_table)
    if hash_val in game.our_player_moves.keys():
        our_moves = game.our_player_moves[hash_val]
    else:
        our_moves = len(game.actions(state, game.our_player))

    if hash_val in game.other_player_moves.keys():
        their_moves = game.other_player_moves[hash_val]
    else:
        their_moves = len(game.actions(state, game.other_player))

    weight += MOVE_WEIGHT * (our_moves - their_moves)
    """

    # finding number of squares on board
    our_pieces, their_pieces = find_num_pieces(state, game.our_player)
    weight += PIECE_WEIGHT * (our_pieces - their_pieces)

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

    """
    our_squares = find_starting_positions(state, game.our_player)
    their_squares = find_starting_positions(state, game.other_player)

    # find the holes with a size less than 4 on the board
    holes = []
    num_holes = 0
    all_positions = our_squares + their_squares
    for position in all_positions:
        current_holes = find_holes(PlacementProblem(position, state))
        for element in current_holes:
            holes.append(element)

    # Remove any duplicate states from the list
    holes = list(holes for holes, _ in itertools.groupby(holes))
    num_holes = len(holes)
    weight += HOLE_WEIGHT * num_holes
    """

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
