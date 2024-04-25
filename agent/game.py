# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import itertools
import numpy as np

from agent.helpers import find_starting_positions, init_board
from agent.placement_algorithms import find_all_placements, PlacementProblem
from referee.game import PlayerColor, PlaceAction, Coord, BOARD_N

# We will always be able to place one of these two pieces on our first go
FIRST_PIECES = [PlaceAction(Coord(2, 3), Coord(2, 4), Coord(2, 5), Coord(2, 6)),
                PlaceAction(Coord(7, 5), Coord(7, 6), Coord(7, 7), Coord(7, 8))]
AB_CUTOFF = 6


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
        self.player = colour
        self.otherPlayer = (PlayerColor.BLUE if colour == PlayerColor.RED else PlayerColor.RED)
        self.first = True
        self.hash_table = init_board()

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
            rows = {FIRST_PIECES[0].c1.r, FIRST_PIECES[0].c2.r, FIRST_PIECES[0].c3.r, FIRST_PIECES[0].c4.r}
            cols = {FIRST_PIECES[0].c1.c, FIRST_PIECES[0].c2.c, FIRST_PIECES[0].c3.c, FIRST_PIECES[0].c4.c}
            same_row, same_col = False
            self.first = False

            for element in rows:
                if element in state.keys:
                    same_row = True
                    break

            for element in cols:
                if element in state.keys:
                    same_col = True
                    break

            if not (same_row and same_col):
                return FIRST_PIECES[0]
            return FIRST_PIECES[1]

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

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state, colour):
        """Return True if this is a final state for the game."""
        return not self.actions(state, colour)


def alpha_beta_cutoff_search(state, game, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a, game.player), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a, game.otherPlayer), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = cutoff_test or (lambda state, depth: depth > AB_CUTOFF or game.terminal_test(state))
    eval_fn = eval_fn or (lambda state: game.utility(state, game.player))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action
