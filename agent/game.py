# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import itertools

from agent.helpers import find_starting_positions
from agent.placement_algorithms import find_all_placements, PlacementProblem
from referee.game import PlayerColor, PlaceAction, Coord, BOARD_N

# We will always be able to place one of these two pieces on our first go
first_pieces = [PlaceAction(Coord(2, 3), Coord(2, 4), Coord(2, 5), Coord(2, 6)),
                PlaceAction(Coord(7, 5), Coord(7, 6), Coord(7, 7), Coord(7, 8))]


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
            rows = {first_pieces[0].c1.r, first_pieces[0].c2.r, first_pieces[0].c3.r, first_pieces[0].c4.r}
            cols = {first_pieces[0].c1.c, first_pieces[0].c2.c, first_pieces[0].c3.c, first_pieces[0].c4.c}
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
                return first_pieces[0]
            return first_pieces[1]

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

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)
