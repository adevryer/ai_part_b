# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import itertools
import numpy as np

from agent.helpers import find_starting_positions
from agent.placement_algorithms import find_all_placements, PlacementProblem
from referee.game import PlayerColor, Action, PlaceAction, Coord, BOARD_N


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Tetress game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.
        match self._color:
            case PlayerColor.RED:
                print("Testing: RED is playing a PLACE action")
                return PlaceAction(
                    Coord(3, 3), 
                    Coord(3, 4), 
                    Coord(4, 3), 
                    Coord(4, 4)
                )
            case PlayerColor.BLUE:
                print("Testing: BLUE is playing a PLACE action")
                return PlaceAction(
                    Coord(2, 3), 
                    Coord(2, 4), 
                    Coord(2, 5), 
                    Coord(2, 6)
                )

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after an agent has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        # There is only one action type, PlaceAction
        place_action: PlaceAction = action
        c1, c2, c3, c4 = place_action.coords

        # Here we are just printing out the PlaceAction coordinates for
        # demonstration purposes. You should replace this with your own logic
        # to update your agent's internal game state representation.
        print(f"Testing: {color} played PLACE action: {c1}, {c2}, {c3}, {c4}")


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def __init__(self, initial):
        """The constructor specifies the initial board."""
        self.initial = initial

    def actions(self, state, colour):
        """Return a list of the allowable moves at this point."""
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

    def result(self, state, action: PlaceAction):
        """Return the state that results from making a move from a state."""
        new_state = state.copy()
        new_state[action.c1] = PlayerColor.RED
        new_state[action.c2] = PlayerColor.RED
        new_state[action.c3] = PlayerColor.RED
        new_state[action.c4] = PlayerColor.RED

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

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)
