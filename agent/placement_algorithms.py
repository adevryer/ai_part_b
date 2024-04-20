# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

from referee.game import Direction

SEARCH_LIMIT = 4


class PlacementNode:
    """ A node for a potential placement of a square on the board to form a piece. Adapted and modified from the
    AIMA's Python Library function for a search tree node."""

    def __init__(self, state, parent=None):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.depth = self.parent.depth + 1 if parent else 0

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(next_state) for next_state in problem.actions(self.state, self.path())]

    def child_node(self, next_state):
        """Create a new child node for a possible placement action."""
        next_node = PlacementNode(next_state, self)
        return next_node

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node.state)
            node = node.parent
        return list(reversed(path_back))


class PlacementProblem:
    """ A class which defines the constructors and possible actions for finding all possible piece positions on the
    current game board. Adapted from AIMA's Python Library """

    def __init__(self, initial, current_map):
        """ Initialise the problem with the initial starting coordinate and the current game map """
        self.initial = initial
        self.current_map = current_map

    def actions(self, state, last_moves):
        """ Return the list of actions which can be achieved given the current state of the game board and the last
        moves done in parent search tree nodes"""
        # Pieces can be placed in four possible directions on the game board
        possible_moves = [state + Direction.Up, state + Direction.Down, state + Direction.Left,
                          state + Direction.Right]

        # Check previous squares in the current piece and add these to the possible placement list
        for element in last_moves:
            if element is not state:
                possible_moves.append(element + Direction.Up)
                possible_moves.append(element + Direction.Down)
                possible_moves.append(element + Direction.Left)
                possible_moves.append(element + Direction.Right)

        # Cannot be placed if a piece is already taking that space
        possible_moves = [element for element in possible_moves if element not in self.current_map]

        # Cannot be placed if we have already placed a piece there during this iteration of piece forming
        possible_moves = [element for element in possible_moves if element not in last_moves]

        return possible_moves


def find_all_placements(problem, limit=SEARCH_LIMIT):
    """ Recursively finds all possible combinations of pieces which can be placed from a given starting coordinate.
    Adapted from AIMA's Python Library function for depth-limited search"""

    placements = []

    def recursive_dls(node, curr_problem, curr_limit):
        if curr_limit == 1:
            # Reached 4 placements, add this piece to the list of possible placements
            # Sort to avoid adding duplicates
            new_path = sorted(node.path())
            if new_path not in placements:
                placements.append(new_path)

        else:
            # Find more possible places where we can place squares and search recursively from there
            for child in node.expand(curr_problem):
                if child not in node.path():
                    recursive_dls(child, curr_problem, curr_limit - 1)

    recursive_dls(PlacementNode(problem.initial, None), problem, limit)
    return placements
