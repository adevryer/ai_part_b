# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import itertools
import random
from random import sample
import numpy as np

from agent.hashing import init_board, board_hash
from agent.helpers import (find_num_pieces, line_lengths, line_length_weight, MOVE_WEIGHT, PIECE_WEIGHT, LINE_WEIGHT,
                           HOLE_WEIGHT)
from agent.search_algorithms import (PlacementProblem, find_all_placements, find_starting_positions,
                                     find_one_placement, find_holes)
from referee.game import PlayerColor, PlaceAction, Coord, BOARD_N

# We will always be able to place one of these two pieces on our first go
FIRST_PIECES = [PlaceAction(Coord(2, 3), Coord(2, 4), Coord(3, 3), Coord(3, 4)),
                PlaceAction(Coord(7, 6), Coord(7, 7), Coord(8, 7), Coord(8, 8))]
AB_CUTOFF = 4
MAX_MOVES = 150


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

            # Only return maximum of 50 PlaceActions to reduce branching factor
            if len(place_actions) > 50:
                place_actions = sample(place_actions, 50)

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

    def heuristic(self, state):
        """Return the value of this final state to player."""
        hash_val = board_hash(state, self.hash_table)
        if hash_val not in self.utility_values.keys():
            value = utility_value(self, state)
            self.utility_values[hash_val] = value
            return value
        else:
            #print("Being used yayayayay!!!!!")
            return self.utility_values[hash_val]

    def utility(self, state, player, num_moves):
        our_pieces, their_pieces = find_num_pieces(state, player)

        # need to fix this a bit
        if self.terminal_test(state, player):
            return 1
        elif num_moves > MAX_MOVES:
            if our_pieces > their_pieces:
                return 1
            elif our_pieces < their_pieces:
                return -1
            else:
                return 0

    def terminal_test(self, state, colour):
        """Return True if this is a final state for the game."""
        return not self.actions(state, colour)


def alpha_beta_cutoff_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    def cutoff_test(state, depth, player):
        if depth > AB_CUTOFF or game.terminal_test(state, player):
            return True
        return False

    # simulate move for our player, we want to maximise our outcome
    def max_value(state, alpha, beta, depth):
        # print("now max")
        # print(f"DEPTH = {depth}")
        if cutoff_test(state, depth, game.our_player):
            return game.heuristic(state)
        v = -np.inf
        actions = game.actions(state, game.our_player)

        for move in actions:
            v = max(v, min_value(game.result(state, move, game.our_player), alpha, beta, depth + 1))
            if v >= beta:
                # print("Max Pruned")
                return v
            alpha = max(alpha, v)
        return v

    # simulate move for the other player, they want to minimise our outcome
    def min_value(state, alpha, beta, depth):
        # print("now min")
        # print(f"DEPTH = {depth}")
        if cutoff_test(state, depth, game.other_player):
            return game.heuristic(state)
        v = np.inf
        actions = game.actions(state, game.other_player)

        for move in actions:
            v = min(v, max_value(game.result(state, move, game.other_player), alpha, beta, depth + 1))
            if v <= alpha:
                # print("Min Pruned")
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    best_score = -np.inf
    beta = np.inf
    best_action = None

    print("Now evaluating all moves...")
    action = game.actions(state, game.our_player)
    print(len(action))
    for a in action:
        game.heuristic(game.result(state, a, game.our_player))
    print("Done!")

    # we play first, play all possible moves and start alpha beta pruning
    for a in game.actions(state, game.our_player):
        v = min_value(game.result(state, a, game.our_player), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


def utility_value(game: Game, state: dict[Coord, PlayerColor]):
    # finding available squares for player moves
    our_squares = find_starting_positions(state, game.our_player)
    their_squares = find_starting_positions(state, game.other_player)
    our_moves = len(our_squares)
    their_moves = len(their_squares)

    """
    for position in our_squares:
        if find_one_placement(PlacementProblem(position, state)):
            our_moves += 1

    for position in their_squares:
        if find_one_placement(PlacementProblem(position, state)):
            their_moves += 1
    """

    # finding number of squares on board
    our_pieces, their_pieces = find_num_pieces(state, game.our_player)

    # finding length of lines on board and if they are too long or not
    row_len, col_len = line_lengths(state)
    num_good_len = 0
    num_bad_len = 0

    row_weights = line_length_weight(row_len)
    num_good_len += row_weights[0]
    num_bad_len += row_weights[1]

    col_weights = line_length_weight(col_len)
    num_good_len += col_weights[0]
    num_bad_len += col_weights[1]
    """
    # find the holes with a size less than 4 on the board
    holes = []
    all_positions = our_squares + their_squares
    for position in all_positions:
        current_holes = find_holes(PlacementProblem(position, state))
        for element in current_holes:
            holes.append(element)

    # Remove any duplicate states from the list
    holes = list(holes for holes, _ in itertools.groupby(holes))
    """
    num_holes = 0  #len(holes)

    weight = (MOVE_WEIGHT * (our_moves - their_moves) + PIECE_WEIGHT * (our_pieces - their_pieces) +
              LINE_WEIGHT * (num_good_len - num_bad_len) + HOLE_WEIGHT * num_holes)

    # print(weight)
    return weight


class MCT_Node:
    # Node in the Monte Carlo search tree, keeps track of the children states.
    def __init__(self, state, game, num_moves=0, parent=None, parent_action=None, U=0, N=0):
        # self.__dict__.update(parent=parent, state=state, U=U, N=N)
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.player = game.our_player if parent is None else (game.other_player if parent == game.our_player
                                                              else game.our_player)
        self.num_moves = num_moves
        self.U = U
        self.N = N
        self.children = {}
        # self.actions = None


def ucb(n, C=1.4):
    return np.inf if n.N == 0 else n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N)


def monte_carlo_tree_search(state, game, N=1000):
    def select(n):
        # select a leaf node in the tree
        if n.children:
            return select(max(n.children.keys(), key=ucb))
        else:
            return n

    def expand(n):
        # expand the leaf node by adding all its children states
        if not n.children and not game.terminal_test(n.state, n.player):
            n.children = {MCT_Node(game=game, state=game.result(n.state, action), parent=n, parent_action=action):
                              action for action in game.actions(n.state)}
        return select(n)

    def simulate(game, state, first_player, num_moves):
        # simulate the utility of current state by random picking a step
        player = first_player
        moves = num_moves
        while not game.terminal_test(state, player) and not moves > MAX_MOVES:
            action = random.choice(list(game.actions(state, player)))
            state = game.result(state, action, player)
            player = game.our_player if player == game.other_player else game.other_player
            moves += 1

        v = game.utility(state, player, moves)
        return -v

    def backprop(n, utility):
        # passing the utility back to all parent nodes
        if utility > 0:
            n.U += utility
        # draw state
        if utility == 0:
             n.U += 0.5
        n.N += 1
        if n.parent:
            backprop(n.parent, -utility)

    root = MCT_Node(state=state, game=game)

    for i in range(N):
        leaf = select(root)
        child = expand(leaf)
        result = simulate(game, child.state, child.player)
        backprop(child, result)

    max_state = max(root.children, key=lambda p: p.N)

    return root.children.get(max_state)
