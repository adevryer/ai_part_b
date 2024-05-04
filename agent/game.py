# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import itertools
import numpy as np

from .hashing import init_board, board_hash
from .helpers import find_num_pieces, line_lengths, line_length_weight, PIECE_WEIGHT, LINE_WEIGHT, CHANGE_WEIGHT
from .search_algorithms import PlacementProblem, find_all_placements, find_starting_positions
from referee.game import PlayerColor, PlaceAction, Coord, BOARD_N

# We will always be able to place one of these two pieces on our first go
FIRST_PIECES = [PlaceAction(Coord(2, 3), Coord(2, 4), Coord(2, 5), Coord(1, 4)),
                PlaceAction(Coord(7, 6), Coord(7, 7), Coord(7, 8), Coord(8, 7))]
AB_CUTOFF = 4
MAX_MOVES = 150
MAX_ACTIONS = 10


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
        self.our_player_moves = {}
        self.other_player_moves = {}

    def actions(self, state, colour):
        """Return a list of the allowable moves at this point."""
        if not self.first:
            """
            hash_val = board_hash(state, self.hash_table)
            if colour == self.our_player:
                if hash_val in self.our_player_moves:
                    return self.our_player_moves[hash_val]
            elif colour == self.other_player:
                if hash_val in self.other_player_moves:
                    return self.other_player_moves[hash_val]
            """

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

            #if colour == self.our_player:
            #    self.our_player_moves[hash_val] = place_actions
            #elif colour == self.other_player:
            #    self.other_player_moves[hash_val] = place_actions

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
    cutoff = int
    branch_factor = int

    def cutoff_test(state, depth, player):
        if depth > cutoff or game.terminal_test(state, player):
            return True
        return False

    # simulate move for our player, we want to maximise our outcome
    def max_value(state, prev_state, alpha, beta, depth):
        if cutoff_test(state, depth, game.our_player):
            return game.heuristic(state, prev_state)
        v = -np.inf

        for move in select_best(state, game, game.our_player, False, branch_factor):
            v = max(v, min_value(game.result(state, move, game.our_player), state, alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    # simulate move for the other player, they want to minimise our outcome
    def min_value(state, prev_state, alpha, beta, depth):
        if cutoff_test(state, depth, game.other_player):
            return game.heuristic(state, prev_state)
        v = np.inf

        for move in select_best(state, game, game.other_player, True, branch_factor):
            v = min(v, max_value(game.result(state, move, game.other_player), state, alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    best_score = -np.inf
    beta = np.inf
    best_action = None

    size = len(game.actions(state, game.our_player))
    print(f'{size} moves available')
    if size > 50:
        cutoff = 2
        branch_factor = 9
    elif 25 < size <= 50:
        cutoff = 4
        branch_factor = 6
    elif 1 < size <= 25:
        cutoff = 6
        branch_factor = 3
    elif size == 1:
        return game.actions(state, game.our_player)[0]

    # we play first, play all possible moves and start alpha beta pruning
    for a in select_best(state, game, game.our_player, False, branch_factor):
        v = min_value(game.result(state, a, game.our_player), state, best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a

    print(f"Value of: {game.heuristic(game.result(state, best_action, game.our_player), state)}")
    return best_action


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


"""
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
        new_state = state
        while not game.terminal_test(state, player) and not moves > MAX_MOVES:
            action = random.choice(list(game.actions(state, player)))
            new_state = game.result(state, action, player)
            player = game.our_player if player == game.other_player else game.other_player
            moves += 1

        v = game.utility(new_state, player, moves)
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
"""
