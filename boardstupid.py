# Name:         Angela Kerlin
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Board Stupid
# Term:         Summer 2021

import math
import random
from typing import Callable, Generator, Optional, Tuple, List


class GameState:

    def __init__(self, board: Tuple[Tuple[Optional[int], ...], ...],
                 player: int) -> None:
        """
        An instance of GameState has the following attributes.

            player: Set as either 1 (MAX) or -1 (MIN).
            moves: A tuple of integers representing empty indices of the board.
            selected: The index that the current player believes to be their
                      optimal move; defaults to -1.
            util: The utility of the board; either 1 (MAX wins), -1 (MIN wins),
                  0 (tie game), or None (non-terminal game state).
            traverse: A callable that takes an integer as its only argument to
                      be used as the index to apply a move on the board,
                      returning a new GameState object with this move applied.
                      This callable provides a means to traverse the game tree
                      without modifying parent states.
            display: A string representation of the board, which should only be
                     used for debugging and not parsed for strategy.

        In addition, instances of GameState may be stored in hashed
        collections, such as sets or dictionaries.

        >>> board = ((   0,    0,    0,    0,   \
                         0,    0, None, None,   \
                         0, None,    0, None,   \
                         0, None, None,    0),) \
                    + ((None,) * 16,) * 3

        >>> state = GameState(board, 1)
        >>> state.util
        None
        >>> state.player
        1
        >>> state.moves
        (0, 1, 2, 3, 4, 5, 8, 10, 12, 15)
        >>> state = state.traverse(0)
        >>> state.player
        -1
        >>> state.moves
        (1, 2, 3, 4, 5, 8, 10, 12, 15)
        >>> state = state.traverse(5)
        >>> state.player
        1
        >>> state.moves
        (1, 2, 3, 4, 8, 10, 12, 15)
        >>> state = state.traverse(1)
        >>> state.player
        -1
        >>> state.moves
        (2, 3, 4, 8, 10, 12, 15)
        >>> state = state.traverse(10)
        >>> state.player
        1
        >>> state.moves
        (2, 3, 4, 8, 12, 15)
        >>> state = state.traverse(2)
        >>> state.player
        -1
        >>> state.moves
        (3, 4, 8, 12, 15)
        >>> state = state.traverse(15)
        >>> state.player
        1
        >>> state.moves
        (3, 4, 8, 12)
        >>> state = state.traverse(3)
        >>> state.util
        1
        """
        self.player: int = player
        self.moves: Tuple[int] = GameState._get_moves(board, len(board))
        self.selected: int = -1
        self.util: Optional[int] = GameState._get_utility(board, len(board))
        self.traverse: Callable[[int], GameState] = \
            lambda index: GameState._traverse(board, len(board), player, index)
        self.display: str = GameState._to_string(board, len(board))
        self.keys: Tuple[int, ...] = tuple(hash(single) for single in board)

    def __eq__(self, other: "GameState") -> bool:
        return self.keys == other.keys

    def __hash__(self) -> int:
        return hash(self.keys)

    @staticmethod
    def _traverse(board: Tuple[Tuple[Optional[int], ...], ...],
                  width: int, player: int, index: int) -> "GameState":
        """
        Return a GameState instance in which the board is updated at the given
        index by the current player.

        Do not call this method directly; instead, call the |traverse| instance
        attribute, which only requires an index as an argument.
        """
        i, j = index // width ** 2, index % width ** 2
        single = board[i][:j] + (player,) + board[i][j + 1:]
        return GameState(board[:i] + (single,) + board[i + 1:], -player)

    @staticmethod
    def _get_moves(board: Tuple[Tuple[Optional[int], ...], ...],
                   width: int) -> Tuple[int]:
        """
        Return a tuple of the unoccupied indices remaining on the board.
        """
        return tuple(j + i * width ** 2 for i, single in enumerate(board)
                     for j, square in enumerate(single) if square == 0)

    @staticmethod
    def _get_utility(board: Tuple[Tuple[Optional[int], ...], ...],
                     width: int) -> Optional[int]:
        """
        Return the utility of the board; either 1 (MAX wins), -1 (MIN wins),
        0 (tie game), or None (non-terminal game state).
        """
        for line in GameState._iter_lines(board, width):
            if line == (1,) * width:
                return 1
            if line == (-1,) * width:
                return -1
        return 0 if len(GameState._get_moves(board, width)) == 0 else None

    @staticmethod
    def _iter_lines(board: Tuple[Tuple[Optional[int], ...], ...],
                    width: int) -> Generator[Tuple[int], None, None]:
        """
        Iterate over all groups of indices that represent a winning condition.
        X lines are row-wise, Y lines are column-wise, and Z lines go through
        all single boards; combinations of these axes refer to the direction
        of the line in 2D or 3D space.
        """
        for single in board:
            # x lines (2D rows)
            for i in range(0, len(single), width):
                yield single[i:i + width]
            # y lines (2D columns)
            for i in range(width):
                yield single[i::width]
            # xy lines (2D diagonals)
            yield single[::width + 1]
            yield single[width - 1:len(single) - 1:width - 1]
        # z lines
        for i in range(width ** 2):
            yield tuple(single[i] for single in board)
        for j in range(width):
            # xz lines
            yield tuple(board[i][j * width + i] for i in range(len(board)))
            yield tuple(board[i][j * width + width - 1 - i]
                        for i in range(len(board)))
            # yz lines
            yield tuple(board[i][j + i * width] for i in range(len(board)))
            yield tuple(board[i][-j - 1 - i * width]
                        for i in range(len(board)))
        # xyz lines
        yield tuple(board[i][i * width + i] for i in range(len(board)))
        yield tuple(board[i][i * width + width - 1 - i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - width * (i + 1) + i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - (i * width) - i - 1]
                    for i in range(len(board)))

    @staticmethod
    def _to_string(board: Tuple[Tuple[Optional[int], ...], ...],
                   width: int) -> str:
        """
        Return a string representation of the game board, in which integers
        represent the indices of empty spaces and the characters "X" and "O"
        represent previous move selections for MAX and MIN, repsectively.
        """
        display = "\n"
        for i in range(width):
            for j in range(width):
                line = board[j][i * width:i * width + width]
                start = j * width ** 2 + i * width
                for k, space in enumerate(line):
                    if space == 0:
                        space = start + k
                    else:
                        space = ("X" if space == 1
                                 else "O" if space == -1
                                 else "-")
                    display += "{0:>4}".format(space)
                display += " " * width
            display += "\n"
        return display




class Node:
    # Represents one node in the monte carlo search tree

    def __init__(self, state, expanded: bool = False, move=-1):
        state.selected = move
        self.state = state
        self.expandedChildren = []
        self.wins = 0
        self.attempts = 0
        self.childrenAttempts = 0 # sum of all children's attempts
        self.expanded = expanded

    def getChildrenUcb(self, bias: float) -> "list[Tuple[Node, float]]":
        """
        Returns all the children of this node (expanded and not expanded) and
        its corresponding UCB as a tuple
        """
        # If its a leaf, return None? TODO
        children = [] # TODO CHANGE TO SET
        doneStates = set()
        for child in self.expandedChildren:
            # copy expanded children, and get their UCB's
            children.append((child, child.getUcb(self.childrenAttempts, bias)))
            doneStates.add(child.state)

        # speedup -> if len children ==  len moves just return
        moves = self.state.moves
        if len(doneStates) == len(moves):
            return children

        for move in moves:
            # for each possible move, generate a new child if that direction was
            #   not already expanded
            state = self.state.traverse(move)
            if state not in doneStates:
                node = Node(state)
                children.append((node,
                    node.getUcb(self.childrenAttempts, bias)))
        return children

    def getChildrenWN(self) -> "list[Tuple[Node, float]]":
        """
        Returns all the expanded children of this node and each
        corresponding win/attempt ratio as a tuple
        """
        # children = []
        # for child in self.expandedChildren:
        #     # copy expanded children, and get their W/N's
        #     children.append((child, child.getWinAttemptRatio()))
        # return children
        return [(x, x.getWinAttemptRatio()) for x in self.expandedChildren]

    def getChildren(self) -> "list[Node]":
        """
        Returns all the children of this node (expanded and not expanded)
        """
        children = [x for x in self.expandedChildren]
        doneMoves = [x.state for x in self.expandedChildren]#moves already taken
        
        for move in self.state.moves:
            state = self.state.traverse(move)
            if state not in doneMoves:
                children.append(Node(state))
        
        return children
    
    def expandChild(self, node: "Node"):
        node.expanded = True
        self.expandedChildren.append(node)
    # def expandChild(self, index):
        # """
        # Given the index to explore, expand a child permanantly and return it
        # """
        # child = Node(self.state.traverse(index), True)
        # self.expandedChildren.append(child)
        # return child
    
    def rootInit(self) -> None:
        """
        Expand all the children of the root node
        """
        for move in self.state.moves:
            self.expandedChildren.append(Node(self.state.traverse(move), True,
                move=move))

    def getUcb(self, totAttempts: int, bias: float) -> float:
        """
        Calculates the UCB of this node and returns it
        If no attempts have been made on this node, return bias
        """
        if self.attempts == 0:
            return bias
        return (self.wins / self.attempts)\
            + bias * math.sqrt(math.log10(totAttempts) / self.attempts)
    
    def getWinAttemptRatio(self) -> float:
        """
        Returns this node's win/attempt ratio
        """
        if self.attempts == 0:
            return 0
        return self.wins / self.attempts
    
    def isExpanded(self):
        """
        Returns true if this node is permanently expanded to its parent, false
        if it is not
        """
        return self.expanded

    def __eq__(self, other) -> bool:
        return self.state == other.state
    
    def __repr__(self) -> str:
        return self.state.display
    
    def metaInfo(self) -> str:
        return "move " + str(self.state.selected)\
            + " wins: " + str(self.wins)\
            + " attempts: " + str(self.attempts)\
            + " ratio: " + str(self.getWinAttemptRatio())




class MonteCarlo:

    def __init__(self, state: GameState):
        self.root = Node(state)    # the root of the search tree
        self.root.rootInit()       # explore the root's children
        self.player = state.player # player to optimize
        self.bias = 2 ** 0.5       # exploration bias of the search

    def setSelectedMove(self) -> None:
        """
        Finds best move given current mc tree and sets it in the root's
        selected variable
        """
        kids = self.root.getChildrenWN()
        best = maxRand(kids) # get highest win/attempts child
        self.root.state.selected = best.state.selected

    def monteCarloSearch(self) -> None:
        """
        Performs one iteration of monte carlo search
        """
        path = [self.root]
        newNode = self.selectExpand(self.root, path)
        # print("path: ", path) #dd
        util = self.simulate(newNode)
        self.backprop(path, util)

    def selectExpand(self, node: Node, path: "list[Node]") -> Node:
        """
        Selects the next node to expand upon, expands it, and returns this new
        node
        """
        kids = node.getChildrenUcb(self.bias) # get a list of Node/UCB tuples
        if len(kids) == 0:
            # node has no children, select this parent
            return node
        best = maxRand(kids) # get highest UCB child
        if not best.isExpanded():
            path.append(best)
            node.expandChild(best) # make this child permanant to the parent
            return best
        path.append(best)
        return self.selectExpand(best, path)

    def simulate(self, node: Node) -> bool:
        """
        Simulates a random route to a terminating state and returns the win or
        loss with a true for win and false for loss
        """
        if node.state.util is not None:
            # print(node) #dd
            return node.state.util
        kids = node.getChildren()
        child = random.choice(kids)
        return self.simulate(child)

    def backprop(self, affected: "list[Node]", util: int) -> None:
        """
        Given a list of affected nodes and a utility to represent who won,
        Update the wins/attempts/childrenAttempts of each node
        """
        # handle special case of last node (no childrenAttempts increment)
        node = affected[-1]
        node.attempts += 1
        if node.state.player == util * -1:
            node.wins += 1
        elif util == 0:
            node.wins += 0.25

        i = len(affected) - 2
        while i >= 0:
            # traverse list in backwards order, updating each node
            node = affected[i]
            node.attempts += 1
            node.childrenAttempts += 1
            if node.state.player == util * -1:
                # if player represented on this level won, increment wins
                node.wins += 1
            elif util == 0:
                # weight tie as half a win
                node.wins += 0.25
            i -= 1




def find_best_move(state: GameState) -> None:
    """
    Search the game tree for the optimal move for the current player, storing
    the index of the move in the given GameState object's selected attribute.
    The move must be an integer indicating an index in the 3D board - ranging
    from 0 to 63 - with 0 as the index of the top-left space of the top board
    and 63 as the index of the bottom-right space of the bottom board.

    This function must perform a Monte Carlo Tree Search to select a move,
    using additional functions as necessary. During the search, whenever a
    better move is found, the selected attribute should be immediately updated
    for retrieval by the instructor's game driver. Each call to this function
    will be given a set number of seconds to run; when the time limit is
    reached, the index stored in selected will be used for the player's turn.
    """
    mc = MonteCarlo(state)
    count = 0
    while True:
    # for x in range(1000):
        mc.monteCarloSearch()
        if count % 10 == 0:
            mc.setSelectedMove()
        count += 1
    # print([x.metaInfo() for x in mc.root.getChildren()])

def maxRand(opts: List[Tuple[Node, float]]) -> Node:
    """
    Given a list of nodes and their corresponding metrics', return a random node
    that has the max metric of the set
    """
    maxMet = max([x[1] for x in opts]) # get highest metric of list
    maxOpts = [x[0] for x in opts if x[1] == maxMet]
    return random.choice(maxOpts)






def main() -> None:
    pass




def play_game(board: Tuple[Tuple[int, ...]]) -> None:
    """
    Play a game of 3D Tic-Tac-Toe with the computer.

    If you lose, you lost to a machine.
    If you win, your implementation was bad.
    You lose either way.
    """
    # board = tuple(tuple(0 for _ in range(i, i + 16))
    #               for i in range(0, 64, 16))
    state = GameState(board, 1)
    while state.util is None:
        # human move
        print(state.display)
        state = state.traverse(int(input("Move: ")))
        if state.util is not None:
            break
        # computer move
        find_best_move(state)
        move = (state.selected if state.selected != -1
                else random.choice(state.moves))
        state = state.traverse(move)
    print(state.display)
    if state.util == 0:
        print("Tie Game")
    else:
        print(f"Player {state.util} Wins!")


if __name__ == "__main__":
    main()
