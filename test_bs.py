# Test cases for board stupid assignment
# Angela Kerlin
import unittest
from boardstupid import *

class TestBoardStupid(unittest.TestCase):

    def test_play_game(self):
        board = (
            tuple(0 for i in range(16)),
            (0,None,None,0,
            None,None,None,None,
            None,None,None,None,
            0,None,None,0),
            (0,None,None,0,
            None,None,None,None,
            None,None,None,None,
            0,None,None,0),
            (0,None,None,0,
            None,None,None,None,
            None,None,None,None,
            0,None,None,0),
        )
        play_game(board)

    def test_basic_understanding(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)

        self.assertEqual(state.moves, (0,1,2,3,4,5,6,8,9,10))
        winner = state.player
        state2 = state.traverse(0)
        self.assertEqual(state.moves, (0,1,2,3,4,5,6,8,9,10))
        self.assertEqual(state2.moves, (1,2,3,4,5,6,8,9,10))

        state2 = state2.traverse(4)
        state2 = state2.traverse(1)
        state2 = state2.traverse(5)
        state2 = state2.traverse(2)
        state2 = state2.traverse(6)
        state2 = state2.traverse(3)

        self.assertEqual(state2.util, winner)

    def test_node_get_children_no_expanded(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)

        ns = Node(state)
        children = [x[0] for x in ns.getChildrenUcb(2 ** 0.5)]
        self.assertEqual(len(children), 10)
        # print(children)
        for x in [0,1,2,3,4,5,6,8,9,10]:
            self.assertTrue(Node(state.traverse(x)) in children)

    def test_node_get_children_some_expanded(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)

        ns = Node(state)
        ns.expandedChildren = [Node(state.traverse(0)), Node(state.traverse(10)), Node(state.traverse(4))]

        children = [x[0] for x in ns.getChildrenUcb(2 ** 0.5)]
        self.assertEqual(len(children), 10)
        for x in [0,1,2,3,4,5,6,8,9,10]:
            self.assertTrue(Node(state.traverse(x)) in children)
    
    def test_mc_init(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)

        mc = MonteCarlo(state)
        children = mc.root.expandedChildren
        self.assertEqual(len(children), 10)
        for x in [0,1,2,3,4,5,6,8,9,10]:
            self.assertTrue(Node(state.traverse(x)) in children)

    def test_mc_init(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)

        ns = Node(state)
        child = Node(state.traverse(4))
        ns.expandChild(child)
        self.assertEqual(child, Node(state.traverse(4)))
        self.assertEqual(len(ns.expandedChildren), 1)

    def test_get_ucbs(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
    
        n1 = Node(state)
        n2 = Node(state.traverse(4))
        n1.expandChild(n2)
        bias = 2 ** 0.5

        self.assertEqual(n1.getUcb(0, bias), bias)
        self.assertEqual(n2.getUcb(0, bias), bias)

        n2.attempts = 5
        self.assertAlmostEqual(n2.getUcb(5, bias), 0.528760817)
        n2.wins = 3
        self.assertAlmostEqual(n2.getUcb(5, bias), 1.128760817)
    
    def test_get_max_of_node_list(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
    
        n1 = Node(state)
        n2 = Node(state.traverse(4))
        opts = [(n1, 3), (n2, 1)]
        m = max(opts, key=lambda item: item[1])
        self.assertEqual(m, (n1, 3))
    
    def test_expand_child(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        node = Node(state)
        child = Node(state.traverse(4))

        node.expandChild(child)
        self.assertTrue(child.isExpanded())
        self.assertEqual(len(node.expandedChildren), 1)
        self.assertEqual(node.expandedChildren, [child])

    def test_expand_child_twice(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        node = Node(state)
        child1 = Node(state.traverse(4))
        child2 = Node(state.traverse(4))
        
        node.expandChild(child1)
        node.expandChild(child2)
        self.assertEqual(len(node.expandedChildren), 2)
        self.assertEqual(node.expandedChildren, [child1, child2])

    def test_selectExpand1(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        mc = MonteCarlo(state)

        path = [mc.root]
        node = mc.selectExpand(mc.root, path)
        self.assertNotEqual(mc.root, node)
        # print(path)
        self.assertEqual(len(path), 3)
        # print(node)
    
    def test_simulate(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        mc = MonteCarlo(state)

        path = [mc.root]
        # only works until I make expansion random on equal max ucb
        node = mc.selectExpand(mc.root, path)
        util = mc.simulate(node, 0)
        self.assertTrue(util in [0,1,-1])

    def test_backprop(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0,    0,
                     0,    0,    0,    0,
                     0,    0,    0,    0),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        mc = MonteCarlo(state)

        path = [mc.root]
        # only works until I make expansion random on equal max ucb
        node = mc.selectExpand(mc.root, path)
        util = mc.simulate(node, 0)
        mc.backprop(path, util)
        # print(util)
        # print([x.state.player for x in path])
        if util == -1:
            self.assertEqual(path[0].wins, 1)
            self.assertEqual(path[0].attempts, 1)
            self.assertEqual(path[0].childrenAttempts, 1)
            self.assertEqual(path[1].wins, 0)
            self.assertEqual(path[1].attempts, 1)
            self.assertEqual(path[1].childrenAttempts, 1)
            self.assertEqual(path[2].wins, 1)
            self.assertEqual(path[2].attempts, 1)
            self.assertEqual(path[2].childrenAttempts, 0)
        elif util == 1:
            self.assertEqual(path[0].wins, 0)
            self.assertEqual(path[0].attempts, 1)
            self.assertEqual(path[0].childrenAttempts, 1)
            self.assertEqual(path[1].wins, 1)
            self.assertEqual(path[1].attempts, 1)
            self.assertEqual(path[1].childrenAttempts, 1)
            self.assertEqual(path[2].wins, 0)
            self.assertEqual(path[2].attempts, 1)
            self.assertEqual(path[2].childrenAttempts, 0)
        elif util == 0:
            tie = .4
            self.assertEqual(path[0].wins, tie)
            self.assertEqual(path[0].attempts, 1)
            self.assertEqual(path[0].childrenAttempts, 1)
            self.assertEqual(path[1].wins, tie)
            self.assertEqual(path[1].attempts, 1)
            self.assertEqual(path[1].childrenAttempts, 1)
            self.assertEqual(path[2].wins, tie)
            self.assertEqual(path[2].attempts, 1)
            self.assertEqual(path[2].childrenAttempts, 0)
        else:
            self.assertTrue(False)

    def test_set_selected1(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        mc = MonteCarlo(state)

        options = [(Node(state.traverse(x)), x) for x in [0,1,2,3,4,5,6,8,9,10]]
        count = 0
        for node in mc.root.expandedChildren:
            for opt in options:
                if node == opt[0]:
                    self.assertEqual(node.state.selected, opt[1])
                    count += 1
        self.assertEqual(count, 10)

    def test_set_selected2(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        mc = MonteCarlo(state)

        best = mc.root.expandedChildren[4]
        mc.root.childrenAttempts = 2
        best.wins = 100
        best.attempts = 2
        mc.setSelectedMove()
        self.assertEqual(mc.root.state.selected, best.state.selected)

    def test_maxRand(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        mc = MonteCarlo(state)

        opts = mc.root.getChildrenUcb(mc.bias)
        selected = []
        for _ in range(20):
            selected.append(maxRand(opts))
        # print([x.state.selected for x in selected])
        # all selected are not the same (extremely unlikely if working right)
        self.assertNotEqual(len([x for x in selected if x == selected[0]]), len(selected))

    def test_one_iter(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0,    0,
                     0,    0,    0,    0,
                     0,    0,    0,    0),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        mc = MonteCarlo(state)

        mc.monteCarloSearch()

        t = []
        newChild = [t + x.expandedChildren for x in mc.root.expandedChildren if len(x.expandedChildren) > 0]
        # print(newChild)
        self.assertEqual(len(newChild), 1)

        mc.setSelectedMove()
        self.assertNotEqual(mc.root.state.selected, -1)

    def test_iters(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0,    0,
                     0,    0,    0,    0,
                     0,    0,    0,    0),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        mc = MonteCarlo(state)

        mc.monteCarloSearch()
        mc.monteCarloSearch()
        mc.monteCarloSearch()

        t = []
        newChild = [t + x.expandedChildren for x in mc.root.expandedChildren if len(x.expandedChildren) > 0]
        # print(newChild)
        self.assertTrue(len(newChild) > 1)

        mc.setSelectedMove()
        self.assertNotEqual(mc.root.state.selected, -1)
        # print(mc.root.state.selected)

    def test_full_simple1(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                  None, None, None, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        state = state.traverse(0)
        state = state.traverse(5)
        state = state.traverse(1)
        state = state.traverse(6)
        state = state.traverse(2)
        state = state.traverse(4)
        mc = MonteCarlo(state)
        # print("STATE: ", state.display)

        mc.monteCarloSearch()
        mc.monteCarloSearch()
        mc.monteCarloSearch()
        mc.setSelectedMove()

        winner = [x for x in mc.root.expandedChildren if x.state.selected == 3][0]
        # print(winner)
        # print(winner.metaInfo())

        # print([(x[0].state.selected, x[1]) for x in mc.root.getChildrenWN()])
        self.assertEqual(state.selected, mc.root.state.selected)
        self.assertEqual(state.selected, 3)

    def test_full_simple2(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0, None,
                     0,    0,    0, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        state = state.traverse(0)
        state = state.traverse(5)
        state = state.traverse(1)
        state = state.traverse(6)
        state = state.traverse(2)
        state = state.traverse(7)
        mc = MonteCarlo(state)
        # print("STATE: ", state.display)

        count = 0
        for _ in range(10000):
            mc.monteCarloSearch()
            count += 1
        # print("count: ",count)
        mc.setSelectedMove()

        winner = [x for x in mc.root.expandedChildren if x.state.selected == 3][0]
        # print(winner)
        # print(winner.metaInfo())

        # print([(x[0].state.selected, x[1]) for x in mc.root.getChildrenWN()])
        self.assertEqual(state.selected, mc.root.state.selected)
        self.assertEqual(state.selected, 3)

    def given_test(self):
        board = ((0, 0, 0, 0,
            0, 0, None, None,
            0, None, 0, None,
            0, None, None, 0),) \
            + ((None,) * 16,) * 3
        state = GameState(board, 1)
        # print(state.display)
        find_best_move(state)
        self.assertEqual(state.selected, 0)
 
    def test_depth_check(self):
        board = ((   0,    0,    0,    0,
                     0,    0,    0,    0,
                  None, None, None, None,
                  None, None, None, None),) \
                + ((None,) * 16,) * 3
        state = GameState(board, 1)
        state = state.traverse(0)
        state = state.traverse(5)
        state = state.traverse(1)
        state = state.traverse(6)
        state = state.traverse(2)
        state = state.traverse(4)
        mc = MonteCarlo(state)

        newNode = Node(state.traverse(3).traverse(7))
        mc.root.expandedChildren[0].expandChild(newNode)
        path = [mc.root, mc.root.expandedChildren[0], newNode]
        # print(path, newNode)

        util = mc.simulate(newNode, 0)
        self.assertEqual(abs(util), 2)

        mc.backprop(path, util)
        mc.setSelectedMove()
        self.assertEqual(state.selected, mc.root.state.selected)
        self.assertEqual(state.selected, 3)



        


