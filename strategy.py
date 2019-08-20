from engine import TetrisEngine
import time
import random
import numpy as np
from dqn_agent import DQN
from torch.autograd import Variable
import torch
import pickle
from dqn_agent import DQN, ReplayMemory, Transition
import itertools


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


class Strategy:
    def run(self, width=10, height=20, simulation=False):
        engine = TetrisEngine(width, height)
        state = engine.clear()
        score = 0
        while True:
            action = self.get_action(engine.copy())
            if action is None:
                done = True
            else:
                state, reward, done = engine.step(action)
                score += int(reward)
                if not simulation:
                    print(engine)
                    print(action)
                    # time.sleep(.005)
            if done:
                if not simulation:
                    print('score {0}'.format(score))
                break
        return score

    def ave_score(self, width=10, height=20, n_sim=10000):
        scores = [self.run(simulation=True) for _ in range(n_sim)]
        return np.mean(scores)


class RandomStrategy(Strategy):
    def get_action(self, engine):
        actions = list(engine.value_action_map.keys())
        index = random.randrange(len(actions))
        return actions[index]


class Evaluator:
    def __init__(self):
        self._array = None

    def _calc_n_enclosed_squares(self, board):
        n_empty_from_top = 0
        for col in range(board.shape[1]):
            res = np.where(board[:, col] == 1)
            if res[0].size == 0:
                n_empty_from_top += board.shape[0]
            else:
                n_empty_from_top += res[0][0]
        n_filled = np.where(board == 1)[0].size
        return board.size - n_filled - n_empty_from_top

    def _get_num_squares_enclosed(self, board):
        n_enclosed = 0
        for row in reversed(range(board.shape[0])):
            if not np.any(board[row, :] == 1):
                break
            for col in range(board.shape[1]):
                if board[row, col] == 0:
                    if col + 1 == board.shape[1] or board[row, col + 1] == 1:
                        if col == 0 or board[row, col - 1] == 1:
                            n_enclosed += 1
        return n_enclosed

    def value(self, engine):
        if self._array is None:
            self._array = np.array([(engine.board.shape[1] - x - 1) ** 2 for x in range(engine.board.shape[1])])
        n_enclosed_squares = self._calc_n_enclosed_squares(engine.board.T)
        n_enclosed = self._get_num_squares_enclosed(engine.board.T)
        return -engine.board.sum(axis=0).dot(self._array) - 999999 * engine.n_deaths - 100 * n_enclosed_squares - 10 * n_enclosed


class MCStrategy(Strategy):
    def __init__(self, depth=10, n_sim=1000, evaluator=None):
        self._depth = depth
        self._n_sim = n_sim
        self._array = None
        self._evaluator = Evaluator() if evaluator is None else evaluator

    def _simulate(self, engine, action):
        random_strategy = RandomStrategy()
        values = []
        for _ in range(self._n_sim):
            sim_engine = engine.copy()
            sim_engine.step(action)
            for _ in range(self._depth):
                next_action = random_strategy.get_action(sim_engine)
                _, _, done = sim_engine.step(next_action)
                values.append(self._evaluator.value(sim_engine))
        return np.mean(values)

    def get_action(self, engine):
        actions = list(engine.value_action_map.keys())
        values = []
        for action in actions:
            value = self._simulate(engine, action)
            values.append(value)
        index = np.argmax(values)
        return actions[index]


class BeamSearchStrategy(Strategy):
    class Node:
        def __init__(self, engine, prev_actions):
            self.engine = engine.copy()
            self.engine.execute_action(prev_actions[-1])
            self.actions = prev_actions

        def get_initial_action(self):
            return self.actions[0].action

        def new_node(self, action):
            return BeamSearchStrategy.Node(self.engine, self.actions + [action])

        def get_possible_actions_at_next_step(self):
            player_actions = self.engine.get_player_actions()
            oponent_actions = self.engine.get_oponent_actions()
            return player_actions, oponent_actions

    def __init__(self, beam_width=5, depth=20, evaluator=None):
        self._beam_width = beam_width
        self._depth = depth
        self._evaluator = Evaluator() if evaluator is None else evaluator
        self._actions = []

    def _search(self, nodes, start_depth):
        final_nodes = []
        for depth in range(start_depth, self._depth):
            if not nodes:
                break
            new_nodes = []
            for node in nodes:
                player_actions, oponent_actions = node.get_possible_actions_at_next_step()
                if oponent_actions:
                    final_nodes.append(node)
                for action in player_actions:
                    new_nodes.append(node.new_node(action))
            nodes = new_nodes
            nodes = dict({tuple(node.engine.get_board_with_shape().reshape(-1)): node for node in nodes})
            nodes = list(nodes.values())
            random.shuffle(nodes)
            nodes = nodes[:self._beam_width]
        final_nodes = dict({tuple(node.engine.board.reshape(-1)): node for node in final_nodes})
        nodes = sorted(final_nodes.values(), key=lambda node: -self._evaluator.value(node.engine))
        return nodes

    def _drop_down(self, engine):
        n_row = engine.get_lowest_row_number_with_filled_square()
        drop_actions = []
        while engine.lowest_row_of_piece() + 8 < n_row:
            action = engine.execute_idle_action()  # idle -- drop down
            drop_actions.append(action)
        return drop_actions

    def get_action(self, engine):
        if not self._actions:
            drop_actions = self._drop_down(engine)
            nodes = []
            for action in engine.get_player_actions():
                node = BeamSearchStrategy.Node(engine, [action])
                nodes.append(node)
            nodes = self._search(nodes, 0)
            if nodes:
                self._actions = drop_actions + nodes[0].actions
        if self._actions:
            action = self._actions[0]
            self._actions = self._actions[1:]
            return action.action


class DQNModelStrategy(Strategy):
    def __init__(self, checkpoint_file_name="checkpoint.pth.tar"):
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'ReplayMemory':
                    return ReplayMemory
                return super().find_class(module, name)
        self._model = DQN()
        if use_cuda:
            self._model.cuda()
        pickle.Unpickler = CustomUnpickler
        checkpoint = torch.load(checkpoint_file_name, pickle_module=pickle)
        self._model.load_state_dict(checkpoint['state_dict'])

    def get_action(self, engine):
        state = FloatTensor(engine.board[None, None, :, :])
        action = self._model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1).type(LongTensor)
        return int(action[0, 0])


if __name__ == "__main__":
    # strategy = DQNModelStrategy()
    # print(strategy.ave_score(n_sim=100))
    strategy = BeamSearchStrategy(500, 2000)
    strategy.run()
    # print(strategy.ave_score(n_sim=100))
