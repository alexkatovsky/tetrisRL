from engine import TetrisEngine
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pickle
import numpy as np


class TetrisGame:
    class Action:
        def __init__(self, actions, node):
            self._actions = actions
            self._node = node

        def get_resulting_board(self):
            return self._node.engine.board

    class Node:
        def __init__(self, engine, prev_actions):
            self.engine = engine.copy()
            self.died = self.engine.execute_action(prev_actions[-1])
            self.actions = prev_actions

        def get_initial_action(self):
            return self.actions[0].action

        def new_node(self, action):
            return TetrisGame.Node(self.engine, self.actions + [action])

        def get_possible_actions_at_next_step(self):
            player_actions = self.engine.get_player_actions()
            oponent_actions = self.engine.get_oponent_actions()
            return player_actions, oponent_actions

    def __init__(self, width=10, height=20, engine=None):
        if engine is None:
            self._engine = TetrisEngine(width, height)
        else:
            self._engine = engine
        self._at_start = True

    def _search(self, nodes):
        final_nodes = []
        while True:
            if not nodes:
                break
            new_nodes = []
            for node in nodes:
                if node.died:
                    final_nodes.append(node)
                else:
                    player_actions, oponent_actions = node.get_possible_actions_at_next_step()
                    if oponent_actions:
                        final_nodes.append(node)
                    for action in player_actions:
                        new_node = node.new_node(action)
                        if new_node.died:
                            final_nodes.append(new_node)
                        else:
                            new_nodes.append(new_node)
            nodes = new_nodes
            nodes = dict({tuple(node.engine.get_board_with_shape().reshape(-1)): node for node in nodes})
            nodes = list(nodes.values())
        final_nodes = dict({tuple(node.engine.board.reshape(-1)): node for node in final_nodes})
        return list(final_nodes.values())

    def _drop_down(self, engine):
        n_row = engine.get_lowest_row_number_with_filled_square()
        drop_actions = []
        while engine.lowest_row_of_piece() + 8 < n_row:
            action = engine.execute_idle_action()  # idle -- drop down
            drop_actions.append(action)
        return drop_actions

    def get_actions(self):
        engine = self._engine.copy()
        drop_actions = self._drop_down(engine)
        nodes = []
        for action in engine.get_player_actions():
            node = TetrisGame.Node(engine, [action])
            nodes.append(node)
        nodes = self._search(nodes)
        actions = [TetrisGame.Action(drop_actions + node.actions, node) for node in nodes]
        return actions

    def end_game_on_no_actions(self):
        self._at_start = True
        reward = -9999.0
        self._engine.clear()
        return reward

    def _score(self):
        return 1 / (1 - self._engine.get_lowest_row_number_with_filled_square() / self._engine.height) ** 2

    def execute_action(self, action):
        self._engine = action._node.engine
        if action._node.died:
            died = True
            self._engine.clear()
        else:
            player_actions, oponent_actions = action._node.get_possible_actions_at_next_step()
            assert(player_actions == [])
            died = self._engine.execute_action(random.sample(oponent_actions, 1)[0])
        self._at_start = died
        reward = -99.0 if died else self._score()
        return reward

    def at_start(self):
        return self._at_start

    def get_number_of_moves(self):
        return self._engine.time


class DQNetwork(nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.lin1 = nn.Linear(768, 256)
        self.head = nn.Linear(256, 1)

    def get_parameters(self):
        return self.parameters()

    def get_state(self, board):
        return torch.FloatTensor(board[None, None, :, :])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))


class DQNFeedForwardNetwork(nn.Module):
    def __init__(self, dims):
        super(DQNFeedForwardNetwork, self).__init__()
        self._layers = []
        for dim1, dim2 in zip(dims[:-1], dims[1:]):
            self._layers.append(nn.Linear(dim1, dim2))

    def get_parameters(self):
        parameters = []
        for fc in self._layers:
            parameters += list(fc.parameters())
        return parameters

    def get_state(self, board):
        return torch.FloatTensor(np.concatenate(board).reshape(1, -1))

    def forward(self, x):
        for layer in self._layers[:-1]:
            x = F.relu(layer(x))
        x = self._layers[-1](x)
        return x


class Memory(object):
    class Transition:
        def __init__(self, action, possible_actions, reward):
            self.action = action
            self.possible_actions = possible_actions
            self.reward = reward

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Memory.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))


class DQN:
    def __init__(self, model_file_path='dqn_model.pickle', buffer_size=1000, gamma=0.99, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=500, batch_size=64):
        self._model_file_path = model_file_path
        self._buffer_size = buffer_size
        self._gamma = gamma
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._batch_size = batch_size

    def _get_state(self, board):
        return self._network.get_state(board)

    def evaluate(self, action):
        board = action.get_resulting_board()
        state = self._get_state(board)
        value = self._network(state)
        return float(value[0][0])

    def _save_model(self):
        pickle.dump(self, open(self._model_file_path, 'bw+'))

    @staticmethod
    def load_model(model_file_path):
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'Memory':
                    from dqn import Memory
                    return Memory
                if name == 'TetrisGame':
                    from dqn import TetrisGame
                    return TetrisGame
                if name == 'DQNetwork':
                    from dqn import DQNetwork
                    return DQNetwork
                if name == 'DQNFeedForwardNetwork':
                    from dqn import DQNFeedForwardNetwork
                    return DQNFeedForwardNetwork
                if name == 'DQNLin':
                    from dqn import DQNLin
                    return DQNLin
                return super().find_class(module, name)
        pickle_data = CustomUnpickler(open(model_file_path, 'rb')).load()
        return pickle_data

    def _epsilon_threshold(self):
        threshold = self._epsilon_end + (self._epsilon_start - self._epsilon_end) * math.exp(-1. * self._n_games / self._epsilon_decay)
        print(threshold)
        return threshold

    def select_action(self, actions):
        if random.random() < self._epsilon_threshold():
            index = random.randint(0, len(actions) - 1)
        else:
            ratings = [(index, self.evaluate(action)) for index, action in enumerate(actions)]
            ratings.sort(key=lambda x: x[1])
            index = ratings[0][0]
        return actions[index]

    def _train_on_minibatch(self):
        transitions = self._memory.sample(self._batch_size)
        # target_values = []
        # q_values = []
        # for transition in transitions:
        #     target_value = torch.Tensor([transition.reward])
        #     if transition.possible_actions:
        #         max_q_value = max([self.evaluate(action) for action in transition.possible_actions])
        #         target_value = target_value + self._gamma * max_q_value
        #     target_values.append(target_value.reshape(-1))
        #     q_values.append(self.evaluate(transition.action).reshape(-1))
        # loss = F.smooth_l1_loss(torch.cat(q_values), torch.cat(target_values))
        states = []
        for transition in transitions:
            board = transition.action.get_resulting_board()
            state = self._get_state(board)
            states.append(state)
        with torch.no_grad():
            max_q_values = []
            for transition in transitions:
                if transition.possible_actions:
                    next_states = torch.cat([self._get_state(action.get_resulting_board()) for action in transition.possible_actions])
                    q_values = self._network(next_states)
                    max_q_values.append(q_values.max().reshape(-1))
                else:
                    max_q_values.append(torch.FloatTensor([0]))
            max_q_values = torch.cat(max_q_values)
            rewards = torch.autograd.Variable(torch.cat([torch.FloatTensor([t.reward]) for t in transitions]))
            target_values = (max_q_values * self._gamma) + rewards
        q_values = self._network(torch.autograd.Variable(torch.cat(states)))
        loss = F.smooth_l1_loss(q_values, target_values)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def train(self):
        self._n_games = 0
        self._memory = Memory(self._buffer_size)
        self._network = self._create_network()
        self._optimizer = optim.RMSprop(self._network.get_parameters(), lr=.01)
        self._game = TetrisGame()
        epoch = 0
        while True:
            n_moves = self._game.get_number_of_moves()
            epoch += 1
            possible_actions = self._game.get_actions()
            action = self.select_action(possible_actions)
            reward = self._game.execute_action(action)
            print(self._game._engine)
            if self._game.at_start():
                self._n_games += 1
                print(n_moves)
            print(reward)
            self._memory.push(action, possible_actions, reward)
            self._train_on_minibatch()
            if epoch % 100 == 0:
                self._save_model()


class DQNConv(DQN):
    def __init__(self, *args, **kwargs):
        DQN.__init__(self, *args, **kwargs)

    def _create_network(self):
        return DQNetwork()


class DQNLin(DQN):
    def __init__(self, *args, **kwargs):
        DQN.__init__(self, *args, **kwargs)

    def _create_network(self):
        return DQNFeedForwardNetwork([200, 100, 20, 1])


if __name__ == "__main__":
    dqn = DQNLin(model_file_path='DQNFeedForwardNetwork.pkl')
    dqn.train()
