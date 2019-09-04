from engine import TetrisEngine
import random
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, width=10, height=20):
        self._engine = TetrisEngine(width, height)
        self._engine.clear()
        self._at_start = True

    def _search(self, nodes):
        final_nodes = []
        while True:
            if not nodes:
                break
            new_nodes = []
            for node in nodes:
                player_actions, oponent_actions = node.get_possible_actions_at_next_step()
                if oponent_actions:
                    final_nodes.append(node)
                for action in player_actions:
                    new_node = node.new_node(action)
                    if not new_node.died:
                        new_nodes.append(new_node)
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

    def get_actions(self):
        engine = self._engine.copy()
        drop_actions = self._drop_down(engine)
        nodes = []
        for action in engine.get_player_actions():
            node = TetrisGame.Node(engine, [action])
            if not node.died:
                nodes.append(node)
        nodes = self._search(nodes)
        actions = [TetrisGame.Action(drop_actions + node.actions, node) for node in nodes]
        return actions

    def execute_action(self, action):
        self._at_start = action._node.died
        self._engine = action._node.engine
        reward = -1 if action._node.died else 0
        return reward

    def at_start(self):
        return self._at_start

    def get_number_of_moves(self):
        return self._engine.time


class DQNetwork(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.lin1 = nn.Linear(768, 256)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))


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
        return random.sample(self.memory, batch_size)


class DQN:
    def __init__(self, buffer_size=100, gamma=0.99, epsilom=0.9):
        self._buffer_size = buffer_size
        self._gamma = gamma
        self._epsilon = epsilom

    def select_action(self, actions):
        if random.random() < self._epsilom:
            index = random.randint(0, len(actions))
        else:
            ratings = [(index, self._network(action.get_resulting_board())) for index, action in enumerate(actions)]
            ratings.sort(key=lambda x: x[1])
            index = ratings[0][0]
        return actions[index]

    def _train_on_minibatch(self):
        pass

    def train(self):
        self._memory = Memory(self._buffer_size)
        self._network = DQNetwork()
        self._game = TetrisGame()
        epoch = 0
        while True:
            n_moves = self._game.get_number_of_moves()
            epoch += 1
            actions = self._game.get_actions()
            action = self._select_action(actions)
            reward = self._game.execute_action(action)
            if self._game.at_start():
                print(n_moves)
            self._save(action, actions, reward)
            self._train_on_minibatch()


if __name__ == "__main__":
    model_file_path = "dqn_model.pickle"
    dqn = DQN(model_file_path)
    dqn.train()
