from engine import TetrisEngine
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


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
        reward = -9999.0 if died else 0.5
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
        return random.sample(self.memory, min(batch_size, len(self.memory)))


class DQN:
    def __init__(self, model_file_path, buffer_size=500, gamma=0.99, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=200, batch_size=50):
        self._buffer_size = buffer_size
        self._gamma = gamma
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._batch_size = batch_size
        self._model_file_path = model_file_path

    def _q(self, action):
        board = action.get_resulting_board()
        state = torch.FloatTensor(board[None, None, :, :])
        value = self._network(state)
        return value[0][0]

    def _epsilon_threshold(self):
        threshold = self._epsilon_end + (self._epsilon_start - self._epsilon_end) * math.exp(-1. * self._n_games / self._epsilon_decay)
        print(threshold)
        return threshold

    def select_action(self, actions):
        if random.random() < self._epsilon_threshold():
            index = random.randint(0, len(actions) - 1)
        else:
            ratings = [(index, self._q(action)) for index, action in enumerate(actions)]
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
        #         max_q_value = max([self._q(action) for action in transition.possible_actions])
        #         target_value = target_value + self._gamma * max_q_value
        #     target_values.append(target_value.reshape(-1))
        #     q_values.append(self._q(transition.action).reshape(-1))
        # loss = F.smooth_l1_loss(torch.cat(q_values), torch.cat(target_values))
        non_final_mask = torch.ByteTensor(tuple(map(lambda t: bool(t.possible_actions), transitions)))
        states = []
        non_final_states = []
        for transition in transitions:
            board = transition.action.get_resulting_board()
            state = torch.FloatTensor(board[None, None, :, :])
            if transition.possible_actions:
                non_final_states.append(state)
            states.append(state)
        with torch.no_grad():
            non_final_next_states = torch.autograd.Variable(torch.cat(non_final_states), volatile=True)
            max_q_values = torch.autograd.Variable(torch.zeros(len(transitions)).type(torch.FloatTensor))
            max_q_values[non_final_mask] = self._network(non_final_next_states).max()
            rewards = torch.autograd.Variable(torch.cat([torch.FloatTensor([t.reward]) for t in transitions]))
            target_valuess = (max_q_values * self._gamma) + rewards
        q_values = self._network(torch.autograd.Variable(torch.cat(states)))
        loss = F.smooth_l1_loss(q_values, target_valuess)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def train(self):
        self._n_games = 0
        self._memory = Memory(self._buffer_size)
        self._network = DQNetwork()
        self._optimizer = optim.RMSprop(self._network.parameters(), lr=.01)
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


if __name__ == "__main__":
    model_file_path = "dqn_model.pickle"
    dqn = DQN(model_file_path)
    dqn.train()
