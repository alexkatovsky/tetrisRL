from engine import TetrisEngine
import time
import random
import numpy as np
# import dqn_agent
from torch.autograd import Variable
import torch
import pickle
# import dqn
import itertools
from multiprocessing import Pool
import functools
import os


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
                    # time.sleep(.05)
            if done:
                if not simulation:
                    print('score {0}'.format(score))
                break
        return score

    def ave_score(self, width=10, height=20, n_sim=10000):
        with Pool(10) as pool:
            scores = pool.map(functools.partial(self.run, 10, 20), [True] * n_sim)
        print(f'scores: {np.mean(scores)} ({scores})')
        return np.mean(scores)


class RandomStrategy(Strategy):
    def __init__(self, avoid_death=False):
        self._avoid_death = avoid_death

    def get_action(self, engine):
        actions = list(engine.value_action_map.keys())
        if self._avoid_death:
            new_actions = []
            for action in actions:
                if not engine.copy().execute_action(action):
                    new_actions.append(action)
            if new_actions:
                actions = new_actions
        index = random.randrange(len(actions))
        return actions[index]


class Evaluator:
    def __init__(self, params):
        self._array = None
        self._params = params

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
        p = self._params
        return -p[0] * engine.board.sum(axis=0).dot(self._array) - p[1] * engine.n_deaths - p[2] * n_enclosed_squares - p[3] * n_enclosed


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
            self.died = self.engine.execute_action(prev_actions[-1])
            self.actions = prev_actions

        def get_initial_action(self):
            return self.actions[0].action

        def new_node(self, action):
            return BeamSearchStrategy.Node(self.engine, self.actions + [action])

        def get_possible_actions_at_next_step(self):
            player_actions = self.engine.get_player_actions()
            oponent_actions = self.engine.get_oponent_actions()
            return player_actions, oponent_actions

    def __init__(self, beam_width=500, evaluator=None):
        self._beam_width = beam_width
        self._evaluator = Evaluator([1, 9999, 100, 10]) if evaluator is None else evaluator
        self._actions = []

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

    def get_action(self, engine):
        if not self._actions:
            drop_actions = self._drop_down(engine)
            nodes = []
            for action in engine.get_player_actions():
                node = BeamSearchStrategy.Node(engine, [action])
                if not node.died:
                    nodes.append(node)
            nodes = self._search(nodes)
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
                    return dqn_agent.ReplayMemory
                return super().find_class(module, name)
        self._model = dqn_agent.DQN()
        if use_cuda:
            self._model.cuda()
        pickle.Unpickler = CustomUnpickler
        checkpoint = torch.load(checkpoint_file_name, pickle_module=pickle)
        self._model.load_state_dict(checkpoint['state_dict'])

    def get_action(self, engine):
        state = FloatTensor(engine.board[None, None, :, :])
        action = self._model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1).type(LongTensor)
        return int(action[0, 0])


class DQNStrategy(Strategy):
    def __init__(self, model_file_path="dqn_model.pickle"):
        self._model = dqn.DQN.load_model(model_file_path)
        self._actions = []

    def get_action(self, engine):
        if not self._actions:
            actions = dqn.TetrisGame(engine=engine).get_actions()
            q_values = sorted([(action, self._model.evaluate(action)) for action in actions], key=lambda x: x[1])
            if q_values:
                action = q_values[-1][0]
                self._actions = action._actions
        if self._actions:
            action = self._actions[0]
            self._actions = self._actions[1:]
            return action.action


def smac_opt():
    # Import ConfigSpace and different types of parameters
    from smac.configspace import ConfigurationSpace
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter
    # Import SMAC-utilities
    from smac.scenario.scenario import Scenario
    from smac.facade.smac_bo_facade import SMAC4BO
    from smac.stats.stats import Stats
    from smac.runhistory.runhistory import RunHistory
    from smac.utils.io.traj_logging import TrajLogger

    def fun_to_optimize(x):
        params = [x[f'x{i}'] for i in range(0, 4)]
        print(f'params:{params}')
        strategy = BeamSearchStrategy(evaluator=Evaluator(params))
        ret = -strategy.ave_score(n_sim=10)
        print(ret)
        return ret

    cs = ConfigurationSpace()
    x0 = UniformFloatHyperparameter("x0", 0, 100, default_value=1)
    x1 = UniformFloatHyperparameter("x1", 0, 100, default_value=1)
    x2 = UniformFloatHyperparameter("x2", 0, 100, default_value=1)
    x3 = UniformFloatHyperparameter("x3", 0, 100, default_value=1)
    cs.add_hyperparameters([x0, x1, x2, x3])

    # Scenario object
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": 999999,   # max. number of function evaluations; for this example set to a low number
                         "cs": cs,               # configuration space
                         "deterministic": "false"
                         })

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = fun_to_optimize(cs.get_default_configuration())
    print("Default Value: %.2f" % def_value)

    old_output_dir = os.path.expanduser("smac3-output_2019-10-01_19:09:12_849207/run_1935803228/")

    rh_path = os.path.join(old_output_dir, "runhistory.json")
    runhistory = RunHistory(aggregate_func=None)
    runhistory.load_json(rh_path, scenario.cs)
    # ... stats, ...
    stats_path = os.path.join(old_output_dir, "stats.json")
    stats = Stats(scenario)
    stats.load(stats_path)
    # ... and trajectory.
    traj_path = os.path.join(old_output_dir, "traj_aclib2.json")
    trajectory = TrajLogger.read_traj_aclib_format(
        fn=traj_path, cs=scenario.cs)
    incumbent = trajectory[-1]["incumbent"]

    # Optimize, using a SMAC-object
    smac = SMAC4BO(scenario=scenario,
                   rng=np.random.RandomState(42),
                   tae_runner=fun_to_optimize,
                   runhistory=runhistory,
                   stats=stats,
                   restore_incumbent=incumbent,
                   )

    smac.optimize()


if __name__ == "__main__":
    # smac_opt()
    strategy = BeamSearchStrategy(evaluator=Evaluator([1.3737181751809944, 27.285650098591436, 99.9951369844601, 44.74908917400367]))
    strategy.run()
    # strategy.ave_score(n_sim=10)
