from engine import TetrisEngine
import time
import random
import numpy as np
from dqn_agent import DQN
from torch.autograd import Variable
import torch
import pickle
from dqn_agent import DQN, ReplayMemory, Transition


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
            state, reward, done = engine.step(action)
            score += int(reward)
            if not simulation:
                print(engine)
                print(action)
                time.sleep(.1)
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
    strategy = DQNModelStrategy()
    print(strategy.ave_score(n_sim=100))
