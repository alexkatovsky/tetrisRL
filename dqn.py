class DQN:
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
            reward, board = self._game.execute_action(action)
            if self._game.at_start():
                print(n_moves)
            self._save(action, actions, reward)
            self._train_on_minibatch()


if __name__ == "__main__":
    model_file_path = "dqn_model.pickle"
    dqn = DQN(model_file_path)
    dqn.train()
