import numpy as np

from models.DQN import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN. Identical to DQNAgent except for how the next-state value is
    bootstrapped: the main network selects the best next action and the target
    network evaluates it, which reduces the overestimation bias of vanilla DQN.

        a* = argmax_a Q_main(s', a)
        y  = r + gamma * Q_target(s', a*)
    """

    def _bootstrap_values(self, next_states: np.ndarray) -> np.ndarray:
        # Select the best next action with the main network ...
        main_q_next = self.model(next_states, training=False).numpy()
        best_actions = np.argmax(main_q_next, axis=1)
        # ... and evaluate it with the target network.
        target_q_next = self.target_model(next_states, training=False).numpy()
        return target_q_next[np.arange(len(best_actions)), best_actions]
