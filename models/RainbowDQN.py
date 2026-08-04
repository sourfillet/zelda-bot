from collections import deque

import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class SumTree:
    """
    Binary SumTree for O(log n) priority updates and proportional sampling.

    Internal array of size 2*capacity - 1:
      - Indices [0, capacity-2]:        internal nodes holding sums
      - Indices [capacity-1, 2*capacity-2]: leaves holding priorities
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Iteratively propagate a priority change up to the root."""
        while idx > 0:
            parent = (idx - 1) // 2
            self.tree[parent] += change
            idx = parent

    def _retrieve(self, idx, s):
        """Walk the tree from idx to find the leaf whose cumulative priority covers s."""
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Experience replay buffer with proportional prioritization (PER).

    Transitions with higher TD errors are sampled more often.
    Importance-sampling weights correct for the resulting bias.
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha          # Prioritization exponent (0=uniform, 1=full)
        self.beta = beta            # IS correction exponent (anneals toward 1.0)
        self.beta_increment = beta_increment
        self.epsilon = epsilon      # Minimum priority to prevent zero
        self._max_priority = 1.0

    def __len__(self):
        return self.tree.n_entries

    def add(self, transition):
        """Store transition with the current maximum priority."""
        self.tree.add(self._max_priority, transition)

    def sample(self, batch_size):
        """
        Sample batch_size transitions proportionally to their priorities.

        Returns:
            batch      — list of stored (s, a, r, s', done) tuples
            indices    — SumTree indices (needed for priority updates)
            is_weights — importance-sampling weights (float32 array)
        """
        batch = []
        indices = []
        priorities = []

        total = self.tree.total
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError(
                f"PER SumTree total={total!r}. The tree has been corrupted by a "
                "non-finite priority, which is caused by inf/NaN TD errors. "
                "Check for Q-value divergence or unnormalized model inputs."
            )

        segment = total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(max(priority, self.epsilon))

        sampling_probs = np.array(priorities, dtype=np.float64) / self.tree.total
        is_weights = (len(self) * sampling_probs) ** (-self.beta)
        is_weights /= is_weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)
        return batch, indices, is_weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """Recompute priorities from TD errors and update the SumTree."""
        for idx, td_error in zip(indices, td_errors, strict=False):
            err = abs(float(td_error))
            # Guard: a non-finite TD error (inf/NaN from Q-value overflow) must
            # not corrupt the tree. Fall back to the current max priority so the
            # transition is still sampled frequently until a valid error is computed.
            if not np.isfinite(err):
                err = self._max_priority ** (1.0 / self.alpha)
            priority = (err + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self._max_priority = min(max(self._max_priority, priority), 10.0)


class RainbowDQNAgent:
    """
    Rainbow DQN agent for single-room combat training.

    Improvements over vanilla DQN:
      - Dueling network head  (separate V(s) and A(s,a) streams)
      - Prioritized experience replay  (PER, alpha=0.6)
      - n-step returns  (n=3, reduces temporal credit-assignment lag)
      - Soft (Polyak) target updates  (tau=0.0005, per training step —
        an effective target-sync timescale of ~2000 steps)
      - Huber loss  (robust to large TD errors in early training)
      - Double DQN target  (decouples action selection from evaluation)

    Matches the DQNAgent interface so main.py requires only a new branch.
    """

    def __init__(self, input_shape, action_size, learning_rate, discount_factor,
                 epsilon, epsilon_decay, epsilon_min):
        self.input_shape = input_shape  # (height, width, stacked_frames), e.g. (84, 84, 4)
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=20000)
        self.batch_size = 32
        self.train_start = 1000

        # n-step return accumulator (one deque per episode; cleared on done)
        self.n_step = 3
        self.n_step_buffer: deque = deque()

        # Soft target update coefficient (Polyak)
        self.tau = 0.0005

        self.train_counter = 0

        # Build main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        # Hard-copy weights to initialize target network
        self.target_model.set_weights(self.model.get_weights())

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_model(self):
        """
        Dueling DQN via Keras functional API.

        Shared convolutional backbone → split into value and advantage streams:
            Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
        """
        inp = Input(shape=self.input_shape)
        # Normalize uint8 frames [0, 255] → [0, 1] inside the model so that
        # states can be stored as uint8 in the replay buffer (4× less memory)
        # while Q-values stay in a numerically stable range during training.
        x = tf.keras.layers.Rescaling(1.0 / 255.0)(inp)
        x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(x)
        x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)

        # Value stream: V(s)
        v = Dense(256, activation='relu')(x)
        v = Dense(1, activation='linear')(v)

        # Advantage stream: A(s, a)
        a = Dense(256, activation='relu')(x)
        a = Dense(self.action_size, activation='linear')(a)

        # Combine: subtract mean advantage so that V is identifiable
        # keras.ops.mean works on symbolic KerasTensors (Keras 3 / TF 2.18+)
        q = v + a - keras.ops.mean(a, axis=1, keepdims=True)

        model = Model(inputs=inp, outputs=q)
        model.compile(
            loss=tf.keras.losses.Huber(delta=2.0),
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        )
        return model

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def update_target_model(self):
        """
        Hard copy of main → target weights.
        Called by main.py at startup and after loading a saved model.
        Internal soft updates happen via _soft_update_target() inside train().
        """
        self.target_model.set_weights(self.model.get_weights())

    def _soft_update_target(self):
        """Polyak averaging: target = tau * main + (1 - tau) * target."""
        for main_w, target_w in zip(self.model.weights, self.target_model.weights, strict=False):
            target_w.assign(self.tau * main_w + (1.0 - self.tau) * target_w)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state):
        """
        Epsilon-greedy action selection.
        Returns a one-hot encoded action vector (matches DQNAgent interface).
        """
        if np.random.rand() <= self.epsilon:
            action_index = np.random.randint(self.action_size)
        else:
            q_values = self.model(state, training=False).numpy()
            action_index = int(np.argmax(q_values[0]))

        action = np.zeros(self.action_size, dtype=int)
        action[action_index] = 1
        return action

    # ------------------------------------------------------------------
    # n-step bookkeeping
    # ------------------------------------------------------------------

    def _store_n_step(self):
        """
        Pop the oldest transition from n_step_buffer, compute the discounted
        n-step return, and store (s_t, a_t, R_n, s_{t+n}, done_n, steps) in the
        prioritized replay buffer.

        Stops reward accumulation early if a done=True is encountered within
        the window, correctly handling episode boundaries.

        `steps` is how many rewards actually went into R_n, which is not always
        n_step: flushing at the end of an episode drains partial windows of
        length n_step-1, n_step-2, ... Those transitions bootstrap from a state
        that is `steps` frames ahead, not n_step, so train() has to discount
        each sample by gamma**steps individually.
        """
        buf = list(self.n_step_buffer)
        R_n = 0.0
        final_ns = buf[-1][3]
        done_n = buf[-1][4]
        steps = len(buf)

        for i, (_, _, r, ns, d) in enumerate(buf):
            R_n += (self.discount_factor ** i) * r
            if d:
                final_ns = ns
                done_n = True
                steps = i + 1
                break

        s_t, a_t = buf[0][0], buf[0][1]
        self.memory.add((s_t, a_t, R_n, final_ns, done_n, steps))
        self.n_step_buffer.popleft()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, state, action, reward, next_state, done):
        """
        Accumulate n-step transitions, then train from the prioritized buffer.

        Steps:
          1. Convert action to index, append to n_step_buffer.
          2. Store n-step transition when buffer reaches n.
          3. On episode end, flush remaining partial transitions.
          4. Sample a mini-batch with IS weights.
          5. Compute Double-DQN targets with n-step discounting.
          6. Update replay priorities from TD errors.
          7. Fit with IS weights as sample_weight (corrects PER bias).
          8. Soft-update the target network.

        Returns the scalar training loss, or None if training hasn't started.
        """
        # Normalise action representation
        if isinstance(action, np.ndarray) and action.shape == (self.action_size,):
            action_index = int(np.argmax(action))
        else:
            action_index = int(action)

        self.n_step_buffer.append((state, action_index, reward, next_state, done))

        # Store once the window is full
        if len(self.n_step_buffer) >= self.n_step:
            self._store_n_step()

        # Flush remaining partial windows at episode end
        if done:
            while len(self.n_step_buffer) > 0:
                self._store_n_step()

        # Wait for enough samples
        if len(self.memory) < self.train_start:
            return None

        # Sample with importance-sampling weights
        batch, indices, is_weights = self.memory.sample(self.batch_size)

        states = np.vstack([t[0] for t in batch])
        next_states = np.vstack([t[3] for t in batch])
        actions = [t[1] for t in batch]
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        # Per-sample window length; partial windows flushed at episode end are
        # shorter than n_step and must not be discounted as if they were full.
        n_steps = np.array([t[5] for t in batch], dtype=np.float32)

        # Current Q-values (reference for building the full target vector)
        current_q = self.model(states, training=False).numpy()
        # Double DQN: select best action with main network ...
        main_q_next = self.model(next_states, training=False).numpy()
        # ... evaluate that action with the target network
        target_q_next = self.target_model(next_states, training=False).numpy()

        gamma_n = self.discount_factor ** n_steps
        targets = current_q.copy()
        td_errors = np.zeros(self.batch_size, dtype=np.float32)

        for i in range(self.batch_size):
            if dones[i]:
                target_val = rewards[i]
            else:
                best_action = int(np.argmax(main_q_next[i]))
                target_val = rewards[i] + gamma_n[i] * target_q_next[i][best_action]
            td_errors[i] = abs(target_val - current_q[i][actions[i]])
            targets[i][actions[i]] = target_val

        # Refresh priorities before the gradient step
        self.memory.update_priorities(indices, td_errors)

        # Fit; IS weights correct for the non-uniform sampling distribution
        history = self.model.fit(
            states, targets,
            epochs=1, verbose=0,
            sample_weight=is_weights
        )

        # Soft target update every training step (no hard copy every N steps)
        self.train_counter += 1
        self._soft_update_target()

        return history.history['loss'][0]

    # ------------------------------------------------------------------
    # Epsilon and persistence
    # ------------------------------------------------------------------

    def flush_episode(self):
        """Flush any remaining transitions in the n-step buffer.

        Call this at the end of every episode, even when the episode ends by
        hitting max_frames rather than a terminal state, to prevent stale
        transitions from bleeding into the next episode's n-step calculation.
        """
        while len(self.n_step_buffer) > 0:
            self._store_n_step()

    def update_epsilon(self):
        """Decay the exploration rate by the configured multiplier."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save the main Q-network weights."""
        self.model.save(filepath)
