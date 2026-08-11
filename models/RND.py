"""Random Network Distillation — an exploration bonus that generalizes.

Two convolutional networks over the same observation. The *target* has random
weights and is never trained. The *predictor* is trained to reproduce the
target's output on whatever states the agent actually visits. The squared error
between them is the novelty bonus:

    bonus(s) = ||predictor(s) - target(s)||^2

On a state seen a hundred times the predictor has fitted it and the error is
near zero. On a state never seen the predictor has nothing to go on and the
error is large. So the error behaves like a visit count that decays with
familiarity — except it is defined over *every* state, including ones never
visited, which a lookup table cannot do.

The random target is load-bearing. It is deterministic (so stochastic
transitions cannot inflate the bonus the way a next-state predictor's would —
the "noisy TV" failure), defined everywhere, and carries no structure to infer,
so the only way to lower error on an input is to have trained on that input.

Reference: Burda et al., "Exploration by Random Network Distillation" (2018).

Two normalizations are not optional. Without observation normalization the
random target's outputs are scale-dependent and the bonus is dominated by raw
pixel magnitude; without dividing the bonus by its own running standard
deviation there is no way to set `beta` that stays sane as training progresses,
because the raw error falls by orders of magnitude.
"""

from collections import deque
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Width of the embedding both networks produce. The bonus is the squared error
# over this vector, so it only needs to be wide enough that fitting it is
# non-trivial.
EMBED_DIM = 64

# Clip normalized observations to this many standard deviations. Stops a single
# outlier frame from producing an enormous bonus early on, when the running
# statistics are still based on very few samples.
OBS_CLIP = 5.0


class RunningNorm:
    """Streaming mean/variance (Welford), for observations and for the bonus."""

    def __init__(self, shape: tuple[int, ...] | None = None) -> None:
        self.mean = np.zeros(shape, dtype=np.float64) if shape else 0.0
        self.var = np.ones(shape, dtype=np.float64) if shape else 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> Any:
        return np.sqrt(self.var) + 1e-8


class RNDNovelty:
    """Intrinsic novelty signal over observations.

    Args:
        input_shape: (H, W, C) of the slice fed to RND — not necessarily the
            whole agent observation. See `planes`.
        learning_rate: Adam LR for the predictor.
        planes: which channels of the agent's observation to feed in. None
            (the default from main.py) uses all of them. Restricting to an
            adapter's HUD planes is tempting — they identify the room and carry
            no enemy motion — but degenerates: a HUD is nearly constant within a
            room, so RND sees about one state per room, fits it immediately and
            then pays nothing, which is `new_room` with extra steps. Measured on
            Zelda level1, per-decision bonus decayed 976x over 8 episodes on HUD
            planes against 62x on the full stack.
    """

    def __init__(self, input_shape: tuple[int, int, int], learning_rate: float = 0.0001,
                 planes: tuple[int, ...] | None = None,
                 train_batch: int = 32, train_interval: int = 4) -> None:
        self.planes = planes
        self.input_shape = input_shape
        self.target = self._build(trainable=False)
        self.predictor = self._build(trainable=True)
        self.optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.obs_norm = RunningNorm(input_shape)
        self.reward_norm = RunningNorm()
        self._train_step = self._build_train_step()
        self.train_batch = train_batch
        self.train_interval = train_interval
        self._buffer: deque = deque(maxlen=2000)
        self._since_train = 0

    def _build(self, trainable: bool) -> Any:
        # Deliberately shallower than the Q-network. The predictor only has to
        # tell states apart, not evaluate them, and a smaller net fits visited
        # states faster so the bonus decays on a useful timescale.
        model = Sequential([
            Conv2D(32, (8, 8), strides=4, activation='leaky_relu', input_shape=self.input_shape),
            Conv2D(64, (4, 4), strides=2, activation='leaky_relu'),
            Conv2D(64, (3, 3), strides=1, activation='leaky_relu'),
            Flatten(),
            Dense(EMBED_DIM),
        ])
        model.trainable = trainable
        return model

    def _build_train_step(self) -> Any:
        @tf.function(reduce_retracing=True)
        def step(obs: Any) -> Any:
            target = tf.stop_gradient(self.target(obs, training=False))
            with tf.GradientTape() as tape:
                pred = self.predictor(obs, training=True)
                loss = tf.reduce_mean(tf.square(pred - target))
            grads = tape.gradient(loss, self.predictor.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.predictor.trainable_variables, strict=True))
            return loss
        return step

    def _select(self, states: np.ndarray) -> np.ndarray:
        """Take the configured channel slice and drop any leading batch axis."""
        if states.ndim == 3:
            states = states[np.newaxis, ...]
        if self.planes is not None:
            states = states[..., list(self.planes)]
        return states.astype(np.float32)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        norm = (obs - self.obs_norm.mean) / self.obs_norm.std
        return np.clip(norm, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def bonus(self, state: np.ndarray, update_stats: bool = True) -> float:
        """Novelty of a single state, normalized by the running bonus std.

        Returned in units of "standard deviations of recent novelty", so `beta`
        means the same thing at episode 1 and episode 500 even though the raw
        error falls by orders of magnitude.
        """
        obs = self._select(state)
        if update_stats:
            self.obs_norm.update(obs)
        obs = self._normalize(obs)
        target = self.target(obs, training=False).numpy()
        pred = self.predictor(obs, training=False).numpy()
        raw = float(np.mean(np.square(pred - target)))
        if update_stats:
            self.reward_norm.update(np.array([raw]))
        return float(raw / self.reward_norm.std)

    def train(self, states: np.ndarray) -> float:
        """One gradient step fitting the predictor to the target on `states`."""
        obs = self._normalize(self._select(states))
        return float(self._train_step(tf.convert_to_tensor(obs)).numpy())

    def observe(self, state: np.ndarray) -> float | None:
        """Buffer a visited state and periodically fit the predictor to it.

        Batched rather than one gradient step per state: single-sample Adam
        updates on a conv net are noisy enough to make the bonus jitter, which
        shows up directly in the reward.

        Returns the predictor loss on the steps where it trains, else None.
        """
        self._buffer.append(np.asarray(state, dtype=np.uint8))
        if len(self._buffer) < self.train_batch:
            return None
        self._since_train += 1
        if self._since_train < self.train_interval:
            return None
        self._since_train = 0
        idx = np.random.randint(0, len(self._buffer), self.train_batch)
        batch = np.concatenate([self._buffer[i] for i in idx], axis=0)
        return self.train(batch)
