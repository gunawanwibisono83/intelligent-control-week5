import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt

# Setup environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)

# Bangun model DQN
model = keras.Sequential([
    keras.Input(shape=(state_size,)),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(action_size, activation="linear")
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

scores = []

# Training loop
for episode in range(1000):  # bisa diganti jumlah episode sesuai device
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time_step in range(500):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            scores.append(total_reward)
            print(f"Episode: {episode+1}, Score: {total_reward}, Epsilon: {epsilon:.4f}")
            break

    # Training minibatch
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for s, a, r, ns, d in minibatch:
            target = r if d else r + gamma * np.amax(model.predict(ns, verbose=0)[0])
            target_f = model.predict(s, verbose=0)
            target_f[0][a] = target
            model.fit(s, target_f, epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Save model
model.save("dqn_cartpole.h5")
print("✅ Training selesai, model tersimpan sebagai dqn_cartpole.h5")

# Plot skor training
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Performa DQN CartPole (Training)")
plt.show()

env.close()
