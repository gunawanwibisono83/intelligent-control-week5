import gymnasium as gym
import numpy as np
from tensorflow import keras
import time

# Load environment dengan render
env = gym.make("CartPole-v1", render_mode="human")

# Load model hasil training
model = keras.models.load_model("dqn_cartpole.h5")

state_size = env.observation_space.shape[0]

def select_action(state):
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# Simulasi beberapa episode
for episode in range(1000):  # jalankan sesuai episode
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for t in range(500):
        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = np.reshape(next_state, [1, state_size])
        total_reward += reward

        if done:
            print(f"Simulasi Episode {episode+1} selesai. Skor: {total_reward}")
            break
        time.sleep(0.02)  # supaya animasi tidak terlalu cepat

env.close()
