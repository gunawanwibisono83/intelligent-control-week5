# ================================
# TRAINING DQN UNTUK PENDULUM-V1 (FINAL VERSION)
# ================================

# ---- Nonaktifkan log TensorFlow + aktifkan output real-time ----
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # sembunyikan log TensorFlow
sys.stdout.reconfigure(line_buffering=True)  # print real-time (tanpa buffering)

# ---- Import library utama ----
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt

# ---- Optimasi TensorFlow untuk Windows ----
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("Memulai proses training DQN Pendulum...\n", flush=True)

# ===============================
# 1️⃣ Inisialisasi environment & parameter
# ===============================
env = gym.make("Pendulum-v1", render_mode=None)  # non-render biar cepat
state_size = env.observation_space.shape[0]  # [cosθ, sinθ, θ_dot]
action_space = [-2.0, 0.0, 2.0]  # Diskretisasi aksi kontinu
action_size = len(action_space)

learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
episodes = 1000
memory = deque(maxlen=50000)

# ===============================
# 2️⃣ Model DQN
# ===============================
print("Membuat model DQN...", flush=True)
model = keras.Sequential([
    keras.Input(shape=(state_size,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(action_size, activation="linear")
])
model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate))

print("Model berhasil dibuat dan dikompilasi!", flush=True)

# --- Warm up model agar TensorFlow tidak freeze di episode pertama ---
dummy_state = np.zeros((1, state_size))
_ = model.predict(dummy_state, verbose=0)
print("Model warm-up selesai, siap training...\n", flush=True)

# ===============================
# 3️⃣ Fungsi pilih aksi (epsilon-greedy)
# ===============================
def select_action(state):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# ===============================
# 4️⃣ Training loop
# ===============================
reward_history = []

for episode in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for t in range(200):
        action_idx = select_action(state)
        action = [action_space[action_idx]]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = np.reshape(next_state, [1, state_size])

        # Normalisasi reward agar stabil
        reward = (reward + 16) / 16.0

        # Simpan pengalaman
        memory.append((state, action_idx, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Update model jika memori cukup
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for s, a, r, ns, d in minibatch:
                target = r
                if not d:
                    target += gamma * np.amax(model.predict(ns, verbose=0)[0])
                target_f = model.predict(s, verbose=0)
                target_f[0][a] = target
                model.fit(s, target_f, epochs=1, verbose=0)

        if done:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    reward_history.append(total_reward)
    print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}\n", flush=True)

# ===============================
# 5️⃣ Simpan model hasil training
# ===============================
model.save("dqn_pendulum.h5")
print("\nModel tersimpan sebagai 'dqn_pendulum.h5'", flush=True)

# ===============================
# 6️⃣ Plot hasil training
# ===============================
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward (Normalized)")
plt.title("DQN Training on Pendulum-v1 (Discrete Actions)")
plt.grid(True)
plt.tight_layout()
plt.savefig("dqn_pendulum_training_result.png")  # Simpan grafik ke file PNG
plt.show()

env.close()
print("\nTraining selesai. Grafik disimpan sebagai 'dqn_pendulum_training_result.png'")
