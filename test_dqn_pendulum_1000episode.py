# ==============================================
# FINAL TESTING DQN UNTUK PENDULUM-V1 (CLEAN VERSION)
# ==============================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # sembunyikan log TensorFlow

import gymnasium as gym
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import time

# ==============================================
# 1️⃣ Load Model DQN
# ==============================================
model_path = r"C:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M5\Assigment\dqn_pendulum_1000episode.h5"

print("Memuat model DQN hasil training...\n")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ File model tidak ditemukan di:\n{model_path}\nPastikan file .h5 hasil training sudah tersedia.")
else:
    print(f"✅ File model ditemukan di:\n{model_path}")
    model = keras.models.load_model(model_path, compile=False)
    print("✅ Model berhasil dimuat tanpa kompilasi ulang.\n")

# ==============================================
# 2️⃣ Inisialisasi Environment
# ==============================================
env = gym.make("Pendulum-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_space = [-2.0, 0.0, 2.0]
action_size = len(action_space)

# ==============================================
# 3️⃣ Fungsi Pemilihan Aksi (Policy)
# ==============================================
def select_action(state):
    """Memilih aksi berdasarkan Q-value tertinggi dari model DQN"""
    q_values = model.predict(state, verbose=0)
    action_idx = np.argmax(q_values[0])
    return action_space[action_idx]

# ==============================================
# 4️⃣ Pengujian Model (Reward per Episode)
# ==============================================
test_episodes = 1000
reward_history = []
epsilon = 0.05  # nilai eksplorasi rendah untuk testing

print("Memulai simulasi pengujian model...\n")

for episode in range(test_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    # Jalankan simulasi per episode
    for t in range(200):
        action = [select_action(state)]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = np.reshape(next_state, [1, state_size])

        if done:
            break
        time.sleep(0.02)  # animasi lebih halus

    reward_history.append(total_reward)
    print(f"✅Episode {episode+1}/{test_episodes} selesai | Total Reward: {total_reward:.2f}")

env.close()

# ==============================================
# 5️⃣ Analisis Hasil Testing
# ==============================================
print("\nHASIL PENGUJIAN MODEL DQN:")
print(f"Rata-rata Reward : {np.mean(reward_history):.2f}")
print(f"Reward Terbaik   : {np.max(reward_history):.2f}")
print(f"Reward Terendah  : {np.min(reward_history):.2f}")

# ==============================================
# 6️⃣ Visualisasi Grafik Reward per Episode
# ==============================================
plt.figure(figsize=(8,5))
plt.plot(reward_history, marker='o', linewidth=1.8, color='red', label="DQN Testing Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Pendulum Testing Performance (Per Episode)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dqn_pendulum_testing_result.png")
plt.show()

print("\nrafik disimpan sebagai 'dqn_pendulum_testing_result.png'")

# (Opsional) Simpan hasil untuk perbandingan multi-metode
np.save("dqn_rewards.npy", reward_history)
