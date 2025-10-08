# ==============================================
# FINAL TESTING PID CONTROLLER UNTUK PENDULUM-V1
# ==============================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# ==============================================
# 1️⃣ Inisialisasi Environment
# ==============================================
env = gym.make("Pendulum-v1", render_mode="human")

# (opsional) Tingkatkan batas torsi agar swing-up lebih mudah
env.unwrapped.max_torque = 8.0  

print("Memulai simulasi kontrol PID pada Pendulum-v1...\n")

# ==============================================
# 2️⃣ Parameter Kontrol
# ==============================================
g = 10.0      # gravitasi
l = 1.0       # panjang pendulum

# Energy-based swing-up gain
k_E = 4.5
E_des = g * l  # energi target di posisi tegak

# PID stabilizer (untuk menjaga tegak di atas)
Kp = 45.0
Ki = 0.02
Kd = 8.0
integral_limit = 1.0

# Threshold switching
enter_theta = 0.12    # ambil alih PID jika |θ| < 0.12 rad
exit_theta  = 0.25    # kembali ke swing-up jika keluar dari zona ini
enter_thdot = 0.5
exit_thdot  = 1.0

# ==============================================
# 3️⃣ Fungsi Kontrol
# ==============================================
def energy_control(theta, theta_dot):
    """Kontrol energi untuk swing-up"""
    E = 0.5 * (theta_dot**2) + g * (1 - np.cos(theta))
    tau = -k_E * theta_dot * np.cos(theta) * (E - E_des)
    return tau, E

def pid_control(theta, theta_dot, integral, prev_error):
    """Kontrol PID untuk menjaga keseimbangan di atas"""
    error = -theta
    integral += error
    integral = np.clip(integral, -integral_limit, integral_limit)
    derivative = error - prev_error
    tau = Kp * error + Ki * integral + Kd * derivative
    return tau, integral, error

# ==============================================
# 4️⃣ Simulasi PID
# ==============================================
episodes = 1000
max_steps = 200
rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    integral = 0.0
    prev_error = 0.0
    total_reward = 0.0
    currently_pid = False

    for t in range(max_steps):
        cos_th, sin_th, theta_dot = state
        theta = np.arctan2(sin_th, cos_th)

        # --- Hysteresis switching antara energy & PID mode ---
        if currently_pid:
            if (abs(theta) > exit_theta) or (abs(theta_dot) > exit_thdot):
                currently_pid = False
        else:
            if (abs(theta) < enter_theta) and (abs(theta_dot) < enter_thdot):
                currently_pid = True

        # --- Tentukan aksi kontrol ---
        if currently_pid:
            tau, integral, prev_error = pid_control(theta, theta_dot, integral, prev_error)
            mode = "PID"
        else:
            tau, E = energy_control(theta, theta_dot)
            # damping tambahan agar tidak terlalu agresif
            if abs(theta) < 0.6:
                tau += -3.0 * theta_dot - 6.0 * theta
            mode = "ENERGY"

        # --- Batasi aksi ---
        tau = float(np.clip(tau, -env.unwrapped.max_torque, env.unwrapped.max_torque))

        # --- Step environment ---
        next_state, reward, terminated, truncated, _ = env.step([tau])
        done = terminated or truncated
        total_reward += reward
        state = next_state

        # --- Sedikit delay agar animasi halus ---
        time.sleep(0.02)

        if done:
            break

    rewards.append(total_reward)
    print(f"Episode {episode+1}/{episodes} selesai | Total Reward: {total_reward:.2f}")

env.close()

# ==============================================
# 5️⃣ Analisis Hasil
# ==============================================
print("\nHASIL PENGUJIAN PID CONTROLLER:")
print(f"Rata-rata Reward : {np.mean(rewards):.2f}")
print(f"Reward Terbaik   : {np.max(rewards):.2f}")
print(f"Reward Terendah  : {np.min(rewards):.2f}")

# ==============================================
# 6️⃣ Visualisasi Hasil
# ==============================================
plt.figure(figsize=(8,5))
plt.plot(rewards, marker='o', linewidth=1.8, color='blue', label="PID Testing Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PID Pendulum Testing Performance (Per Episode)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pid_pendulum_testing_result.png")
plt.show()

print("\nGrafik disimpan sebagai 'pid_pendulum_testing_result.png'")

# Simpan hasil untuk perbandingan
np.save("pid_rewards.npy", rewards)
