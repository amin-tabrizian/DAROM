from stable_baselines3 import SAC
from src.merging import Merging
import numpy as np
import gymnasium as gym
from src.wrapper import DelayWrapper, PhysicsPredictorWrapper
from src.safety_controller import safetyCheck
from src.merging import Merging
from src.utils import get_user_action
import math
import sys
import os

# Get model path from command line or environment variable, default to MediumGRU
model_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv('MODEL_PATH', "models/GRU-uniform-delay/GRU-uniform-delay_best")
# Get use_safety from environment variable, default to True
use_safety = os.getenv('USE_SAFETY', 'True').lower() == 'true'
# Get delay mode from environment variable, default to 'all'
delay_mode = os.getenv('DELAY_MODE', 'bursty')
# Get use_predictor from environment variable, default to False
use_predictor = os.getenv('USE_PREDICTOR', 'False').lower() == 'true'


max_delay = 20
env = DelayWrapper(Merging(seed=42, render_mode='none'), max_delay=max_delay, mode='all', delay_mode=delay_mode)
if use_predictor:
    env = PhysicsPredictorWrapper(env)
# env = Merging(seed=42, render_mode='none')
# model_path = "EasyGRU/EasyGRU"
model = SAC.load(model_path, env=env)
print('Model loaded from ', model_path, 'Safe mode: ', use_safety, 'Delay mode: ', delay_mode, 'Predictor: ', use_predictor)

episode_reward = 0.0
step = 0
no_check = False
success = 0
collision = 0
max_steps = 0
emergency_stopping = 0
epoches = 501

n_runs = 3
epoches_per_run = epoches

success_list = []
collision_list = []
max_steps_list = []
emergency_stopping_list = []
episode_reward_list = []

# For average velocity, acc, jerk statistics
velocity_all_episodes = []
acc_all_episodes = []
jerk_all_episodes = []

for run_idx in range(n_runs):
    print("="*20 + f" Run {run_idx+1} / {n_runs} " + "="*20)
    success = 0
    collision = 0
    max_steps = 0
    emergency_stopping = 0
    episode_rewards = []

    for ep in range(epoches_per_run):
        print('episode', ep)
        obs, _ = env.reset()
        no_check = False
        count = 0

        episode_reward = 0.0
        step = 0
        
        velocities = []
        accs = []
        jerks = []
        prev_velocity = None
        prev_acc = None
        dt = getattr(env.unwrapped, 'dt', 0.1)  # Try to get dt from env; fallback to 0.1s if unavailable

        while True:
            action, _ = model.predict(obs, deterministic=True)

            # For user input mode, uncomment the following:
            # action = get_user_action()
            if use_safety:
                action, safety_info = safetyCheck(action, obs, True, 'all')
            obs_new, reward, terminated, truncated, info = env.step(action)

            done = np.any(terminated) or np.any(truncated)
            # print('observation \n', obs, 'action', action)
            # Assume obs is (n_agents, obs_dim). ego is index 0.
            # Velocity: element 2 (v_ego), acc: element 3, if available. Otherwise, approximate acc via diff.
            ego_v = obs[0][2] if isinstance(obs, np.ndarray) and obs.ndim > 1 else obs[2]
            velocities.append(ego_v)
            if prev_velocity is not None:
                acc = (ego_v - prev_velocity) / dt
                accs.append(acc)
                # Jerk: delta acc / dt
                if prev_acc is not None:
                    jerk = (acc - prev_acc) / dt
                    jerks.append(jerk)
                prev_acc = acc
            prev_velocity = ego_v

            episode_reward += float(reward)
            step += 1
            count += 1
            obs = obs_new
            if done:
                print('info', info)
                print('episode reward', episode_reward)
                break

        episode_rewards.append(episode_reward)
        if velocities:
            velocity_all_episodes.append(np.mean(velocities))
        if accs:
            acc_all_episodes.append(np.mean(accs))
        if jerks:
            jerk_all_episodes.append(np.mean(jerks))

        if info['message'] == 'Arrived':
            success += 1
        if info['message'] == 'Collision':
            collision += 1
        if info['message'] == 'Max Steps':
            max_steps += 1
        if info['message'] == 'Emergency Stopping':
            emergency_stopping += 1

    success_list.append(success / epoches_per_run * 100)
    collision_list.append(collision / epoches_per_run * 100)
    max_steps_list.append(max_steps / epoches_per_run * 100)
    emergency_stopping_list.append(emergency_stopping / epoches_per_run * 100)
    episode_reward_list.append(np.mean(episode_rewards))

def mean_and_error(arr):
    m = np.mean(arr)
    std = np.std(arr, ddof=1)
    bound = 1.96 * std / math.sqrt(len(arr))  # 95% confidence interval
    return m, bound

success_mean, success_err = mean_and_error(success_list)
collision_mean, collision_err = mean_and_error(collision_list)
max_steps_mean, max_steps_err = mean_and_error(max_steps_list)
emergency_stopping_mean, emergency_stopping_err = mean_and_error(emergency_stopping_list)
reward_mean, reward_err = mean_and_error(episode_reward_list)

vel_mean, vel_err = mean_and_error(velocity_all_episodes)
acc_mean, acc_err = mean_and_error(acc_all_episodes)
jerk_mean, jerk_err = mean_and_error(jerk_all_episodes)

print("\n========== RESULTS OVER {} RUNS ==========".format(n_runs))
print(f"Success rate: {success_mean:.2f}% ± {success_err:.2f}%")
print(f"Collision rate: {collision_mean:.2f}% ± {collision_err:.2f}%")
print(f"Max steps rate: {max_steps_mean:.2f}% ± {max_steps_err:.2f}%")
print(f"Emergency stopping rate: {emergency_stopping_mean:.2f}% ± {emergency_stopping_err:.2f}%")
print(f"Average Episode Reward: {reward_mean:.2f} ± {reward_err:.2f}")

print(f"\nEgo Vehicle Average Velocity: {vel_mean:.2f} ± {vel_err:.2f} m/s")
print(f"Ego Vehicle Average Acceleration: {acc_mean:.2f} ± {acc_err:.2f} m/s²")
print(f"Ego Vehicle Average Jerk: {jerk_mean:.2f} ± {jerk_err:.2f} m/s³")