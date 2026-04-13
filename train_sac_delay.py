# from delay_encoder import DelayAwareGRUEncoder
import os
import time
import argparse
import warnings
from typing import Optional
from src.wrapper import SafetyWrapper

import numpy as np

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from src.delay_encoder import DelayAwareEncoder, DelayAwareGRUEncoder, TinyTransformerEncoder

import wandb
from wandb.integration.sb3 import WandbCallback

from src.merging import Merging
from src.wrapper import DelayWrapper, PhysicsPredictorWrapper

class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # Track episode reward and length
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log to wandb
            wandb.log({
                'train/episode_reward': self.current_episode_reward,
                'train/episode_length': self.current_episode_length,
                'train/mean_episode_reward': np.mean(self.episode_rewards),
                'train/mean_episode_length': np.mean(self.episode_lengths),
            })
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

class EvalAndBestSaveCallback(BaseCallback):
    """
    After every `eval_freq` steps, run an evaluation policy (deterministic) for `n_eval_episodes` episodes.
    If the mean reward improves, save the current best policy.
    """
    def __init__(self, eval_env_fn, save_path, eval_freq=100, n_eval_episodes=21, verbose=1):
        super().__init__()
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.verbose = verbose
        self.best_mean_reward = -np.inf
        self._num_timesteps = 0

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.eval_freq == 0:
            # Re-create eval env each time so there's no data leak or lingering state.
            eval_env = self.eval_env_fn()
            arrival_counts = 0
            rewards_list = []
            for ep in range(self.n_eval_episodes):
                obs, _ = eval_env.reset()
                done = False
                episode_arrivals = 0
                ep_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    ep_reward += reward
                    done = np.any(terminated) or np.any(truncated)
                    # Check for 'Arrival' message in info
                    real_info = info
                    if isinstance(info, (tuple, list)) and len(info) > 0:
                        real_info = info[-1]
                    if isinstance(real_info, dict) and 'message' in real_info:
                        if real_info['message'] == 'Arrived':
                            episode_arrivals += 1
                arrival_counts += episode_arrivals
                rewards_list.append(ep_reward)
            arrival_rate = arrival_counts / self.n_eval_episodes
            mean_reward = np.mean(rewards_list)
            if self.verbose > 0:
                print(f"Eval after {self.num_timesteps:,} steps: arrival_rate={arrival_rate:.4f} ({arrival_counts}/{self.n_eval_episodes}), mean_reward={mean_reward:.2f}")
            wandb.log({'eval/step': self.num_timesteps, 'eval/arrival_rate': arrival_rate, 'eval/mean_reward': mean_reward})
            # Save model if arrival_rate is higher than previously seen (i.e., best so far)
            if arrival_rate > getattr(self, "best_arrival_rate", -1.0):
                self.best_arrival_rate = arrival_rate
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(f"New best model with arrival_rate={arrival_rate:.4f} (mean_reward={mean_reward:.2f}) saved at {self.save_path}")
                wandb.log({'best_model_step': self.num_timesteps, 'best_model_arrival_rate': arrival_rate, 'best_model_mean_reward': mean_reward})
            elif abs(arrival_rate - 1.0) < 1e-4:
                # Only save if this reward is higher than previous "perfect arrival" reward
                if not hasattr(self, "best_perfect_reward") or mean_reward > self.best_perfect_reward:
                    self.best_perfect_reward = mean_reward
                    self.model.save(self.save_path)
                    if self.verbose > 0:
                        print(f"Arrival rate is 1.0, and mean_reward improved to {mean_reward:.2f}. Saving model at {self.save_path}")
                    wandb.log({'arrival_rate_1.0_step': self.num_timesteps, 'arrival_rate_1.0_best_mean_reward': mean_reward})
                else:
                    if self.verbose > 0:
                        print(f"Arrival rate is 1.0, but mean_reward={mean_reward:.2f} did not improve (best={self.best_perfect_reward:.2f}), not saving.")
            # After evaluations are done, reset env to ensure a proper cleanup/reset of internal state
            eval_env.reset()
        return True

class StepLoggingCallback(EpisodeRewardCallback):
    def __init__(self):
        super().__init__()
        self.step_num = 0

    def _on_step(self) -> bool:
        # infos = self.locals.get("infos", [])
        # observations = self.locals.get("new_obs") if "new_obs" in self.locals else self.locals.get("obs")
        # rewards = self.locals.get("rewards")
        # dones = self.locals.get("dones")
        # Print observations, rewards, dones, info per step to console ONLY (no wandb.log)
        # if observations is not None and rewards is not None:
            # batch_size = len(rewards) if hasattr(rewards, '__len__') and not isinstance(rewards, str) else 1
            # for i in range(batch_size):
            #     obs = observations[i] if hasattr(observations, '__getitem__') else observations
            #     rew = rewards[i] if hasattr(rewards, '__getitem__') else rewards
            #     dn = dones[i] if hasattr(dones, '__getitem__') else dones
            #     info = infos[i] if isinstance(infos, (list, tuple)) and len(infos) > 0 else infos
                # print("Step:", self.step_num)
                # print("  Obs:", obs)
                # print("  Reward:", float(rew))
                # print("  Done:", bool(dn))
                # if info is not None and isinstance(info, dict) and len(info) > 0:
                #     print("  Info:")
                #     for k, v in info.items():
                #         print(f"    {k}: {v}")
                # self.step_num += 1
        pass
        # return super()._on_step()

def save_model(model: SAC, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    return path

def load_model(path: str, env: gym.Env) -> SAC:
    return SAC.load(path, env=env)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='merging-SAC_experiments')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, choices=['must', 'allow', 'never'])
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--radius', type=int, default=50)
    parser.add_argument('--total_timesteps', type=int, default=10_000_000)
    parser.add_argument('--render_mode', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='models_experiments')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--encoder_mode', type=str, default='None', choices=['Transformer', 'GRU', 'MLP', 'None'])
    parser.add_argument('--use_predictor', action='store_true', help='Wrap env with PhysicsPredictorWrapper')
    parser.add_argument('--delay_mode', type=str, default='uniform', choices=['uniform', 'exponential', 'triangular', 'bursty', 'bimodal'])
    return parser.parse_args()

def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    args = parse_args()

    if args.run_name is None:
        args.run_name = f"SAC-merging-delay-wrapper-{time.strftime('%Y%m%d-%H%M%S')}"

    if args.wandb_mode == 'disabled':
        os.environ['WANDB_MODE'] = 'disabled'
    elif args.wandb_mode == 'offline':
        os.environ['WANDB_MODE'] = 'offline'

    config = dict(
        algo='SAC',
        radius=args.radius,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        policy='MlpPolicy',
        encoder_mode=args.encoder_mode,
        delay_mode=args.delay_mode,
    )
    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name, config=config, sync_tensorboard=True, save_code=True, resume=args.resume, id=args.run_id)
    run.log_code('src')
    # Create env
    max_delay = 20
    def _fn():
        env = DelayWrapper(Merging(seed=args.seed, render_mode=args.render_mode), max_delay=max_delay, mode='all', delay_mode=args.delay_mode)
        if args.use_predictor:
            env = PhysicsPredictorWrapper(env)
        return env

    # Define evaluation environment function
    def make_eval_env():
        # Use separate seed for eval to avoid stochasticity coupling with train
        env = DelayWrapper(Merging(seed=args.seed + 42, render_mode=None), max_delay=max_delay, mode='all', delay_mode=args.delay_mode)
        if args.use_predictor:
            env = PhysicsPredictorWrapper(env)
        return env

    # SB3 expects the Gym API; our env already implements gymnasium API. SB3 v2 works with gymnasium.
    vec_env = DummyVecEnv([_fn])

    wrapped_env = vec_env.envs[0]
    # When PhysicsPredictorWrapper is active, the DelayWrapper sits one level below
    delay_env = wrapped_env.env if args.use_predictor else wrapped_env
    entity_seq_len, entity_feat_dim = delay_env.orig_shape
    act_dim = delay_env.env.action_space.shape[0]
    encoder_mode = args.encoder_mode
    if encoder_mode == 'Transformer':
        policy_kwargs = dict(
        features_extractor_class=TinyTransformerEncoder,
        features_extractor_kwargs=dict(
            entity_seq_len=entity_seq_len,
            entity_feat_dim=entity_feat_dim,
            act_hist_len=delay_env.max_delay,
            act_dim=act_dim,
            delay_dim=1,
            d_model=32,
            n_heads=1,
            ffn_hidden=64,
            n_layers=1,
            features_dim=128,
        ),
        net_arch=[256, 256],      # actor/critic MLP after transformer
    )
    elif encoder_mode == 'GRU':
        policy_kwargs = dict(features_extractor_class=DelayAwareGRUEncoder, features_extractor_kwargs=dict(act_dim=act_dim, hist_len=delay_env.max_delay+1, delay_dim=1, features_dim=128, gru_hidden=64))
    elif encoder_mode == 'MLP':
        policy_kwargs = dict(
        features_extractor_class=DelayAwareEncoder,
        features_extractor_kwargs=dict(act_hist_dim=max_delay * act_dim, delay_dim=1)
        )
    else:
        policy_kwargs = dict()

    model = SAC('MlpPolicy', vec_env, seed=args.seed, verbose=1, tensorboard_log=os.path.join('runs', args.run_name), device='cuda', learning_rate=3e-5, policy_kwargs=policy_kwargs, batch_size=512)    
    
    
    

    # Optionally load existing weights
    if args.load_path and os.path.exists(args.load_path):
        try:
            model = SAC.load(args.load_path, env=vec_env)
            print(f"Loaded model from {args.load_path}")
        except Exception as e:
            warnings.warn(f"Failed to load model at {args.load_path}: {e}")

    ensure_dirs(args.save_dir)
    model_path = os.path.join(args.save_dir, f"{args.run_name}.zip")
    best_model_path = os.path.join(args.save_dir, f"{args.run_name}_best.zip")

    # Create callbacks
    episode_callback = EpisodeRewardCallback()
    eval_and_save_callback = EvalAndBestSaveCallback(
        eval_env_fn=make_eval_env,
        save_path=best_model_path,
        eval_freq=100_000,
        n_eval_episodes=51,
        verbose=1,
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=args.save_dir,
        model_save_freq=5000,
        verbose=2,
    )

    # Combine callbacks
    callbacks = [episode_callback, eval_and_save_callback, wandb_callback]

    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    # Save final
    save_model(model, model_path)
    print(f"Saved latest model to {model_path}")
    print(f"Best model (by eval mean episode reward) at: {best_model_path}")

if __name__ == '__main__':
    main()
