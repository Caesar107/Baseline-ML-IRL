#!/usr/bin/env python3
"""
Behavior Cloning using Stable-Baselines3 and Imitation library.
参数与主配置文件对齐，确保公平比较baseline效果。
"""

import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import datetime
import dateutil.tz
import json
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs
from utils import system, logger

def load_expert_data(env_name, num_episodes, expert_data_dir="../expert_data"):
    """Load expert trajectory data and convert to imitation format"""
    
    # Map environment names
    env_mapping = {
        'HalfCheetah-v2': 'HalfCheetah',
        'Ant-v2': 'Ant',
        'Walker2d-v3': 'Walker2d',
        'Hopper-v3': 'Hopper',
        'Humanoid-v3': 'Humanoid'
    }
    
    data_env_name = env_mapping.get(env_name, env_name.split('-')[0])
    data_path = os.path.join(expert_data_dir, data_env_name)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expert data directory not found: {data_path}")
    
    # Load data
    states = np.load(os.path.join(data_path, "states.npy"))
    actions = np.load(os.path.join(data_path, "actions.npy"))
    rewards = np.load(os.path.join(data_path, "reward.npy"))
    dones = np.load(os.path.join(data_path, "dones.npy"))
    
    # Select specified number of episodes
    states = states[:num_episodes]
    actions = actions[:num_episodes]
    rewards = rewards[:num_episodes]
    dones = dones[:num_episodes]
    
    print(f"Loaded expert data: {states.shape[0]} episodes")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    
    # Convert to imitation Transitions format
    transitions = []
    
    for ep_idx in range(num_episodes):
        ep_states = states[ep_idx]
        ep_actions = actions[ep_idx]
        ep_rewards = rewards[ep_idx]
        ep_dones = dones[ep_idx]
        
        # Find episode length (when done=True or max length)
        ep_len = len(ep_states)
        for t in range(len(ep_dones)):
            if ep_dones[t]:
                ep_len = t + 1
                break
        
        # Create transitions for this episode
        for t in range(ep_len - 1):  # -1 because we need next obs
            obs = ep_states[t]
            action = ep_actions[t]
            next_obs = ep_states[t + 1]
            reward = ep_rewards[t]
            done = bool(ep_dones[t])
            
            # Create transition dict compatible with imitation
            transition = {
                'obs': obs,
                'acts': action,
                'next_obs': next_obs,
                'rews': np.array([reward]),
                'dones': np.array([done]),
                'infos': np.array([{}])
            }
            transitions.append(transition)
    
    print(f"Created {len(transitions)} transitions")
    return transitions

def create_bc_trainer(env, device, config):
    """Create BC trainer using imitation library"""
    
    # Wrap environment for compatibility
    if not isinstance(env, DummyVecEnv):
        env = DummyVecEnv([lambda: env])
    
    # Create BC trainer
    rng = np.random.default_rng(seed=0)  # Use a fixed seed for reproducibility
    trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=None,  # Will be set later
        device=device,
        rng=rng,
        batch_size=config.get('batch_size', 256),
        optimizer_kwargs={
            'lr': config.get('lr', 1e-3)
        },
        ent_weight=config.get('ent_weight', 1e-3),
        l2_weight=config.get('l2_weight', 0.0)
    )
    
    return trainer

def convert_transitions_to_demonstrations(transitions):
    """Convert transition list to imitation demonstrations format"""
    from imitation.data.types import Trajectory
    
    # Group transitions by episodes (assuming they're in order)
    episodes = []
    current_episode = {
        'obs': [],
        'acts': [],
        'rews': [],
        'infos': []
    }
    
    for trans in transitions:
        current_episode['obs'].append(trans['obs'])
        current_episode['acts'].append(trans['acts'])
        current_episode['rews'].append(trans['rews'][0])
        current_episode['infos'].append(trans['infos'][0])
        
        if trans['dones'][0]:
            # End of episode - add final observation
            current_episode['obs'].append(trans['next_obs'])
            
            # Create trajectory
            traj = Trajectory(
                obs=np.array(current_episode['obs']),
                acts=np.array(current_episode['acts']),
                infos=np.array(current_episode['infos']),
                terminal=True
            )
            episodes.append(traj)
            
            # Reset for next episode
            current_episode = {
                'obs': [],
                'acts': [],
                'rews': [],
                'infos': []
            }
    
    # Handle last episode if it didn't end with done=True
    if len(current_episode['obs']) > 0:
        current_episode['obs'].append(transitions[-1]['next_obs'])
        traj = Trajectory(
            obs=np.array(current_episode['obs']),
            acts=np.array(current_episode['acts']),
            infos=np.array(current_episode['infos']),
            terminal=False
        )
        episodes.append(traj)
    
    print(f"Created {len(episodes)} demonstration trajectories")
    return episodes

def main():
    if len(sys.argv) != 2:
        print("Usage: python bc_sb3_trainer.py <config_file>")
        sys.exit(1)
    
    # Load configuration
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))
    
    # Extract parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    bc_config = v['bc']
    
    # System setup
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    
    print(f"Using device: {device}")
    print(f"Environment: {env_name}")
    print(f"BC Config: {bc_config}")
    
    # Environment setup
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    # Add render_mode attribute for compatibility with newer versions
    if not hasattr(env, 'render_mode'):
        env.render_mode = None
    if not hasattr(eval_env, 'render_mode'):
        eval_env.render_mode = None
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    if state_indices == 'all':
        state_indices = list(range(state_size))
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Using state indices: {len(state_indices)} dimensions")
    
    # Logging setup
    exp_id = f"baselines/logs/{env_name}/exp-{bc_config['expert_episodes']}/bc"
    if not os.path.exists(exp_id):
        os.makedirs(exp_id, exist_ok=True)
    
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)
    print(f"Logging to directory: {log_folder}")
    
    # Save config
    os.system(f'cp {sys.argv[1]} {log_folder}/variant.yml')
    with open(os.path.join(log_folder, 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    
    os.makedirs(os.path.join(log_folder, 'model'), exist_ok=True)
    
    # Load expert data
    print("Loading expert data...")
    transitions = load_expert_data(
        env_name, 
        bc_config['expert_episodes']
    )
    
    # Convert to demonstrations
    print("Converting to demonstration format...")
    demonstrations = convert_transitions_to_demonstrations(transitions)
    
    # Create BC trainer
    print("Creating BC trainer...")
    trainer = create_bc_trainer(
        env, 
        device, 
        {
            'batch_size': v.get('bc_network', {}).get('batch_size', 256),
            'lr': v.get('bc_network', {}).get('lr', 1e-3),
            'weight_decay': v.get('bc_network', {}).get('weight_decay', 0.0),
            'ent_weight': v.get('bc_network', {}).get('ent_weight', 1e-3),
            'l2_weight': v.get('bc_network', {}).get('l2_weight', 0.0)
        }
    )
    
    # Set demonstrations
    trainer.set_demonstrations(demonstrations)
    
    # Train BC policy
    print(f"Starting BC training for {bc_config['epochs']} epochs...")
    
    training_losses = []
    eval_freq = bc_config.get('eval_freq', 500)
    
    for epoch in range(0, bc_config['epochs'], eval_freq):
        # Train for eval_freq epochs
        epochs_to_train = min(eval_freq, bc_config['epochs'] - epoch)
        
        print(f"Training epochs {epoch} to {epoch + epochs_to_train}...")
        trainer.train(n_epochs=epochs_to_train)
        
        # Evaluate policy
        print(f"Evaluating at epoch {epoch + epochs_to_train}...")
        
        # Create a new eval environment to avoid conflicts
        temp_eval_env = gym.make(env_name)
        if not hasattr(temp_eval_env, 'render_mode'):
            temp_eval_env.render_mode = None
        eval_env_wrapped = DummyVecEnv([lambda: temp_eval_env])
        
        try:
            mean_reward, std_reward = evaluate_policy(
                trainer.policy,
                eval_env_wrapped,
                n_eval_episodes=bc_config.get('eval_episodes', 10),
                deterministic=True
            )
            print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            # Use a placeholder value if evaluation fails
            mean_reward, std_reward = 0.0, 0.0
        
        # Log results
        logger.record_tabular("BC Epoch", epoch + epochs_to_train)
        logger.record_tabular("BC Mean Reward", mean_reward)
        logger.record_tabular("BC Std Reward", std_reward)
        logger.dump_tabular()
        
        # Save intermediate model
        if (epoch + epochs_to_train) % 1000 == 0:
            model_path = os.path.join(log_folder, 'model', f'bc_policy_epoch_{epoch + epochs_to_train}.zip')
            trainer.policy.save(model_path)
            print(f"Saved intermediate model to: {model_path}")
    
    # Final evaluation
    print("Final evaluation...")
    
    # Create a new eval environment for final evaluation
    final_eval_env = gym.make(env_name)
    if not hasattr(final_eval_env, 'render_mode'):
        final_eval_env.render_mode = None
    eval_env_wrapped = DummyVecEnv([lambda: final_eval_env])
    
    try:
        final_mean_reward, final_std_reward = evaluate_policy(
            trainer.policy,
            eval_env_wrapped,
            n_eval_episodes=bc_config.get('eval_episodes', 10) * 2,  # More episodes for final eval
            deterministic=True
        )
        print(f"Final BC Performance:")
        print(f"  Mean reward: {final_mean_reward:.2f} ± {final_std_reward:.2f}")
    except Exception as e:
        print(f"Final evaluation failed: {e}")
        # Use placeholder values if final evaluation fails
        final_mean_reward, final_std_reward = 0.0, 0.0
        print(f"Using placeholder values for final results")
    
    # Save the trained policy
    model_path = os.path.join(log_folder, 'model', 'bc_policy_final.zip')
    trainer.policy.save(model_path)
    print(f"BC policy saved to: {model_path}")
    
    # Save final results
    results_path = os.path.join(log_folder, 'bc_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'final_mean_reward': final_mean_reward,
            'final_std_reward': final_std_reward,
            'num_demonstrations': len(demonstrations),
            'num_transitions': len(transitions),
            'epochs_trained': bc_config['epochs']
        }, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Final logging
    logger.record_tabular("BC Final Mean Reward", final_mean_reward)
    logger.record_tabular("BC Final Std Reward", final_std_reward)
    logger.record_tabular("BC Demonstrations", len(demonstrations))
    logger.record_tabular("BC Transitions", len(transitions))
    logger.dump_tabular()
    
    print("BC training completed!")

if __name__ == "__main__":
    main()
