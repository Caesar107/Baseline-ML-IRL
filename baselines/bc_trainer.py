#!/usr/bin/env python3
"""
Behavior Cloning (BC) implementation for the ML-IRL project.
This script implements BC as a baseline algorithm to compare with IRL methods.
"""

import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs
from utils import system, logger, eval
from models.bc_policy import BCPolicy
import datetime
import dateutil.tz
import json

class BCTrainer:
    """Behavior Cloning Trainer"""
    
    def __init__(self, policy, optimizer, device='cpu'):
        self.policy = policy
        self.optimizer = optimizer
        self.device = device
        self.policy.to(device)
        
    def train_step(self, states, actions):
        """Single training step"""
        self.policy.train()
        
        # Forward pass
        predicted_actions = self.policy(states)
        
        # MSE loss
        loss = nn.MSELoss()(predicted_actions, actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, env_fn, num_episodes=10, max_ep_len=1000):
        """Evaluate the BC policy"""
        self.policy.eval()
        episode_returns = []
        
        for _ in range(num_episodes):
            env = env_fn()
            state = env.reset()
            episode_return = 0
            
            for _ in range(max_ep_len):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action = self.policy(state_tensor).cpu().numpy().flatten()
                
                next_state, reward, done, _ = env.step(action)
                episode_return += reward
                state = next_state
                
                if done:
                    break
            
            episode_returns.append(episode_return)
            env.close()
        
        return np.mean(episode_returns), np.std(episode_returns)
    
    def get_action(self, state, deterministic=True):
        """Get action from policy (for compatibility with SAC interface)"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action = self.forward(state)
        
        return action.cpu().numpy()

class BCTrainer:
    """Behavior Cloning Trainer"""
    
    def __init__(self, state_dim, action_dim, device, config):
        self.device = device
        self.config = config
        
        # Initialize policy
        self.policy = BCPolicy(
            state_dim, 
            action_dim,
            hidden_sizes=config.get('hidden_sizes', [256, 256]),
            activation=config.get('activation', 'relu')
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 0.0)
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def train_step(self, states, actions):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        predicted_actions = self.policy(states)
        loss = self.criterion(predicted_actions, actions)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, expert_states, expert_actions, epochs, batch_size=256, eval_freq=100):
        """Train the BC policy"""
        
        # Convert to tensors
        expert_states = torch.FloatTensor(expert_states).to(self.device)
        expert_actions = torch.FloatTensor(expert_actions).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(expert_states, expert_actions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        training_losses = []
        
        print(f"Starting BC training for {epochs} epochs...")
        print(f"Dataset size: {len(expert_states)} samples")
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_states, batch_actions in dataloader:
                loss = self.train_step(batch_states, batch_actions)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            training_losses.append(avg_loss)
            
            if epoch % eval_freq == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
                
                # Log to logger if available
                logger.record_tabular("BC Epoch", epoch)
                logger.record_tabular("BC Loss", avg_loss)
                logger.dump_tabular()
        
        print("BC training completed!")
        return training_losses

def load_expert_data(env_name, num_episodes, expert_data_dir="expert_data"):
    """Load expert trajectory data"""
    
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
    
    # Select specified number of episodes
    states = states[:num_episodes]
    actions = actions[:num_episodes]
    
    print(f"Loaded expert data: {states.shape[0]} episodes")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    
    # Flatten to (n_samples, dim)
    states_flat = states.reshape(-1, states.shape[-1])
    actions_flat = actions.reshape(-1, actions.shape[-1])
    
    return states_flat, actions_flat

def evaluate_bc_policy(policy, env, num_episodes=10, max_steps=1000):
    """Evaluate the BC policy in the environment"""
    
    episode_returns = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        episode_return = 0
        episode_length = 0
        
        for step in range(max_steps):
            action = policy.get_action(obs, deterministic=True)[0]
            obs, reward, done, _ = env.step(action)
            
            episode_return += reward
            episode_length += 1
            
            if done:
                break
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
    
    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'episode_returns': episode_returns
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python bc_trainer.py <config_file>")
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
    
    # Environment setup
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    
    if state_indices == 'all':
        state_indices = list(range(state_size))
    
    # Logging setup
    exp_id = f"logs/{env_name}/exp-{bc_config['expert_episodes']}/bc"
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)
    
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
    expert_states, expert_actions = load_expert_data(
        env_name, 
        bc_config['expert_episodes']
    )
    
    # Filter states by indices
    expert_states = expert_states[:, state_indices]
    
    print(f"Expert data after filtering: states {expert_states.shape}, actions {expert_actions.shape}")
    
    # Initialize BC trainer
    bc_trainer = BCTrainer(
        state_dim=len(state_indices),
        action_dim=action_size,
        device=device,
        config={
            'hidden_sizes': v.get('bc_network', {}).get('hidden_sizes', [256, 256]),
            'activation': v.get('bc_network', {}).get('activation', 'relu'),
            'lr': v.get('bc_network', {}).get('lr', 1e-3),
            'weight_decay': v.get('bc_network', {}).get('weight_decay', 0.0)
        }
    )
    
    # Train BC policy
    training_losses = bc_trainer.train(
        expert_states=expert_states,
        expert_actions=expert_actions,
        epochs=bc_config['epochs'],
        batch_size=v.get('bc_network', {}).get('batch_size', 256),
        eval_freq=bc_config['eval_freq']
    )
    
    # Evaluate policy periodically during training
    print("\nEvaluating BC policy...")
    
    for eval_epoch in range(0, bc_config['epochs'], bc_config['eval_freq']):
        if eval_epoch > 0:  # Skip initial evaluation
            eval_results = evaluate_bc_policy(
                bc_trainer.policy, 
                gym_env, 
                num_episodes=bc_config['eval_episodes'],
                max_steps=v['env']['T']
            )
            
            print(f"Evaluation at epoch {eval_epoch}:")
            print(f"  Mean return: {eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}")
            print(f"  Mean length: {eval_results['mean_length']:.1f}")
            
            # Log evaluation results
            logger.record_tabular("BC Eval Epoch", eval_epoch)
            logger.record_tabular("BC Eval Return", eval_results['mean_return'])
            logger.record_tabular("BC Eval Std", eval_results['std_return'])
            logger.record_tabular("BC Eval Length", eval_results['mean_length'])
            logger.dump_tabular()
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_results = evaluate_bc_policy(
        bc_trainer.policy, 
        gym_env, 
        num_episodes=bc_config['eval_episodes'] * 2,  # More episodes for final eval
        max_steps=v['env']['T']
    )
    
    print(f"Final BC Performance:")
    print(f"  Mean return: {final_results['mean_return']:.2f} ± {final_results['std_return']:.2f}")
    print(f"  Mean length: {final_results['mean_length']:.1f}")
    print(f"  Episode returns: {final_results['episode_returns']}")
    
    # Save the trained policy
    model_path = os.path.join(log_folder, 'model', 'bc_policy.pkl')
    torch.save(bc_trainer.policy.state_dict(), model_path)
    print(f"BC policy saved to: {model_path}")
    
    # Save final results
    results_path = os.path.join(log_folder, 'bc_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'final_mean_return': final_results['mean_return'],
            'final_std_return': final_results['std_return'],
            'final_mean_length': final_results['mean_length'],
            'episode_returns': final_results['episode_returns'],
            'training_losses': training_losses
        }, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Final logging
    logger.record_tabular("BC Final Return", final_results['mean_return'])
    logger.record_tabular("BC Final Std", final_results['std_return'])
    logger.record_tabular("BC Final Length", final_results['mean_length'])
    logger.dump_tabular()

if __name__ == "__main__":
    main()
