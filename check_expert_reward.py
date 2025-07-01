#!/usr/bin/env python3 cd /home/yche767/ML-IRL && python check_expert_reward.py --env ant --save_stats ant_expert_stats.txt
"""
Script to analyze stored expert reward data without environment replay.
This helps analyze the quality of expert data and compare with IRL results.
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path

def load_expert_data(env_name, expert_data_dir="expert_data"):
    """
    Load expert trajectory data for a given environment.
    
    Args:
        env_name: Environment name (e.g., 'HalfCheetah', 'Ant', etc.)
        expert_data_dir: Directory containing expert data
    
    Returns:
        tuple: (states, actions, rewards, dones, episode_lengths)
    """
    # Map full environment names to data directory names
    env_mapping = {
        'HalfCheetah-v2': 'HalfCheetah',
        'HalfCheetah-v3': 'HalfCheetah',
        'Ant-v2': 'Ant',
        'Walker2d-v3': 'Walker2d',
        'Hopper-v3': 'Hopper',
        'Humanoid-v3': 'Humanoid'
    }
    
    data_env_name = env_mapping.get(env_name, env_name.split('-')[0])
    data_path = os.path.join(expert_data_dir, data_env_name)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expert data directory not found: {data_path}")
    
    # Load data files
    states = np.load(os.path.join(data_path, "states.npy"))
    actions = np.load(os.path.join(data_path, "actions.npy"))
    rewards = np.load(os.path.join(data_path, "reward.npy"))
    dones = np.load(os.path.join(data_path, "dones.npy"))
    lengths = np.load(os.path.join(data_path, "lens.npy"))
    
    print(f"Loaded expert data for {data_env_name}:")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Dones shape: {dones.shape}")
    print(f"  Lengths shape: {lengths.shape}")
    
    return states, actions, rewards, dones, lengths

def analyze_expert_rewards(rewards, lengths):
    """
    Analyze the stored expert rewards comprehensively.
    
    Args:
        rewards: Expert reward trajectories [n_episodes, max_len]
        lengths: Episode lengths [n_episodes] or [n_episodes, 1]
    
    Returns:
        dict: Analysis results
    """
    episode_returns = []
    episode_lengths = []
    
    print(f"Analyzing {rewards.shape[0]} expert episodes...")
    
    for ep in range(rewards.shape[0]):
        # Handle different length array shapes
        if len(lengths.shape) > 1 and lengths.shape[1] > 0:
            ep_len = int(lengths[ep, 0]) if lengths[ep, 0] > 0 else rewards.shape[1]
        else:
            ep_len = int(lengths[ep]) if len(lengths) > ep and lengths[ep] > 0 else rewards.shape[1]
        
        # Sum rewards for this episode up to actual length
        ep_len = min(ep_len, rewards.shape[1])  # Don't exceed max trajectory length
        ep_return = np.sum(rewards[ep, :ep_len])
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        
        # Print first few episodes for debugging
        if ep < 5:
            print(f"  Episode {ep+1}: Return = {ep_return:.2f}, Length = {ep_len}")
    
    # Calculate statistics
    episode_returns = np.array(episode_returns)
    episode_lengths = np.array(episode_lengths)
    
    results = {
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'median_return': np.median(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'num_episodes': len(episode_returns),
        'total_steps': np.sum(episode_lengths)
    }
    
    # Additional analysis
    results['return_per_step'] = results['mean_return'] / results['mean_length']
    results['q25_return'] = np.percentile(episode_returns, 25)
    results['q75_return'] = np.percentile(episode_returns, 75)
    
    return results

def plot_expert_analysis(env_name, results, output_path=None):
    """
    Create visualization of expert reward statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Return distribution
    ax1 = axes[0, 0]
    ax1.hist(results['episode_returns'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(results['mean_return'], color='red', linestyle='--', label=f'Mean: {results["mean_return"]:.2f}')
    ax1.axvline(results['median_return'], color='orange', linestyle='--', label=f'Median: {results["median_return"]:.2f}')
    ax1.set_xlabel('Episode Return')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{env_name}: Expert Return Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode returns over time
    ax2 = axes[0, 1]
    episodes = range(len(results['episode_returns']))
    ax2.plot(episodes, results['episode_returns'], 'o-', alpha=0.7, markersize=4)
    ax2.axhline(results['mean_return'], color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Return')
    ax2.set_title(f'{env_name}: Episode Returns')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Episode length distribution
    ax3 = axes[1, 0]
    ax3.hist(results['episode_lengths'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(results['mean_length'], color='red', linestyle='--', label=f'Mean: {results["mean_length"]:.1f}')
    ax3.set_xlabel('Episode Length')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{env_name}: Episode Length Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Return vs Length scatter
    ax4 = axes[1, 1]
    ax4.scatter(results['episode_lengths'], results['episode_returns'], alpha=0.6)
    ax4.set_xlabel('Episode Length')
    ax4.set_ylabel('Episode Return')
    ax4.set_title(f'{env_name}: Return vs Length')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze stored expert reward data')
    parser.add_argument('--env', type=str, required=True,
                       choices=['halfcheetah', 'ant', 'walker2d', 'hopper', 'humanoid'],
                       help='Environment name')
    parser.add_argument('--expert_dir', type=str, default='expert_data',
                       help='Expert data directory (default: expert_data)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output plot path (default: {env}_expert_analysis.png)')
    parser.add_argument('--save_stats', type=str, default=None,
                       help='Save statistics to text file (optional)')
    
    args = parser.parse_args()
    
    # Map environment names to full names
    env_mapping = {
        'halfcheetah': 'HalfCheetah-v2',
        'ant': 'Ant-v2',
        'walker2d': 'Walker2d-v3',
        'hopper': 'Hopper-v3',
        'humanoid': 'Humanoid-v3'
    }
    
    full_env_name = env_mapping[args.env]
    
    try:
        # Load expert data
        print(f"Loading expert data for {args.env}...")
        states, actions, rewards, dones, lengths = load_expert_data(full_env_name, args.expert_dir)
        
        # Analyze stored rewards
        print("\n" + "="*50)
        print("STORED EXPERT REWARDS ANALYSIS")
        print("="*50)
        results = analyze_expert_rewards(rewards, lengths)
        
        # Print comprehensive statistics
        print(f"Dataset Summary:")
        print(f"  Number of episodes: {results['num_episodes']}")
        print(f"  Total steps: {results['total_steps']}")
        print(f"  Average episode length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        
        print(f"\nReturn Statistics:")
        print(f"  Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
        print(f"  Median return: {results['median_return']:.2f}")
        print(f"  Min return: {results['min_return']:.2f}")
        print(f"  Max return: {results['max_return']:.2f}")
        print(f"  25th percentile: {results['q25_return']:.2f}")
        print(f"  75th percentile: {results['q75_return']:.2f}")
        print(f"  Return per step: {results['return_per_step']:.3f}")
        
        # Quality assessment
        print(f"\nQuality Assessment:")
        cv_return = results['std_return'] / abs(results['mean_return']) if results['mean_return'] != 0 else float('inf')
        print(f"  Coefficient of variation: {cv_return:.3f}")
        
        if cv_return < 0.1:
            print("  ✅ Very consistent performance (CV < 0.1)")
        elif cv_return < 0.3:
            print("  ✅ Good consistency (0.1 <= CV < 0.3)")
        elif cv_return < 0.5:
            print("  ⚠️  Moderate variation (0.3 <= CV < 0.5)")
        else:
            print("  ⚠️  High variation (CV >= 0.5) - may indicate inconsistent expert quality")
        
        # Save statistics to file if requested
        if args.save_stats:
            with open(args.save_stats, 'w') as f:
                f.write(f"Expert Data Analysis for {full_env_name}\n")
                f.write("="*50 + "\n")
                f.write(f"Dataset Summary:\n")
                f.write(f"  Number of episodes: {results['num_episodes']}\n")
                f.write(f"  Total steps: {results['total_steps']}\n")
                f.write(f"  Average episode length: {results['mean_length']:.1f} ± {results['std_length']:.1f}\n")
                f.write(f"\nReturn Statistics:\n")
                f.write(f"  Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}\n")
                f.write(f"  Median return: {results['median_return']:.2f}\n")
                f.write(f"  Min return: {results['min_return']:.2f}\n")
                f.write(f"  Max return: {results['max_return']:.2f}\n")
                f.write(f"  25th percentile: {results['q25_return']:.2f}\n")
                f.write(f"  75th percentile: {results['q75_return']:.2f}\n")
                f.write(f"  Return per step: {results['return_per_step']:.3f}\n")
                f.write(f"  Coefficient of variation: {cv_return:.3f}\n")
            print(f"\nStatistics saved to: {args.save_stats}")
        
        # Create visualization
        if args.output is None:
            output_path = f"{args.env}_expert_analysis.png"
        else:
            output_path = args.output
            
        plot_expert_analysis(full_env_name, results, output_path)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
