#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import json

def find_log_directories(env_name, base_dir="/home/yche767/ML-IRL/logs"):
    """
    Find log directories for a given environment
    Returns a list of (log_dir, label) tuples
    """
    # Map environment names to full environment names
    env_mapping = {
        'ant': 'Ant-v2',
        'walker2d': 'Walker2d-v3', 
        'hopper': 'Hopper-v3',
        'humanoid': 'Humanoid-v3',
        'halfcheetah': 'HalfCheetah-v2'
    }
    
    full_env_name = env_mapping.get(env_name.lower(), env_name)
    env_path = os.path.join(base_dir, full_env_name, "exp-1", "maxentirl_sa")
    
    if not os.path.exists(env_path):
        print(f"Environment path not found: {env_path}")
        return []
    
    # Look for both timestamp directories and named experiment directories
    log_dirs_with_labels = []
    
    # First, try to find timestamp directories (old format)
    timestamp_dirs = glob.glob(os.path.join(env_path, "????_??_??_??_??_??"))
    timestamp_dirs.sort()  # Sort chronologically
    
    # Then, look for named experiment directories (new format)
    all_items = os.listdir(env_path)
    named_dirs = []
    for item in all_items:
        item_path = os.path.join(env_path, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Check if it's not a timestamp directory
            if not glob.fnmatch.fnmatch(item, "????_??_??_??_??_??"):
                named_dirs.append(item_path)
    
    named_dirs.sort()  # Sort alphabetically
    
    # Process timestamp directories first
    for i, log_dir in enumerate(timestamp_dirs):
        # Try to infer label from config or directory
        label = f"Experiment {i+1}"
        
        # Check if variant.json exists to get more info
        variant_path = os.path.join(log_dir, "variant.json")
        if os.path.exists(variant_path):
            try:
                with open(variant_path, 'r') as f:
                    variant = json.load(f)
                
                # Determine experiment type based on config
                use_constraints = variant.get('reward', {}).get('use_constraints', False)
                constraint_type = variant.get('reward', {}).get('constraint_type', 'none')
                
                if not use_constraints:
                    label = "No Constraint (Original)"
                elif constraint_type == 'simple':
                    label = "Simple Constraint"
                elif constraint_type == 'complex':
                    label = "Complex Constraint"
                else:
                    label = f"Constraint: {constraint_type}"
                    
            except Exception as e:
                print(f"Could not read variant.json for {log_dir}: {e}")
        
        log_dirs_with_labels.append((log_dir, label))
    
    # Process named directories
    for log_dir in named_dirs:
        # Use directory name as label
        dir_name = os.path.basename(log_dir)
        # Convert to more readable format
        if dir_name.upper() == 'ML-IRL':
            label = "ML-IRL"
        elif dir_name.upper() == 'PIRO':
            label = "PIRO"
        else:
            # Capitalize first letter and replace underscores/hyphens
            label = dir_name.replace('_', ' ').replace('-', ' ').title()
        
        log_dirs_with_labels.append((log_dir, label))
    
    return log_dirs_with_labels

def parse_args():
    parser = argparse.ArgumentParser(description='Plot training progress for different environments')
    parser.add_argument('--env', type=str, required=True, 
                       choices=['ant', 'walker2d', 'hopper', 'humanoid', 'halfcheetah'],
                       help='Environment name')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: {env}_training_progress.png)')
    parser.add_argument('--base_dir', type=str, default="/home/yche767/ML-IRL/logs",
                       help='Base directory for logs')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Find log directories
    log_dirs_with_labels = find_log_directories(args.env, args.base_dir)
    
    if not log_dirs_with_labels:
        print(f"No log directories found for environment: {args.env}")
        return
    
    print(f"Found {len(log_dirs_with_labels)} experiments for {args.env}")
    
    # Extract directories and labels
    log_dirs = [item[0] for item in log_dirs_with_labels]
    exp_labels = [item[1] for item in log_dirs_with_labels]

    # Read and store data from all experiments
    all_data = []
    for i, log_dir in enumerate(log_dirs):
        progress_file = os.path.join(log_dir, "progress.csv")
        try:
            df = pd.read_csv(progress_file)
            df['Experiment'] = exp_labels[i]
            all_data.append(df)
            print(f"Loaded {len(df)} rows from {exp_labels[i]}")
            print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error loading {progress_file}: {e}")

    if not all_data:
        print("No data loaded successfully!")
        return

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{args.env.title()} Training Progress Comparison', fontsize=16)

    # Colors for each experiment (use more distinct colors)
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = distinct_colors[:len(all_data)]

    # Plot 1: Real Det Return vs Iteration
    ax1 = axes[0, 0]
    for i, df in enumerate(all_data):
        if 'Real Det Return' in df.columns and 'Itration' in df.columns:
            ax1.plot(df['Itration'], df['Real Det Return'], 
                    label=exp_labels[i], color=colors[i], marker='o', markersize=3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Real Det Return')
    ax1.set_title('Real Deterministic Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Real Sto Return vs Iteration  
    ax2 = axes[0, 1]
    for i, df in enumerate(all_data):
        if 'Real Sto Return' in df.columns and 'Itration' in df.columns:
            ax2.plot(df['Itration'], df['Real Sto Return'],
                    label=exp_labels[i], color=colors[i], marker='s', markersize=3)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Real Sto Return')
    ax2.set_title('Real Stochastic Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Reward Loss vs Iteration
    ax3 = axes[1, 0]
    for i, df in enumerate(all_data):
        if 'Reward Loss' in df.columns and 'Itration' in df.columns:
            ax3.plot(df['Itration'], df['Reward Loss'],
                    label=exp_labels[i], color=colors[i], marker='^', markersize=3)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Reward Loss')
    ax3.set_title('Reward Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Both Returns on same plot for comparison
    ax4 = axes[1, 1]
    for i, df in enumerate(all_data):
        if 'Real Det Return' in df.columns and 'Itration' in df.columns:
            ax4.plot(df['Itration'], df['Real Det Return'],
                    label=f'{exp_labels[i]} (Det)', color=colors[i], linestyle='-', marker='o', markersize=2)
        if 'Real Sto Return' in df.columns and 'Itration' in df.columns:
            ax4.plot(df['Itration'], df['Real Sto Return'],
                    label=f'{exp_labels[i]} (Sto)', color=colors[i], linestyle='--', marker='s', markersize=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Return')
    ax4.set_title('Deterministic vs Stochastic Returns')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    if args.output is None:
        output_file = f"/home/yche767/ML-IRL/{args.env}_training_progress.png"
    else:
        output_file = args.output
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Also create a summary statistics table
    print("\n=== Summary Statistics ===")
    for i, df in enumerate(all_data):
        print(f"\n{exp_labels[i]}:")
        if 'Real Det Return' in df.columns:
            det_return = df['Real Det Return']
            print(f"  Real Det Return: mean={det_return.mean():.2f}, std={det_return.std():.2f}, max={det_return.max():.2f}")
        if 'Real Sto Return' in df.columns:
            sto_return = df['Real Sto Return']
            print(f"  Real Sto Return: mean={sto_return.mean():.2f}, std={sto_return.std():.2f}, max={sto_return.max():.2f}")
        if 'Reward Loss' in df.columns:
            reward_loss = df['Reward Loss']
            print(f"  Reward Loss: mean={reward_loss.mean():.2f}, std={reward_loss.std():.2f}, final={reward_loss.iloc[-1]:.2f}")

    plt.show()

if __name__ == "__main__":
    main()
