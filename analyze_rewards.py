#!/usr/bin/env python3
"""
Script to analyze reward performance across all environments and algorithms.
Based on plot.py structure to systematically examine CSV results.
"""

import pandas as pd
import numpy as np
import os
import json
import glob
from pathlib import Path

def find_all_experiments(base_dir="/home/yche767/ML-IRL/logs"):
    """
    Find all experiment directories across all environments
    Returns a dictionary with structure: {env: {algorithm: [experiment_paths]}}
    """
    experiments = {}
    
    # Environment mapping
    env_mapping = {
        'Ant-v2': 'ant',
        'Walker2d-v3': 'walker2d', 
        'Hopper-v3': 'hopper',
        'Humanoid-v3': 'humanoid',
        'HalfCheetah-v2': 'halfcheetah'
    }
    
    for env_dir in os.listdir(base_dir):
        env_path = os.path.join(base_dir, env_dir)
        if not os.path.isdir(env_path):
            continue
            
        short_env_name = env_mapping.get(env_dir, env_dir)
        experiments[short_env_name] = {}
        
        # Look for exp-* directories
        exp_dirs = glob.glob(os.path.join(env_path, "exp-*"))
        for exp_dir in exp_dirs:
            # Look for algorithm directories (maxentirl, maxentirl_sa, etc.)
            alg_dirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
            
            for alg_dir in alg_dirs:
                alg_path = os.path.join(exp_dir, alg_dir)
                
                # Find all experiment runs in this algorithm directory
                exp_runs = []
                for item in os.listdir(alg_path):
                    item_path = os.path.join(alg_path, item)
                    if os.path.isdir(item_path):
                        # Check if it has progress.csv
                        progress_file = os.path.join(item_path, "progress.csv")
                        if os.path.exists(progress_file):
                            exp_runs.append(item_path)
                
                if exp_runs:
                    if alg_dir not in experiments[short_env_name]:
                        experiments[short_env_name][alg_dir] = []
                    experiments[short_env_name][alg_dir].extend(exp_runs)
    
    return experiments

def get_experiment_label(exp_path):
    """
    Get a readable label for an experiment based on its config
    """
    # Default label based on directory name
    dir_name = os.path.basename(exp_path)
    
    # Try to get more info from variant.json
    variant_path = os.path.join(exp_path, "variant.json")
    if os.path.exists(variant_path):
        try:
            with open(variant_path, 'r') as f:
                variant = json.load(f)
            
            # Determine experiment type based on config
            use_constraints = variant.get('reward', {}).get('use_constraints', False)
            constraint_type = variant.get('reward', {}).get('constraint_type', 'none')
            
            if not use_constraints:
                return "No Constraint (Original)"
            elif constraint_type == 'simple':
                return "Simple Constraint"
            elif constraint_type == 'complex':
                return "Complex Constraint"
            else:
                return f"Constraint: {constraint_type}"
                
        except Exception as e:
            pass
    
    # Convert timestamp to readable format or use directory name
    if dir_name.count('_') >= 5:  # Timestamp format
        return f"Run {dir_name[-2:]}"  # Use last part as identifier
    else:
        # Named directories
        if dir_name.upper() == 'ML-IRL':
            return "ML-IRL"
        elif dir_name.upper() == 'PIRO':
            return "PIRO"
        else:
            return dir_name.replace('_', ' ').replace('-', ' ').title()

def analyze_experiment_rewards(exp_path):
    """
    Analyze reward metrics from a single experiment's progress.csv
    """
    progress_file = os.path.join(exp_path, "progress.csv")
    
    try:
        df = pd.read_csv(progress_file)
        
        # Extract key metrics
        metrics = {}
        
        if 'Real Det Return' in df.columns:
            det_returns = df['Real Det Return'].dropna()
            metrics['det_return_mean'] = det_returns.mean()
            metrics['det_return_std'] = det_returns.std()
            metrics['det_return_max'] = det_returns.max()
            metrics['det_return_final'] = det_returns.iloc[-1] if len(det_returns) > 0 else np.nan
            metrics['det_return_data'] = det_returns.tolist()
        
        if 'Real Sto Return' in df.columns:
            sto_returns = df['Real Sto Return'].dropna()
            metrics['sto_return_mean'] = sto_returns.mean()
            metrics['sto_return_std'] = sto_returns.std()
            metrics['sto_return_max'] = sto_returns.max()
            metrics['sto_return_final'] = sto_returns.iloc[-1] if len(sto_returns) > 0 else np.nan
            metrics['sto_return_data'] = sto_returns.tolist()
            
        if 'Reward Loss' in df.columns:
            reward_losses = df['Reward Loss'].dropna()
            metrics['reward_loss_mean'] = reward_losses.mean()
            metrics['reward_loss_std'] = reward_losses.std()
            metrics['reward_loss_final'] = reward_losses.iloc[-1] if len(reward_losses) > 0 else np.nan
            metrics['reward_loss_data'] = reward_losses.tolist()
            
        if 'Itration' in df.columns:
            metrics['iterations'] = df['Itration'].max() + 1
        
        metrics['experiment_path'] = exp_path
        metrics['label'] = get_experiment_label(exp_path)
        
        return metrics
        
    except Exception as e:
        print(f"Error analyzing {exp_path}: {e}")
        return None

def print_environment_summary(env_name, env_data):
    """
    Print a summary of all experiments for a given environment
    """
    print(f"\n{'='*60}")
    print(f"ENVIRONMENT: {env_name.upper()}")
    print(f"{'='*60}")
    
    for alg_name, exp_paths in env_data.items():
        print(f"\nAlgorithm: {alg_name}")
        print(f"{'-'*40}")
        
        for i, exp_path in enumerate(exp_paths):
            metrics = analyze_experiment_rewards(exp_path)
            if metrics is None:
                continue
                
            print(f"\n  Experiment {i+1}: {metrics['label']}")
            print(f"    Path: {os.path.relpath(exp_path)}")
            
            if 'iterations' in metrics:
                print(f"    Iterations: {metrics['iterations']}")
            
            if 'det_return_mean' in metrics:
                print(f"    Det Return: mean={metrics['det_return_mean']:.2f} ± {metrics['det_return_std']:.2f}, "
                      f"max={metrics['det_return_max']:.2f}, final={metrics['det_return_final']:.2f}")
            
            if 'sto_return_mean' in metrics:
                print(f"    Sto Return: mean={metrics['sto_return_mean']:.2f} ± {metrics['sto_return_std']:.2f}, "
                      f"max={metrics['sto_return_max']:.2f}, final={metrics['sto_return_final']:.2f}")
            
            if 'reward_loss_mean' in metrics:
                print(f"    Reward Loss: mean={metrics['reward_loss_mean']:.2f} ± {metrics['reward_loss_std']:.2f}, "
                      f"final={metrics['reward_loss_final']:.2f}")

def create_summary_table(all_experiments):
    """
    Create a summary table comparing all experiments
    """
    print(f"\n{'='*80}")
    print("CROSS-ENVIRONMENT SUMMARY")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Environment':<12} {'Algorithm':<15} {'Experiment':<20} {'Det Final':<10} {'Sto Final':<10} {'Det Max':<10}")
    print(f"{'-'*80}")
    
    for env_name, env_data in all_experiments.items():
        for alg_name, exp_paths in env_data.items():
            for exp_path in exp_paths:
                metrics = analyze_experiment_rewards(exp_path)
                if metrics is None:
                    continue
                
                det_final = f"{metrics.get('det_return_final', 0):.1f}" if 'det_return_final' in metrics else "N/A"
                sto_final = f"{metrics.get('sto_return_final', 0):.1f}" if 'sto_return_final' in metrics else "N/A"
                det_max = f"{metrics.get('det_return_max', 0):.1f}" if 'det_return_max' in metrics else "N/A"
                
                label = metrics['label'][:18] + ".." if len(metrics['label']) > 20 else metrics['label']
                
                print(f"{env_name:<12} {alg_name:<15} {label:<20} {det_final:<10} {sto_final:<10} {det_max:<10}")

def save_detailed_results(all_experiments, output_file="reward_analysis_results.txt"):
    """
    Save detailed results to a text file
    """
    with open(output_file, 'w') as f:
        f.write("Detailed Reward Analysis Results\n")
        f.write("="*50 + "\n\n")
        
        for env_name, env_data in all_experiments.items():
            f.write(f"Environment: {env_name.upper()}\n")
            f.write("-"*30 + "\n")
            
            for alg_name, exp_paths in env_data.items():
                f.write(f"\nAlgorithm: {alg_name}\n")
                
                for i, exp_path in enumerate(exp_paths):
                    metrics = analyze_experiment_rewards(exp_path)
                    if metrics is None:
                        continue
                    
                    f.write(f"\n  Experiment {i+1}: {metrics['label']}\n")
                    f.write(f"    Path: {os.path.relpath(exp_path)}\n")
                    
                    if 'det_return_mean' in metrics:
                        f.write(f"    Det Return: mean={metrics['det_return_mean']:.2f}, "
                               f"std={metrics['det_return_std']:.2f}, "
                               f"max={metrics['det_return_max']:.2f}, "
                               f"final={metrics['det_return_final']:.2f}\n")
                    
                    if 'sto_return_mean' in metrics:
                        f.write(f"    Sto Return: mean={metrics['sto_return_mean']:.2f}, "
                               f"std={metrics['sto_return_std']:.2f}, "
                               f"max={metrics['sto_return_max']:.2f}, "
                               f"final={metrics['sto_return_final']:.2f}\n")
                    
                    if 'reward_loss_mean' in metrics:
                        f.write(f"    Reward Loss: mean={metrics['reward_loss_mean']:.2f}, "
                               f"std={metrics['reward_loss_std']:.2f}, "
                               f"final={metrics['reward_loss_final']:.2f}\n")
            
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"Detailed results saved to: {output_file}")

def main():
    print("Analyzing reward performance across all environments and algorithms...")
    
    # Find all experiments
    all_experiments = find_all_experiments()
    
    if not all_experiments:
        print("No experiments found!")
        return
    
    # Print summary for each environment
    for env_name, env_data in all_experiments.items():
        print_environment_summary(env_name, env_data)
    
    # Print cross-environment summary table
    create_summary_table(all_experiments)
    
    # Save detailed results
    save_detailed_results(all_experiments)
    
    # Print final statistics
    total_experiments = sum(len(exp_paths) for env_data in all_experiments.values() 
                           for exp_paths in env_data.values())
    print(f"\nTotal experiments analyzed: {total_experiments}")
    print(f"Environments: {list(all_experiments.keys())}")

if __name__ == "__main__":
    main()
