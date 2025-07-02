#!/usr/bin/env python3
"""
BC训练结果统计脚本
"""

import os
import pandas as pd
import numpy as np

def analyze_bc_results():
    """分析所有BC训练结果"""
    
    logs_dir = "baselines/logs"
    results = []
    
    # 环境列表
    environments = ['HalfCheetah-v2', 'Ant-v2', 'Walker2d-v3', 'Hopper-v3', 'Humanoid-v3']
    
    for env in environments:
        env_path = os.path.join(logs_dir, env)
        if not os.path.exists(env_path):
            print(f"Warning: {env} logs not found")
            continue
            
        # 查找实验目录
        exp_dirs = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]
        if not exp_dirs:
            print(f"Warning: No experiment directories found for {env}")
            continue
            
        for exp_dir in exp_dirs:
            bc_path = os.path.join(env_path, exp_dir, "bc")
            if not os.path.exists(bc_path):
                continue
                
            # 查找BC运行目录
            run_dirs = [d for d in os.listdir(bc_path) if os.path.isdir(os.path.join(bc_path, d))]
            if not run_dirs:
                continue
                
            # 使用最新的运行目录
            latest_run = sorted(run_dirs)[-1]
            run_path = os.path.join(bc_path, latest_run)
            
            # 读取progress.csv
            progress_file = os.path.join(run_path, "progress.csv")
            if os.path.exists(progress_file):
                try:
                    df = pd.read_csv(progress_file)
                    if not df.empty:
                        # 获取最后一行（最终结果）
                        last_row = df.iloc[-1]
                        
                        # 检查是否有最终奖励
                        final_reward = last_row.get('BC Final Mean Reward', np.nan)
                        final_std = last_row.get('BC Final Std Reward', np.nan)
                        
                        if pd.isna(final_reward):
                            # 如果没有最终奖励，使用最后一次评估的奖励
                            final_reward = last_row.get('BC Mean Reward', np.nan)
                            final_std = last_row.get('BC Std Reward', np.nan)
                        
                        epochs = last_row.get('BC Epoch', np.nan)
                        trajectories = last_row.get('BC Trajectories', np.nan)
                        transitions = last_row.get('BC Transitions', np.nan)
                        
                        # 计算训练过程中的统计信息
                        reward_cols = df['BC Mean Reward'].dropna()
                        if len(reward_cols) > 0:
                            max_reward = reward_cols.max()
                            min_reward = reward_cols.min()
                            avg_reward = reward_cols.mean()
                            std_reward = reward_cols.std()
                        else:
                            max_reward = min_reward = avg_reward = std_reward = np.nan
                        
                        results.append({
                            'Environment': env,
                            'Experiment': exp_dir,
                            'Run': latest_run,
                            'Final Reward': final_reward,
                            'Final Std': final_std,
                            'Max Reward': max_reward,
                            'Min Reward': min_reward,
                            'Avg Reward': avg_reward,
                            'Reward Std': std_reward,
                            'Epochs': epochs,
                            'Trajectories': trajectories,
                            'Transitions': transitions,
                            'Total Evaluations': len(reward_cols),
                            'Status': 'Completed' if pd.notna(final_reward) else 'In Progress'
                        })
                        
                        print(f"✅ {env}: Final Reward = {final_reward:.2f} ± {final_std:.2f} (Epochs: {epochs})")
                        
                except Exception as e:
                    print(f"❌ Error reading {progress_file}: {e}")
            else:
                print(f"⚠️  No progress file found for {env}/{exp_dir}")
    
    if results:
        # 创建DataFrame
        df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("BC Training Results Summary")
        print("="*80)
        
        # 显示简化的结果表
        summary_df = df[['Environment', 'Final Reward', 'Final Std', 'Max Reward', 'Epochs', 'Status']].copy()
        summary_df['Final Reward'] = summary_df['Final Reward'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        summary_df['Final Std'] = summary_df['Final Std'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        summary_df['Max Reward'] = summary_df['Max Reward'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        print(summary_df.to_string(index=False))
        
        # 保存详细结果
        df.to_csv('baselines/bc_results_detailed.csv', index=False, float_format='%.2f')
        summary_df.to_csv('baselines/bc_results_summary.csv', index=False)
        
        print(f"\n📁 Detailed results saved to: baselines/bc_results_detailed.csv")
        print(f"📁 Summary saved to: baselines/bc_results_summary.csv")
        
        # 环境排名
        completed_results = df[df['Status'] == 'Completed'].copy()
        if not completed_results.empty:
            completed_results['Final Reward'] = pd.to_numeric(completed_results['Final Reward'], errors='coerce')
            ranking = completed_results.sort_values('Final Reward', ascending=False)
            
            print(f"\n🏆 BC Performance Ranking:")
            print("-" * 40)
            for i, (_, row) in enumerate(ranking.iterrows(), 1):
                print(f"{i}. {row['Environment']}: {row['Final Reward']:.2f}")
        
        return df
    else:
        print("No BC training results found.")
        return None

if __name__ == "__main__":
    analyze_bc_results()
