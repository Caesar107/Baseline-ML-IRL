#!/usr/bin/env python3
"""
GAIL训练结果统计脚本
"""

import os
import pandas as pd
import numpy as np

def analyze_gail_results():
    """分析所有GAIL训练结果"""
    
    logs_dir = "GAIL/logs"
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
            # GAIL的结果直接在实验目录下，不像BC有子目录
            run_dirs = [d for d in os.listdir(os.path.join(env_path, exp_dir)) 
                       if os.path.isdir(os.path.join(env_path, exp_dir, d))]
            if not run_dirs:
                continue
                
            # 使用最新的运行目录  
            latest_run = sorted(run_dirs)[-1]
            run_path = os.path.join(env_path, exp_dir, latest_run)
            
            # 读取progress.csv
            progress_file = os.path.join(run_path, "progress.csv")
            if os.path.exists(progress_file):
                try:
                    # 手动读取CSV文件，正确解析数据
                    with open(progress_file, 'r') as f:
                        lines = f.readlines()
                    
                    if len(lines) < 2:
                        print(f"⚠️  Empty or invalid CSV file for {env}")
                        continue
                    
                    headers = lines[0].strip().split(',')
                    print(f"Processing {env}... Headers: {headers[:6]}...")  # 只显示前6个
                    
                    if 'GAIL Mean Reward' not in headers:
                        print(f"⚠️  No 'GAIL Mean Reward' column found for {env}")
                        continue
                    
                    mean_reward_idx = headers.index('GAIL Mean Reward')
                    std_reward_idx = headers.index('GAIL Std Reward') if 'GAIL Std Reward' in headers else -1
                    
                    rewards = []
                    stds = []
                    
                    # 解析前500行数据（跳过标题行）
                    for i in range(1, min(501, len(lines))):
                        parts = lines[i].strip().split(',')
                        if len(parts) > mean_reward_idx:
                            try:
                                reward_val = float(parts[mean_reward_idx])
                                rewards.append(reward_val)
                                
                                if std_reward_idx >= 0 and len(parts) > std_reward_idx:
                                    try:
                                        std_val = float(parts[std_reward_idx])
                                        stds.append(std_val)
                                    except:
                                        stds.append(np.nan)
                            except:
                                continue
                    
                    if len(rewards) == 0:
                        print(f"⚠️  No valid reward data found for {env}")
                        continue
                    
                    print(f"Found {len(rewards)} valid reward values for {env}")
                    
                    # 获取最终结果
                    final_reward = rewards[-1]
                    final_std = stds[-1] if len(stds) > 0 else np.nan
                    
                    # 计算统计信息
                    max_reward = max(rewards)
                    min_reward = min(rewards)
                    avg_reward = sum(rewards) / len(rewards)
                    std_reward = np.std(rewards)
                    
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
                        'Total Evaluations': len(rewards),
                        'Status': 'Completed' if pd.notna(final_reward) else 'In Progress'
                    })
                    
                    print(f"✅ {env}: Final Reward = {final_reward:.2f} ± {final_std:.2f} (Evaluations: {len(rewards)})")
                        
                except Exception as e:
                    print(f"❌ Error reading {progress_file}: {e}")
            else:
                print(f"⚠️  No progress file found for {env}/{exp_dir}")
    
    if results:
        # 创建DataFrame
        df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("GAIL Training Results Summary")
        print("="*80)
        
        # 显示简化的结果表
        summary_df = df[['Environment', 'Final Reward', 'Final Std', 'Max Reward', 'Status']].copy()
        summary_df['Final Reward'] = summary_df['Final Reward'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        summary_df['Final Std'] = summary_df['Final Std'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        summary_df['Max Reward'] = summary_df['Max Reward'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        print(summary_df.to_string(index=False))
        
        # 保存详细结果
        df.to_csv('GAIL/gail_results_detailed.csv', index=False, float_format='%.2f')
        summary_df.to_csv('GAIL/gail_results_summary.csv', index=False)
        
        print(f"\n📁 Detailed results saved to: GAIL/gail_results_detailed.csv")
        print(f"📁 Summary saved to: GAIL/gail_results_summary.csv")
        
        # 环境排名
        completed_results = df[df['Status'] == 'Completed'].copy()
        if not completed_results.empty:
            completed_results['Final Reward'] = pd.to_numeric(completed_results['Final Reward'], errors='coerce')
            ranking = completed_results.sort_values('Final Reward', ascending=False)
            
            print(f"\n🏆 GAIL Performance Ranking:")
            print("-" * 40)
            for i, (_, row) in enumerate(ranking.iterrows(), 1):
                print(f"{i}. {row['Environment']}: {row['Final Reward']:.2f}")
        
        return df
    else:
        print("No GAIL training results found.")
        return None

if __name__ == "__main__":
    analyze_gail_results()
