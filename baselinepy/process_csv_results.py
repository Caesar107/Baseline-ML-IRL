#!/usr/bin/env python3
"""
处理实验结果CSV文件，提取Round和Reward两列
- AIRL: AIRL Round, AIRL Mean Reward -> Round, Reward  
- GAIL: GAIL Round, GAIL Mean Reward -> Round, Reward
- BC: BC Mean Reward -> Reward (手动添加Round列)
"""

import os
import pandas as pd
import numpy as np
import csv
import csv

def process_csv_files():
    result_dir = "experiment_results"
    
    if not os.path.exists(result_dir):
        print(f"错误: 目录 {result_dir} 不存在")
        return
    
    total_processed = 0
    
    # 处理每个算法
    for algo in ['BC', 'AIRL', 'GAIL']:
        algo_dir = os.path.join(result_dir, algo)
        if not os.path.exists(algo_dir):
            print(f"跳过不存在的目录: {algo_dir}")
            continue
            
        print(f"\n处理 {algo} 算法结果...")
        algo_count = 0
        
        # 处理每个环境
        for env in ['Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'Walker2d']:
            env_dir = os.path.join(algo_dir, env)
            if not os.path.exists(env_dir):
                continue
                
            # 处理该环境下的所有CSV文件
            for filename in os.listdir(env_dir):
                if not filename.endswith('.csv'):
                    continue
                    
                file_path = os.path.join(env_dir, filename)
                try:
                    # 根据算法类型处理
                    if algo == 'BC':
                        # BC: 不用修改，跳过
                        print(f"  跳过BC文件: {file_path}")
                        continue
                            
                    elif algo == 'AIRL':
                        # AIRL: 使用原始CSV解析方法
                        rewards = []
                        rounds = []
                        
                        try:
                            with open(file_path, 'r') as f:
                                reader = csv.reader(f)
                                header = next(reader)  # 读取表头
                                
                                # 找到AIRL Mean Reward的列索引
                                reward_col_idx = None
                                for i, col_name in enumerate(header):
                                    if col_name.strip() == 'AIRL Mean Reward':
                                        reward_col_idx = i
                                        break
                                
                                if reward_col_idx is None:
                                    print(f"  警告: {file_path} 没有找到AIRL Mean Reward列")
                                    continue
                                
                                # 跳过第一行（有问题的数据），从第二行开始读取
                                first_row = next(reader)  # 跳过第一行
                                
                                # 读取所有数据
                                for row_num, row in enumerate(reader, start=2):  # 从第二行开始计数
                                    if reward_col_idx < len(row):
                                        reward_val = row[reward_col_idx]
                                        
                                        try:
                                            reward_float = float(reward_val)
                                            rewards.append(reward_float)
                                            rounds.append(row_num)
                                        except ValueError:
                                            continue  # 跳过无法转换的值
                                
                                if rewards:
                                    processed_df = pd.DataFrame({
                                        'Round': rounds,
                                        'Reward': rewards
                                    })
                                else:
                                    print(f"  跳过处理后为空的文件: {file_path}")
                                    continue
                                    
                        except Exception as e:
                            print(f"  ❌ 处理AIRL文件失败: {file_path}, 错误: {e}")
                            continue
                            
                    elif algo == 'GAIL':
                        # GAIL: 使用原始CSV解析方法
                        rewards = []
                        rounds = []
                        
                        try:
                            with open(file_path, 'r') as f:
                                reader = csv.reader(f)
                                header = next(reader)  # 读取表头
                                
                                # 找到GAIL Mean Reward的列索引
                                reward_col_idx = None
                                for i, col_name in enumerate(header):
                                    if col_name.strip() == 'GAIL Mean Reward':
                                        reward_col_idx = i
                                        break
                                
                                if reward_col_idx is None:
                                    print(f"  警告: {file_path} 没有找到GAIL Mean Reward列")
                                    continue
                                
                                # 跳过第一行（有问题的数据），从第二行开始读取
                                first_row = next(reader)  # 跳过第一行
                                
                                # 读取所有数据
                                for row_num, row in enumerate(reader, start=2):  # 从第二行开始计数
                                    if reward_col_idx < len(row):
                                        reward_val = row[reward_col_idx]
                                        
                                        try:
                                            reward_float = float(reward_val)
                                            rewards.append(reward_float)
                                            rounds.append(row_num)
                                        except ValueError:
                                            continue  # 跳过无法转换的值
                                
                                if rewards:
                                    processed_df = pd.DataFrame({
                                        'Round': rounds,
                                        'Reward': rewards
                                    })
                                else:
                                    print(f"  跳过处理后为空的文件: {file_path}")
                                    continue
                                    
                        except Exception as e:
                            print(f"  ❌ 处理GAIL文件失败: {file_path}, 错误: {e}")
                            continue
                    
                    # 删除包含NaN的行
                    processed_df = processed_df.dropna()
                    
                    if processed_df.empty:
                        print(f"  跳过处理后为空的文件: {file_path}")
                        continue
                    
                    # 保存处理后的文件
                    processed_df.to_csv(file_path, index=False)
                    print(f"  ✅ 处理完成: {file_path} ({len(processed_df)} 行)")
                    algo_count += 1
                    total_processed += 1
                    
                except Exception as e:
                    print(f"  ❌ 处理失败: {file_path}, 错误: {e}")
        
        print(f"{algo} 算法处理完成: {algo_count} 个文件")
    
    print(f"\n====== 处理完成 ======")
    print(f"总共处理了 {total_processed} 个CSV文件")
    print(f"所有文件现在都包含统一的 'Round' 和 'Reward' 两列")

if __name__ == "__main__":
    process_csv_files()
