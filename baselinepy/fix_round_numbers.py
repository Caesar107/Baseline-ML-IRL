#!/usr/bin/env python3
"""
修正CSV文件中的Round列，确保从1开始连续编号
"""

import pandas as pd
import os

def fix_round_numbers():
    result_dir = "experiment_results"
    
    total_fixed = 0
    
    # 处理AIRL和GAIL文件
    for algo in ['AIRL', 'GAIL']:
        algo_dir = os.path.join(result_dir, algo)
        if not os.path.exists(algo_dir):
            continue
            
        print(f"\n修正 {algo} 文件的Round列...")
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
                    # 读取CSV文件
                    df = pd.read_csv(file_path)
                    
                    if df.empty:
                        print(f"  跳过空文件: {file_path}")
                        continue
                    
                    # 检查是否有Round和Reward两列
                    if 'Round' not in df.columns or 'Reward' not in df.columns:
                        print(f"  跳过格式不正确的文件: {file_path}")
                        continue
                    
                    original_len = len(df)
                    
                    # 删除包含NaN或无效数据的行
                    df = df.dropna()
                    
                    if df.empty:
                        print(f"  处理后为空: {file_path}")
                        continue
                    
                    # 重新生成Round列，从1开始
                    df['Round'] = range(1, len(df) + 1)
                    
                    # 保存修正后的文件
                    df.to_csv(file_path, index=False)
                    
                    print(f"  ✅ 修正完成: {file_path}")
                    print(f"    原始行数: {original_len}, 修正后行数: {len(df)}")
                    
                    algo_count += 1
                    total_fixed += 1
                    
                except Exception as e:
                    print(f"  ❌ 修正失败: {file_path}, 错误: {e}")
        
        print(f"{algo} 算法修正完成: {algo_count} 个文件")
    
    print(f"\n====== 修正完成 ======")
    print(f"总共修正了 {total_fixed} 个CSV文件")
    print(f"所有文件的Round列现在都从1开始连续编号")

if __name__ == "__main__":
    fix_round_numbers()
