#!/usr/bin/env python3
"""
配置参数对比脚本
验证BC baseline配置与主IRL配置是否对齐，确保公平比较
"""

import os
from ruamel.yaml import YAML
import pandas as pd

def load_config(config_path):
    """加载YAML配置文件"""
    yaml = YAML()
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.load(f)
    return None

def compare_configs():
    """比较主配置与BC配置的关键参数"""
    
    # 环境列表
    environments = ['halfcheetah', 'ant', 'walker2d', 'hopper', 'humanoid']
    
    # 环境名称映射
    env_name_mapping = {
        'halfcheetah': 'HalfCheetah-v2',
        'ant': 'Ant-v2', 
        'walker2d': 'Walker2d-v3',
        'hopper': 'Hopper-v3',
        'humanoid': 'Humanoid-v3'
    }
    
    comparison_data = []
    
    for env in environments:
        # 加载主配置
        main_config_path = f"/home/yche767/ML-IRL/configs/samples/agents/{env}.yml"
        main_config = load_config(main_config_path)
        
        # 加载BC配置
        bc_config_path = f"/home/yche767/ML-IRL/baselines/configs/bc_{env}.yml"
        bc_config = load_config(bc_config_path)
        
        if main_config and bc_config:
            # 提取关键参数对比
            row = {
                'Environment': env,
                'Env_Name_Main': main_config['env']['env_name'],
                'Env_Name_BC': bc_config['env']['env_name'],
                'Seed_Main': main_config['seed'],
                'Seed_BC': bc_config['seed'],
                'T_Main': main_config['env']['T'],
                'T_BC': bc_config['env']['T'],
                'BC_Epochs_Main': main_config['bc']['epochs'],
                'BC_Epochs_BC': bc_config['bc']['epochs'],
                'BC_Eval_Freq_Main': main_config['bc']['eval_freq'],
                'BC_Eval_Freq_BC': bc_config['bc']['eval_freq'],
                'BC_Eval_Episodes_Main': main_config['bc']['eval_episodes'],
                'BC_Eval_Episodes_BC': bc_config['bc']['eval_episodes'],
                'BC_Expert_Episodes_Main': main_config['bc']['expert_episodes'],
                'BC_Expert_Episodes_BC': bc_config['bc']['expert_episodes'],
                'Reward_Hidden_Main': main_config.get('reward', {}).get('hidden_sizes', 'N/A'),
                'BC_Hidden_BC': bc_config['bc_network']['hidden_sizes'],
                'Reward_LR_Main': main_config.get('reward', {}).get('lr', 'N/A'),
                'BC_LR_BC': bc_config['bc_network']['lr'],
                'SAC_Batch_Main': main_config.get('sac', {}).get('batch_size', 'N/A'),
                'BC_Batch_BC': bc_config['bc_network']['batch_size'],
            }
            
            comparison_data.append(row)
        else:
            print(f"Warning: Could not load configs for {env}")
    
    # 创建DataFrame并显示
    df = pd.DataFrame(comparison_data)
    
    print("=== 配置参数对比 ===")
    print("验证BC baseline配置与主IRL配置的对齐情况\n")
    
    # 基本信息对比
    print("1. 基本配置对比:")
    basic_cols = ['Environment', 'Env_Name_Main', 'Env_Name_BC', 'Seed_Main', 'Seed_BC', 'T_Main', 'T_BC']
    print(df[basic_cols].to_string(index=False))
    print()
    
    # BC参数对比
    print("2. BC参数对比:")
    bc_cols = ['Environment', 'BC_Epochs_Main', 'BC_Epochs_BC', 'BC_Eval_Freq_Main', 'BC_Eval_Freq_BC', 
               'BC_Eval_Episodes_Main', 'BC_Eval_Episodes_BC', 'BC_Expert_Episodes_Main', 'BC_Expert_Episodes_BC']
    print(df[bc_cols].to_string(index=False))
    print()
    
    # 网络结构对比
    print("3. 网络结构对比:")
    network_cols = ['Environment', 'Reward_Hidden_Main', 'BC_Hidden_BC', 'Reward_LR_Main', 'BC_LR_BC', 
                   'SAC_Batch_Main', 'BC_Batch_BC']
    print(df[network_cols].to_string(index=False))
    print()
    
    # 验证对齐情况
    print("4. 对齐验证:")
    misaligned = []
    
    for _, row in df.iterrows():
        env = row['Environment']
        
        # 检查关键参数是否对齐
        if row['Env_Name_Main'] != row['Env_Name_BC']:
            misaligned.append(f"{env}: Environment name mismatch")
        
        if row['T_Main'] != row['T_BC']:
            misaligned.append(f"{env}: Episode length (T) mismatch")
            
        if row['BC_Epochs_Main'] != row['BC_Epochs_BC']:
            misaligned.append(f"{env}: BC epochs mismatch")
            
        if row['BC_Eval_Freq_Main'] != row['BC_Eval_Freq_BC']:
            misaligned.append(f"{env}: BC eval frequency mismatch")
            
        if row['BC_Eval_Episodes_Main'] != row['BC_Eval_Episodes_BC']:
            misaligned.append(f"{env}: BC eval episodes mismatch")
            
        if row['BC_Expert_Episodes_Main'] != row['BC_Expert_Episodes_BC']:
            misaligned.append(f"{env}: BC expert episodes mismatch")
            
        if str(row['Reward_Hidden_Main']) != str(row['BC_Hidden_BC']):
            misaligned.append(f"{env}: Network hidden sizes mismatch")
            
        if row['Reward_LR_Main'] != row['BC_LR_BC']:
            misaligned.append(f"{env}: Learning rate mismatch")
            
        if row['SAC_Batch_Main'] != row['BC_Batch_BC']:
            misaligned.append(f"{env}: Batch size mismatch")
    
    if misaligned:
        print("⚠️  发现对齐问题:")
        for issue in misaligned:
            print(f"   {issue}")
    else:
        print("✅ 所有关键参数都已对齐！")
        print("   BC baseline配置与主IRL配置匹配，可以进行公平比较。")
    
    # 保存对比结果
    output_file = "/home/yche767/ML-IRL/baselines/config_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\n详细对比结果已保存到: {output_file}")

if __name__ == "__main__":
    compare_configs()
