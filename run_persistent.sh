#!/bin/bash

# 持久化运行脚本 - 支持系统重启后自动恢复
export PYTHONPATH=${PWD}:$PYTHONPATH

# 状态文件
STATE_FILE="trash/experiment_state.txt"
PROGRESS_FILE="trash/experiment_progress.txt"

# 初始化状态
mkdir -p trash
if [ ! -f "$STATE_FILE" ]; then
    echo "0" > "$STATE_FILE"  # 当前轮次
    echo "0" > "$PROGRESS_FILE"  # 当前环境索引
fi

# 读取当前状态
current_round=$(cat "$STATE_FILE")
current_env_idx=$(cat "$PROGRESS_FILE")

# 配置
environments=("ant" "halfcheetah" "hopper" "walker2d" "humanoid")
num_runs=4

echo "Resuming from Round $((current_round + 1)), Environment index $current_env_idx"

for round in $(seq $((current_round + 1)) $num_runs); do
    echo "=== Starting Round $round/$num_runs ==="
    echo "$((round - 1))" > "$STATE_FILE"
    
    timestamp=$(date +"%Y%m%d_%H%M%S")
    
    for env_idx in $(seq $current_env_idx $((${#environments[@]} - 1))); do
        env=${environments[$env_idx]}
        echo "$env_idx" > "$PROGRESS_FILE"
        
        echo "Starting experiment for $env (Round $round)..."
        
        log_file="trash/irl_${env}_round${round}_${timestamp}.log"
        pid_file="trash/irl_${env}_round${round}_${timestamp}.pid"
        
        nohup python ml/irl_samples.py configs/samples/agents/${env}.yml > "$log_file" 2>&1 &
        
        echo $! > "$pid_file"
        echo "Started $env experiment Round $round (PID: $!, Log: $log_file)"
        
        sleep 5
    done
    
    # 重置环境索引
    echo "0" > "$PROGRESS_FILE"
    
    echo "Round $round: All 5 environments started!"
    
    # 等待当前轮次完成
    if [ $round -lt $num_runs ]; then
        echo "Waiting for Round $round to complete..."
        while true; do
            running_count=$(ps aux | grep "python ml/irl_samples.py" | grep -v grep | wc -l)
            if [ $running_count -le 1 ]; then  # 考虑可能有antmaze实验
                echo "Round $round completed! Starting next round in 30 seconds..."
                sleep 30
                break
            else
                echo "Round $round: $running_count experiments still running..."
                sleep 60
            fi
        done
    fi
done

# 清理状态文件
echo "$num_runs" > "$STATE_FILE"
echo "0" > "$PROGRESS_FILE"
echo "All experiments completed!"
