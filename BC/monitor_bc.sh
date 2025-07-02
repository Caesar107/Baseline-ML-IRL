#!/bin/bash

# 监控所有BC训练任务的进度 cd /home/yche767/ML-IRL && BC/monitor_bc.sh

echo "=== BC Training Progress Monitor ==="
echo "Time: $(date)"
echo ""

# 获取脚本所在目录的父目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# 定义所有环境
environments=("halfcheetah" "ant" "walker2d" "hopper" "humanoid")

echo "1. Running Processes:"
echo "-------------------"
# 查看运行中的BC训练进程
bc_processes=$(ps aux | grep "[b]c_imitation_trainer.py" | wc -l)
if [ $bc_processes -gt 0 ]; then
    ps aux | grep "[b]c_imitation_trainer.py" | awk '{print $2, $11, $12, $13, $14}'
    echo "Total BC processes running: $bc_processes"
else
    echo "No BC training processes currently running."
fi

echo ""
echo "2. PID Files Status:"
echo "-------------------"
for env in "${environments[@]}"; do
    pid_file="bc_${env}.pid"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "$env: PID $pid (RUNNING)"
        else
            echo "$env: PID $pid (STOPPED)"
        fi
    else
        echo "$env: No PID file found"
    fi
done

echo ""
echo "3. Recent Log Activity:"
echo "----------------------"
for env in "${environments[@]}"; do
    # 查找最新的日志文件
    latest_log=$(ls -t bc_${env}_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "$env ($latest_log):"
        # 显示最后几行，查看训练进度
        tail -3 "$latest_log" | head -2 | sed 's/^/  /'
        echo ""
    else
        echo "$env: No log file found"
    fi
done

echo "4. Training Results Summary:"
echo "---------------------------"
for env in "${environments[@]}"; do
    # 查找结果文件
    result_dirs=$(find baselines/logs/ -name "*${env}*" -type d 2>/dev/null)
    if [ -n "$result_dirs" ]; then
        latest_dir=$(echo "$result_dirs" | sort | tail -1)
        result_file="$latest_dir/bc_results.json"
        if [ -f "$result_file" ]; then
            echo "$env: Training completed"
            # 提取最终奖励
            final_reward=$(grep -o '"final_mean_reward": [0-9.-]*' "$result_file" | cut -d: -f2 | tr -d ' ')
            echo "  Final reward: $final_reward"
        else
            echo "$env: Training in progress or no results yet"
        fi
    else
        echo "$env: No training directory found"
    fi
done

echo ""
echo "=== End Monitor ==="

# 如果作为脚本运行，提供一些有用的命令
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    echo ""
    echo "Useful commands:"
    echo "  watch -n 30 ./baselines/monitor_bc.sh    # 每30秒自动刷新"
    echo "  tail -f bc_<env>_*.log                   # 实时查看特定环境日志"
    echo "  kill \$(cat bc_<env>.pid)                 # 停止特定环境训练"
fi
