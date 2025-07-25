#!/bin/bash

# 监控所有GAIL训练任务的进度

echo "=== GAIL Training Progress Monitor ==="
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
# 查看运行中的GAIL训练进程
gail_processes=$(ps aux | grep "[g]ail_trainer.py" | wc -l)
if [ $gail_processes -gt 0 ]; then
    ps aux | grep "[g]ail_trainer.py" | awk '{print $2, $11, $12, $13, $14}'
    echo "Total GAIL processes running: $gail_processes"
else
    echo "No GAIL training processes currently running."
fi

echo ""
echo "2. PID Files Status:"
echo "-------------------"
for env in "${environments[@]}"; do
    pid_file="GAIL/trash/gail_${env}.pid"
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
    latest_log=$(ls -t GAIL/trash/gail_${env}_*.log 2>/dev/null | head -1)
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
    result_dirs=$(find GAIL/logs/ -name "*${env}*" -type d 2>/dev/null)
    if [ -n "$result_dirs" ]; then
        latest_dir=$(echo "$result_dirs" | sort | tail -1)
        result_file="$latest_dir/gail_results.json"
        if [ -f "$result_file" ]; then
            echo "$env: Training completed"
            # 提取最终奖励
            mean_after=$(grep -o '"mean_reward_after": [0-9.-]*' "$result_file" | cut -d: -f2 | tr -d ' ')
            improvement=$(grep -o '"improvement": [0-9.-]*' "$result_file" | cut -d: -f2 | tr -d ' ')
            echo "  Final reward: $mean_after"
            echo "  Improvement: $improvement"
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
    echo "  watch -n 30 ./GAIL/monitor_gail.sh       # 每30秒自动刷新"
    echo "  tail -f GAIL/trash/gail_<env>_*.log      # 实时查看特定环境日志"
    echo "  kill \$(cat GAIL/trash/gail_<env>.pid)    # 停止特定环境训练"
fi
