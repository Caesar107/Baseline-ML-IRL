#!/bin/bash

# 批量运行所有GAIL训练任务

echo "Starting all GAIL training tasks..."

# 获取脚本所在目录的父目录（即ML-IRL根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Working directory: $ROOT_DIR"
cd "$ROOT_DIR"

# 定义所有环境
environments=("halfcheetah" "ant" "walker2d" "hopper" "humanoid")

# 运行每个环境的GAIL训练
for env in "${environments[@]}"; do
    config_file="GAIL/configs/gail_${env}.yml"
    log_file="GAIL/trash/gail_${env}_$(date +%Y%m%d_%H%M%S).log"
    pid_file="GAIL/trash/gail_${env}.pid"
    
    echo "Starting GAIL training for $env..."
    echo "Config: $config_file"
    echo "Log file: $log_file"
    
    # 检查配置文件是否存在
    if [ -f "$config_file" ]; then
        # 使用nohup在后台运行，并重定向输出到日志文件
        nohup python GAIL/gail_trainer.py "$config_file" > "$log_file" 2>&1 &
        
        # 获取进程ID
        pid=$!
        echo "Started GAIL training for $env with PID: $pid"
        echo "Log file: $log_file"
        
        # 保存PID到trash目录
        echo "$pid" > "$pid_file"
        
        # 等待一小段时间，避免同时启动过多进程
        sleep 10
    else
        echo "Warning: Config file $config_file not found, skipping $env"
    fi
done

echo ""
echo "All GAIL training tasks started!"
echo ""
echo "To monitor progress:"
echo "  tail -f GAIL/trash/gail_<env>_*.log   # 查看特定环境的训练日志"
echo "  ps aux | grep gail_trainer           # 查看运行中的训练进程"
echo ""
echo "To stop all GAIL training:"
echo "  pkill -f gail_trainer.py             # 停止所有GAIL训练进程"
echo "  kill \$(cat GAIL/trash/gail_*.pid)    # 使用PID文件停止训练"
echo ""
echo "Log and PID files are saved in GAIL/trash/ directory"
