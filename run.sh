export PYTHONPATH=${PWD}:$PYTHONPATH
# UNIT TEST

#MUJOCO DATA COLLECT
#python common/train_gd.py configs/samples/experts/gd.yml
#python common/collect.py configs/samples/experts/gd.
#python common/train_gd.py configs/samples/experts/hopper.yml
#python common/train_gd.py configs/samples/experts/halfcheetah.yml
#python common/train_gd.py configs/samples/experts/walker2d.yml
#python common/collect.py configs/samples/experts/humanoid.yml

# MUJOCO IRL BENCHMARK - 批量运行所有环境
echo "Starting IRL experiments for all environments..."

# 选择运行模式：并行(parallel)或串行(sequential)
MODE=${1:-"parallel"}  # 默认并行模式

# 定义所有要运行的环境
environments=("ant" "halfcheetah" "hopper" "walker2d" "humanoid")

# 运行次数（连续4次实验）
num_runs=4

if [ "$MODE" = "sequential" ]; then
    echo "Running experiments sequentially..."
    for env in "${environments[@]}"; do
        echo "Running experiment for $env..."
        python ml/irl_samples.py configs/samples/agents/${env}.yml
        echo "Completed $env experiment"
    done
    echo "All experiments completed!"
else
    echo "Running experiments in parallel (offline mode) - $num_runs rounds..."
    echo "Script will run completely in background. You can safely close your computer after execution."
    
    # 创建trash目录（如果不存在）
    mkdir -p trash
    
    # 将整个实验流程放到后台运行
    (
        for round in $(seq 1 $num_runs); do
            echo "=== Starting Round $round/$num_runs ===" >> trash/experiment_log.txt
            
            # 获取当前时间戳
            timestamp=$(date +"%Y%m%d_%H%M%S")
            
            for env in "${environments[@]}"; do
                echo "Starting experiment for $env (Round $round)..." >> trash/experiment_log.txt
                
                # 定义日志文件，包含轮次信息
                log_file="trash/irl_${env}_round${round}_${timestamp}.log"
                pid_file="trash/irl_${env}_round${round}_${timestamp}.pid"
                
                # 使用nohup在后台运行，重定向输出到日志文件
                nohup python ml/irl_samples.py configs/samples/agents/${env}.yml > "$log_file" 2>&1 &
                
                # 保存PID
                echo $! > "$pid_file"
                echo "Started $env experiment Round $round (PID: $!, Log: $log_file)" >> trash/experiment_log.txt
                
                # 等待一小段时间避免同时启动过多进程
                sleep 5
            done
            
            echo "Round $round: All 5 environments started in background!" >> trash/experiment_log.txt
            
            # 如果不是最后一轮，等待当前轮次完成再开始下一轮
            if [ $round -lt $num_runs ]; then
                echo "Waiting for Round $round to complete before starting Round $((round+1))..." >> trash/experiment_log.txt
                
                # 等待当前轮次的所有进程完成 - 通过PID文件检查
                while true; do
                    running_count=0
                    for env in "${environments[@]}"; do
                        current_pid_file="trash/irl_${env}_round${round}_${timestamp}.pid"
                        if [ -f "$current_pid_file" ]; then
                            pid=$(cat "$current_pid_file")
                            if kill -0 "$pid" 2>/dev/null; then
                                ((running_count++))
                            fi
                        fi
                    done
                    
                    if [ $running_count -eq 0 ]; then
                        echo "Round $round completed! Starting next round in 30 seconds..." >> trash/experiment_log.txt
                        sleep 30
                        break
                    else
                        echo "Round $round: $running_count experiments still running... checking again in 60 seconds" >> trash/experiment_log.txt
                        sleep 60
                    fi
                done
            fi
        done
        
        echo "All $num_runs rounds completed!" >> trash/experiment_log.txt
        echo "$(date): All experiments finished successfully" >> trash/experiment_log.txt
    ) &
    
    # 保存主控制脚本的PID
    echo $! > trash/main_controller.pid
    
    echo "All experiments started in offline mode!"
    echo "Main controller PID: $! (saved to trash/main_controller.pid)"
    echo "Experiment progress logged to: trash/experiment_log.txt"
    echo "Individual logs saved in: trash/irl_*_round*_*.log"
    echo "PID files saved in: trash/irl_*_round*_*.pid"
    echo ""
    echo "✅ You can safely close your computer now!"
    echo "Use 'tail -f trash/experiment_log.txt' to monitor overall progress"
    echo "Use 'tail -f trash/irl_<env>_round<N>_*.log' to monitor specific experiment"
fi

#MUJOCO TRANSFER
#python common/train_optimal.py configs/samples/experts/ant_transfer.yml
#python ml/irl_samples.py configs/samples/agents/data_transfer.yml(data transfer)