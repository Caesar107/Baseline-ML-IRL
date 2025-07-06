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

if [ "$MODE" = "sequential" ]; then
    echo "Running experiments sequentially..."
    for env in "${environments[@]}"; do
        echo "Running experiment for $env..."
        python ml/irl_samples.py configs/samples/agents/${env}.yml
        echo "Completed $env experiment"
    done
    echo "All experiments completed!"
else
    echo "Running experiments in parallel..."
    for env in "${environments[@]}"; do
        echo "Starting experiment for $env..."
        python ml/irl_samples.py configs/samples/agents/${env}.yml &
        echo "Started $env experiment in background (PID: $!)"
        
        # 等待一小段时间避免同时启动过多进程
        sleep 5
    done
    
    echo "All experiments started in background!"
    echo "Use 'jobs' to check running jobs"
    echo "Use 'ps aux | grep irl_samples' to check all processes"
    echo "Use 'wait' to wait for all background jobs to complete"
fi

#MUJOCO TRANSFER
#python common/train_optimal.py configs/samples/experts/ant_transfer.yml
#python ml/irl_samples.py configs/samples/agents/data_transfer.yml(data transfer)