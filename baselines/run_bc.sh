#!/bin/bash
# BC baseline training script
# 参数与主配置对齐，确保公平比较

echo "=== BC Baseline Training ==="
echo "Parameters aligned with main IRL configs for fair comparison"
echo ""

# Set PYTHONPATH
export PYTHONPATH="/home/yche767/ML-IRL:$PYTHONPATH"

# Define available environments
ENVIRONMENTS=("ant" "halfcheetah" "hopper" "humanoid" "walker2d")

# Function to run BC training for a single environment
run_bc_training() {
    local env=$1
    local config_file="configs/bc_${env}.yml"
    
    echo "=== Training BC for ${env} ==="
    echo "Config file: $config_file"
    
    if [ ! -f "$config_file" ]; then
        echo "Error: Config file $config_file not found!"
        return 1
    fi
    
    echo "Starting BC training with SB3..."
    python bc_sb3_trainer.py "$config_file"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ BC training completed successfully for $env"
    else
        echo "✗ BC training failed for $env (exit code: $exit_code)"
    fi
    
    echo "================================================"
    return $exit_code
}

# Main execution logic
if [ $# -eq 0 ]; then
    # No arguments: run all environments
    echo "No environment specified. Running BC training for all environments..."
    echo "Environments to train: ${ENVIRONMENTS[*]}"
    echo ""
    
    # Track results
    declare -a successful_envs
    declare -a failed_envs
    start_time=$(date)
    
    for env in "${ENVIRONMENTS[@]}"; do
        if run_bc_training "$env"; then
            successful_envs+=("$env")
        else
            failed_envs+=("$env")
        fi
        echo ""
    done
    
    # Print summary
    echo "=== BC Training Summary ==="
    echo "Start time: $start_time"
    echo "End time: $(date)"
    echo ""
    
    if [ ${#successful_envs[@]} -gt 0 ]; then
        echo "✓ Successful environments (${#successful_envs[@]}/${#ENVIRONMENTS[@]}):"
        for env in "${successful_envs[@]}"; do
            echo "  - $env"
            model_dir="logs/${env}_bc"
            if [ -d "$model_dir" ]; then
                echo "    Model saved in: $model_dir"
            fi
        done
        echo ""
    fi
    
    if [ ${#failed_envs[@]} -gt 0 ]; then
        echo "✗ Failed environments (${#failed_envs[@]}/${#ENVIRONMENTS[@]}):"
        for env in "${failed_envs[@]}"; do
            echo "  - $env"
        done
        echo ""
        echo "Please check the logs above for error details."
    fi
    
    echo "All BC baseline training runs completed."
    
elif [ $# -eq 1 ]; then
    # Single environment specified
    ENV=$1
    
    # Check if environment is valid
    if [[ ! " ${ENVIRONMENTS[*]} " =~ " ${ENV} " ]]; then
        echo "Error: Invalid environment '$ENV'"
        echo "Available environments: ${ENVIRONMENTS[*]}"
        echo ""
        echo "Usage: $0 [environment]"
        echo "  No argument: Run all environments"
        echo "  Single argument: Run specific environment"
        exit 1
    fi
    
    run_bc_training "$ENV"
    
else
    echo "Error: Too many arguments"
    echo "Usage: $0 [environment]"
    echo "Available environments: ${ENVIRONMENTS[*]}"
    echo ""
    echo "Examples:"
    echo "  $0           # Run all environments"
    echo "  $0 ant       # Run only Ant environment"
    exit 1
fi
