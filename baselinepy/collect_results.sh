#!/bin/bash

# 结果收集脚本 - 按算法和环境分类整理CSV文件

echo "开始收集所有实验结果..."

# 创建目标目录结构
RESULT_DIR="experiment_results"
mkdir -p "$RESULT_DIR"

# 创建各算法目录
mkdir -p "$RESULT_DIR/BC"
mkdir -p "$RESULT_DIR/AIRL"
mkdir -p "$RESULT_DIR/GAIL"

# 创建环境子目录
for algo in BC AIRL GAIL; do
    mkdir -p "$RESULT_DIR/$algo/Ant"
    mkdir -p "$RESULT_DIR/$algo/HalfCheetah"
    mkdir -p "$RESULT_DIR/$algo/Hopper"
    mkdir -p "$RESULT_DIR/$algo/Humanoid"
    mkdir -p "$RESULT_DIR/$algo/Walker2d"
done

echo "目录结构创建完成！"

# 收集BC结果
echo "收集BC实验结果..."
bc_count=0

# BC结果在BC/BC_log/logs目录下
if [ -d "BC/BC_log/logs" ]; then
    for env_dir in BC/BC_log/logs/*/; do
        env_name=$(basename "$env_dir")
        # 标准化环境名称
        case $env_name in
            "Ant-v2") clean_env="Ant" ;;
            "HalfCheetah-v2") clean_env="HalfCheetah" ;;
            "Hopper-v3") clean_env="Hopper" ;;
            "Humanoid-v3") clean_env="Humanoid" ;;
            "Walker2d-v3") clean_env="Walker2d" ;;
            *) clean_env=$(echo $env_name | sed 's/-v[0-9]//') ;;
        esac
        
        # 查找该环境下的所有BC实验CSV文件
        for csv_file in "$env_dir"exp-*/bc/*/progress.csv; do
            if [ -f "$csv_file" ]; then
                # 提取时间戳作为文件名
                timestamp=$(echo "$csv_file" | grep -o '[0-9]\{4\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}')
                target_file="$RESULT_DIR/BC/$clean_env/bc_${clean_env,,}_${timestamp}.csv"
                cp "$csv_file" "$target_file"
                echo "BC: $csv_file -> $target_file"
                ((bc_count++))
            fi
        done
    done
fi

# 收集AIRL结果
echo "收集AIRL实验结果..."
airl_count=0

if [ -d "AIRL/logs" ]; then
    for env_dir in AIRL/logs/*/; do
        env_name=$(basename "$env_dir")
        # 标准化环境名称
        case $env_name in
            "Ant-v2") clean_env="Ant" ;;
            "HalfCheetah-v2") clean_env="HalfCheetah" ;;
            "Hopper-v3") clean_env="Hopper" ;;
            "Humanoid-v3") clean_env="Humanoid" ;;
            "Walker2d-v3") clean_env="Walker2d" ;;
            *) clean_env=$(echo $env_name | sed 's/-v[0-9]//') ;;
        esac
        
        # 查找该环境下的所有AIRL实验CSV文件
        for csv_file in "$env_dir"exp-*/*/progress.csv; do
            if [ -f "$csv_file" ]; then
                # 提取时间戳作为文件名
                timestamp=$(echo "$csv_file" | grep -o '[0-9]\{4\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}')
                target_file="$RESULT_DIR/AIRL/$clean_env/airl_${clean_env,,}_${timestamp}.csv"
                cp "$csv_file" "$target_file"
                echo "AIRL: $csv_file -> $target_file"
                ((airl_count++))
            fi
        done
    done
fi

# 收集GAIL结果
echo "收集GAIL实验结果..."
gail_count=0

if [ -d "GAIL/logs" ]; then
    for env_dir in GAIL/logs/*/; do
        env_name=$(basename "$env_dir")
        # 标准化环境名称
        case $env_name in
            "Ant-v2") clean_env="Ant" ;;
            "HalfCheetah-v2") clean_env="HalfCheetah" ;;
            "Hopper-v3") clean_env="Hopper" ;;
            "Humanoid-v3") clean_env="Humanoid" ;;
            "Walker2d-v3") clean_env="Walker2d" ;;
            *) clean_env=$(echo $env_name | sed 's/-v[0-9]//') ;;
        esac
        
        # 查找该环境下的所有GAIL实验CSV文件
        for csv_file in "$env_dir"exp-*/*/progress.csv; do
            if [ -f "$csv_file" ]; then
                # 提取时间戳作为文件名
                timestamp=$(echo "$csv_file" | grep -o '[0-9]\{4\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}')
                target_file="$RESULT_DIR/GAIL/$clean_env/gail_${clean_env,,}_${timestamp}.csv"
                cp "$csv_file" "$target_file"
                echo "GAIL: $csv_file -> $target_file"
                ((gail_count++))
            fi
        done
    done
fi

# 生成统计报告
echo ""
echo "====== 实验结果收集完成 ======"
echo "BC实验文件数: $bc_count"
echo "AIRL实验文件数: $airl_count"
echo "GAIL实验文件数: $gail_count"
echo "总文件数: $((bc_count + airl_count + gail_count))"
echo ""
echo "结果保存在: $RESULT_DIR/"
echo ""

# 显示目录结构
echo "目录结构："
tree "$RESULT_DIR" 2>/dev/null || find "$RESULT_DIR" -type d | sed 's|[^/]*/|  |g'

echo ""
echo "可以使用以下命令查看每个算法的结果："
echo "ls $RESULT_DIR/BC/*/  # BC结果"
echo "ls $RESULT_DIR/AIRL/*/ # AIRL结果"
echo "ls $RESULT_DIR/GAIL/*/ # GAIL结果"
