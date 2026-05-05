#!/bin/bash
# GuardAgent 在 AgentAlign 数据集上构建记忆

# 默认参数
LLM="gpt-4o"
SEED=42
NUM_SHOTS=3
LOGS_PATH="./logs_agentalign"
DATASET_PATH="../agent_align_data_v3.json"
MEMORY_PATH="./guardagent_memory_agentalign.json"
LIMIT=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --llm)
            LLM="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num_shots)
            NUM_SHOTS="$2"
            shift 2
            ;;
        --logs_path)
            LOGS_PATH="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --memory_path)
            MEMORY_PATH="$2"
            shift 2
            ;;
        --load_memory)
            LOAD_MEMORY="--load_memory"
            shift
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --llm LLM                    LLM 模型名称 (默认: gpt-4)"
            echo "  --seed SEED                  随机种子 (默认: 42)"
            echo "  --num_shots SHOTS            示例数量 (默认: 3)"
            echo "  --logs_path PATH             日志路径 (默认: ./logs_agentalign)"
            echo "  --dataset_path PATH          数据集路径 (默认: ../agent_align_data_v3.json)"
            echo "  --memory_path PATH           记忆保存路径 (默认: ./guardagent_memory_agentalign.json)"
            echo "  --load_memory                加载已有记忆"
            echo "  --limit N                    限制处理的样本数量（用于测试）"
            echo "  --help                        显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --limit 10                    # 测试模式，只处理 10 个样本"
            echo "  $0 --load_memory                   # 加载已有记忆继续构建"
            echo "  $0 --memory_path custom_memory.json # 使用自定义记忆路径"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "GuardAgent - AgentAlign 记忆构建"
echo "=========================================="
echo "LLM: $LLM"
echo "随机种子: $SEED"
echo "示例数量: $NUM_SHOTS"
echo "日志路径: $LOGS_PATH"
echo "数据集: $DATASET_PATH"
echo "记忆路径: $MEMORY_PATH"
echo "加载记忆: ${LOAD_MEMORY:-否}"
echo "样本限制: ${LIMIT:-无}"
echo "=========================================="

# 运行 GuardAgent
python main_agentalign.py \
    --llm "$LLM" \
    --seed "$SEED" \
    --num_shots "$NUM_SHOTS" \
    --logs_path "$LOGS_PATH" \
    --dataset_path "$DATASET_PATH" \
    --memory_path "$MEMORY_PATH" \
    $LOAD_MEMORY \
    $LIMIT

echo ""
echo "=========================================="
echo "记忆构建完成！"
echo "记忆文件: $MEMORY_PATH"
echo "=========================================="

