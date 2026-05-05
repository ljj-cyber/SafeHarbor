#!/bin/bash
# 在 AgentHarm 和 Agent-SafetyBench 上运行 GuardAgent（不使用 memory）

# 默认参数
LLM="gpt-4o"
SEED=42
OUTPUT_DIR="./guardagent_results"
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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS] [DATASET]"
            echo ""
            echo "Options:"
            echo "  --llm LLM                    LLM 模型名称 (默认: gpt-4o)"
            echo "  --seed SEED                  随机种子 (默认: 42)"
            echo "  --output_dir DIR             输出目录 (默认: ./guardagent_results)"
            echo "  --limit N                    限制处理的样本数量（用于测试）"
            echo "  --help                       显示此帮助信息"
            echo ""
            echo "Dataset:"
            echo "  agentharm                    在 AgentHarm 数据集上运行"
            echo "  asb                          在 Agent-SafetyBench 数据集上运行"
            echo "  both                         在两个数据集上运行（默认）"
            echo ""
            echo "示例:"
            echo "  $0 agentharm --limit 10"
            echo "  $0 asb --llm gpt-4"
            echo "  $0 both"
            exit 0
            ;;
        agentharm|asb|both)
            DATASET="$1"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 默认运行两个数据集
if [ -z "$DATASET" ]; then
    DATASET="both"
fi

echo "=========================================="
echo "GuardAgent 评估（Training-Free，无 Memory）"
echo "=========================================="
echo "LLM: $LLM"
echo "随机种子: $SEED"
echo "输出目录: $OUTPUT_DIR"
echo "数据集: $DATASET"
echo "样本限制: ${LIMIT:-无}"
echo "=========================================="

# 切换到 code 目录
cd "$(dirname "$0")"

# 运行 AgentHarm 评估
if [ "$DATASET" == "agentharm" ] || [ "$DATASET" == "both" ]; then
    echo ""
    echo ">>> 在 AgentHarm 数据集上评估 GuardAgent"
    echo ""
    python eval_guardagent_agentharm.py \
        --llm "$LLM" \
        --seed "$SEED" \
        --split test_public \
        --task_name harmful \
        --output_dir "${OUTPUT_DIR}/agentharm" \
        $LIMIT
fi

# 运行 ASB 评估
if [ "$DATASET" == "asb" ] || [ "$DATASET" == "both" ]; then
    echo ""
    echo ">>> 在 Agent-SafetyBench 数据集上评估 GuardAgent"
    echo ""
    python eval_guardagent_asb.py \
        --llm "$LLM" \
        --seed "$SEED" \
        --data_path "../Agent-SafetyBench/evaluation/data/test_public.json" \
        --output_dir "${OUTPUT_DIR}/asb" \
        $LIMIT
fi

echo ""
echo "=========================================="
echo "评估完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="

