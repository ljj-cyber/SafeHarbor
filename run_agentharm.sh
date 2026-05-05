#!/bin/bash
# Run AgentHarm benchmark (harmful + benign) through the SafeHarbor proxy.
#
# Required environment variables:
#   EVAL_MODEL        e.g. "gpt-4o" or "Qwen2.5-72B-Instruct"
#   SERVER_PORT       port the SafeHarbor proxy_server.py listens on (default: 8055)
#   LLM_PORT          port the upstream LLM (vLLM / Mistral / OpenAI proxy) listens on
#
# Optional environment variables:
#   NUM_EXPERIMENTS   how many full (harmful+benign) runs to do (default: 1)
#   MAX_CONNECTIONS   inspect-ai parallelism (default: 20)
#   MAX_TOKENS        per-response token budget (default: 16384)
#
# Memory system / guard configuration is set by exporting variables that
# proxy_server.py reads, e.g.:
#   MEMORY_SYSTEM_TYPE=memory_tree   # one of: memory_tree | rag | a_mem | (unset)
#   ENABLE_LLAMA_GUARD=false
#   ENABLE_GUARDAGENT=false
#
# Usage:
#   EVAL_MODEL=gpt-4o LLM_PORT=8025 SERVER_PORT=8055 \
#   MEMORY_SYSTEM_TYPE=memory_tree ./run_agentharm.sh

set -e

: "${EVAL_MODEL:?Please export EVAL_MODEL (e.g. gpt-4o)}"
: "${SERVER_PORT:=8055}"
: "${LLM_PORT:?Please export LLM_PORT (the upstream LLM port)}"
NUM_EXPERIMENTS=${NUM_EXPERIMENTS:-1}
MAX_CONNECTIONS=${MAX_CONNECTIONS:-20}
MAX_TOKENS=${MAX_TOKENS:-16384}

# Route the inspect-ai client through our proxy_server.py
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export OPENAI_BASE_URL="http://localhost:${SERVER_PORT}/v1"

echo "=========================================="
echo "AgentHarm sweep ($NUM_EXPERIMENTS run(s))"
echo "  EVAL_MODEL          = $EVAL_MODEL"
echo "  proxy server port   = $SERVER_PORT"
echo "  upstream LLM port   = $LLM_PORT"
echo "  MEMORY_SYSTEM_TYPE  = ${MEMORY_SYSTEM_TYPE:-<none>}"
echo "  ENABLE_LLAMA_GUARD  = ${ENABLE_LLAMA_GUARD:-false}"
echo "  ENABLE_GUARDAGENT   = ${ENABLE_GUARDAGENT:-false}"
echo "=========================================="

for i in $(seq 1 "$NUM_EXPERIMENTS"); do
    echo
    echo "[run $i/$NUM_EXPERIMENTS] $(date '+%Y-%m-%d %H:%M:%S')"

    echo "[run $i] harmful split"
    TASK_TYPE="harmful" inspect eval agentharm.py@agentharm \
        --model "openai/${EVAL_MODEL}" \
        --max-connections "$MAX_CONNECTIONS" \
        --max-tokens "$MAX_TOKENS"

    echo "[run $i] benign split"
    TASK_TYPE="benign" inspect eval agentharm.py@agentharm_benign \
        --model "openai/${EVAL_MODEL}" \
        --max-connections "$MAX_CONNECTIONS" \
        --max-tokens "$MAX_TOKENS"

    if [ "$i" -lt "$NUM_EXPERIMENTS" ]; then
        sleep 5
    fi
done

echo "=========================================="
echo "All runs finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
