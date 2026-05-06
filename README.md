# SafeHarbor: Memory4Safety

SafeHarbor is a **memory-augmented safety guardrail** for LLM agents. It maintains
a hierarchical *Risk Tree* of past attack patterns together with a *Safety
Projector* that decouples safety-relevant directions from semantic ones in the
embedding space. At inference time, the proxy retrieves the most relevant
historical risk evidence for the current request and injects it as additional
safety context for the target LLM, **without any fine-tuning of the underlying
model**.

This repository contains the code needed to (a) **train** the SafeHarbor memory
artefacts on the AgentAlign safety dataset, and (b) **evaluate** the resulting
guardrail on **AgentHarm** and **Agent-SafetyBench**, plus a small set of
baselines (RAG, A-Mem, GuardAgent, Llama Guard) we compare against in the paper.

> **Pre-built artefacts are now shipped with the repo.** As of this commit the
> three core artefacts needed to reproduce SafeHarbor at evaluation time are
> committed directly under `src/`:
>
> | Artefact | Path | Size |
> | --- | --- | --- |
> | Pre-built Risk Tree (pickled `RiskTree`) | `src/final_memory_after_benign_calibration.pkl` | ~19 MB |
> | Human-readable rule dump (companion of the pkl) | `src/final_memory_after_benign_calibration_rules.json` | ~2.7 MB |
> | Trained Safety Projector weights | `src/models/safety_projector.pth` | ~1 MB |
>
> The pkl is the one you should load at runtime; the `_rules.json` is a flat
> JSON dump of every cluster's `defense_strategy` and `benign_boundary_rule`
> for inspection / auditing. You can still rebuild everything from scratch by
> following [§3 Rebuilding the Risk Tree](#3-rebuilding-the-risk-tree).

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Setup](#1-setup)
3. [Smoke test](#2-smoke-test)
4. [Rebuilding the Risk Tree](#3-rebuilding-the-risk-tree)
5. [Running AgentHarm](#4-running-agentharm)
6. [Running Agent-SafetyBench (ASB)](#5-running-agent-safetybench-asb)
7. [Switching defenses](#6-switching-defenses)
8. [License & citation](#license--citation)

---

## Repository layout

```
.
├── agentharm.py                # Inspect-AI task entry: AgentHarm harmful + benign
├── proxy_server.py             # Drop-in OpenAI proxy that injects SafeHarbor / baselines
├── run_agentharm.sh            # Convenience runner for the full AgentHarm sweep
├── prompts.py / scorer.py / metric.py / utils.py / __init__.py
│
├── agents/                     # Agent solvers (default / refusal / guardagent)
├── benchmark/                  # AgentHarm tools (harmful_tools, benign_tools, grading_*.py)
│
├── src/                        # SafeHarbor core
│   ├── README.md
│   ├── risk_tree.py            #   Risk Tree (memory) implementation
│   ├── SafetyProjector.py      #   Dual-head safety projector (encoder + classifier)
│   ├── attacker.py             #   Build script: generate attack data via vLLM
│   ├── memory_defender.py      #   Build script: evolve the Risk Tree from attacks
│   ├── llama_guard.py          #   Llama Guard helper (also used as a baseline)
│   ├── final_memory_after_benign_calibration.pkl        ★ pre-built Risk Tree (shipped, ~19 MB)
│   ├── final_memory_after_benign_calibration_rules.json ★ companion rule dump (shipped, ~2.7 MB)
│   └── models/safety_projector.pth                       ★ trained Safety Projector (shipped, ~1 MB)
│
├── baselines/
│   ├── README.md
│   ├── rag_baseline.py         # FAISS + AgentAlign RAG baseline
│   └── guardagent/             # GuardAgent baseline (Xiang et al., 2024)
│
├── A_mem/                      # A-Mem library (used by the a_mem baseline)
├── AgentAlign/                 # AgentAlign README (training data lives outside the repo)
├── Agent-SafetyBench/          # Agent-SafetyBench (second benchmark)
├── tests/test_smoke.py         # Offline smoke test (see §2)
├── _deprecated/                # Old scratch scripts / runtime artefacts (gitignored)
└── logs/                       # Runtime logs (gitignored)
```

The single most important files are:

| File | Role |
| --- | --- |
| `src/risk_tree.py` | Hierarchical risk memory + retrieval |
| `src/SafetyProjector.py` | Embedding projector (decouples safety from semantics) |
| `src/final_memory_after_benign_calibration.pkl` | Pre-built memory (loaded at server start) |
| `src/models/safety_projector.pth` | Pre-trained projector weights |
| `proxy_server.py` | OpenAI-compatible proxy that wraps the target LLM with SafeHarbor |
| `agentharm.py` | Inspect-AI task definition for AgentHarm |

---

## 1. Setup

```bash
git clone <this repo>
cd Memory4Safety

conda create -n safeharbor python=3.10 -y
conda activate safeharbor
pip install -r requirements.txt
```

You also need an **upstream LLM** that speaks the OpenAI chat-completions API.
Any of the following works out of the box:

- A local vLLM instance: `vllm serve Qwen/Qwen2.5-72B-Instruct --port 8040`
- DeepSeek API (`https://api.deepseek.com/v1`)
- OpenAI API (`https://api.openai.com/v1`)
- Mistral La Plateforme (`https://api.mistral.ai/v1`)

The proxy will forward all requests to it.

### Required external assets

| Asset | Where to get it | Where to put it |
| --- | --- | --- |
| AgentHarm dataset | downloaded automatically by `inspect_ai` from `ai-safety-institute/AgentHarm` | (cache) |
| AgentAlign dataset | `agent_align_data_v3.json` (v3, not v2) — request from the AgentAlign authors or download from their HF repo | `./AgentAlign/agent_align_data_v3.json` |
| Agent-SafetyBench dataset | already vendored under `Agent-SafetyBench/data/` | – |
| Pre-built SafeHarbor memory | **shipped** at `src/final_memory_after_benign_calibration.pkl` (rebuild via §3 if you want a fresh one) | `src/final_memory_after_benign_calibration.pkl` |
| Companion rules dump | **shipped** at `src/final_memory_after_benign_calibration_rules.json` (auto-regenerated whenever the pkl is rebuilt) | `src/final_memory_after_benign_calibration_rules.json` |
| Pre-trained Safety Projector | **shipped** at `src/models/safety_projector.pth` (rebuild via §3.1 if you want a fresh one) | `src/models/safety_projector.pth` |

---

## 2. Smoke test

Before doing anything else, run the offline smoke test to verify the layout
and that all imports work:

```bash
python -m tests.test_smoke
# or
pytest tests/test_smoke.py -v
```

What it checks:

- All required source files exist.
- No hard-coded API keys leaked back into tracked code.
- Lightweight modules import cleanly (`scorer`, `utils`, `agents.*`, …).
- `risk_tree`, `SafetyProjector`, `proxy_server` all import in a subprocess
  (so that any native-extension crash does not bring down the rest).
- A synthetic `SafetyProjector` forward pass returns the expected shapes.
- If `src/models/safety_projector.pth` is present, it loads + has the
  classifier head.
- If `src/final_memory_after_benign_calibration.pkl` is present, it loads
  via `RiskTree.load`. (Skipped when the underlying `all-MiniLM-L6-v2`
  model can't be fetched from HuggingFace — this is OK.)

A clean pass looks like `Ran 9 tests ... OK (skipped=N)`.

---

## 3. Rebuilding the Risk Tree

The full pipeline has **three** stages. You only need to do this once; the
output `final_memory_after_benign_calibration.pkl` is consumed by both
AgentHarm and ASB at evaluation time.

### 3.1 Train the Safety Projector

The projector is a small two-layer MLP that maps the 384-d
`all-MiniLM-L6-v2` embedding to a 128-d "safety-aware" space, plus a
binary classifier head. It is trained with a hybrid Triplet + BCE loss on
triplets mined from AgentAlign.

```bash
cd src

# Optional: point at a custom data path (default: ../AgentAlign/agent_align_data_v3.json)
export AGENT_ALIGN_PATH=../AgentAlign/agent_align_data_v3.json

# Will save weights to ./models/safety_projector.pth
python SafetyProjector.py \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-3 \
    --margin 0.5
```

GPU is preferred but CPU works (slower). A full run on AgentAlign-v3 takes
roughly **5 min on a single A100** and **~30 min on CPU**.

### 3.2 Generate mutated attacks

The attacker takes harmful AgentAlign samples, runs them through one of four
red-team strategies (benign decomposition / argument injection / scenario
disguise / format-shift) using a *strong* upstream LLM (Qwen2.5-72B in our
paper), and keeps only those mutations whose intent survives a verifier
LLM check.

```bash
cd src

# Point the attacker at your upstream LLM (vLLM / OpenAI-compatible).
export ATTACKER_LLM_BASE_URL=http://localhost:8040/v1
export ATTACKER_LLM_API_KEY=EMPTY
export ATTACKER_LLM_MODEL=Qwen2.5-72B-Instruct
export ATTACKER_DATA_PATH=../AgentAlign/agent_align_data_v3.json
export ATTACKER_OUTPUT_PATH=./attack_results.json
export ATTACKER_MAX_WORKERS=8

python attacker.py
```

Output: `src/attack_results.json` (~25 MB, varies with success rate).

### 3.3 Evolve the Risk Tree (Phase 1 + 2)

`memory_defender.py` consumes `attack_results.json` and evolves the
hierarchical Risk Tree (Phase 1: defense-strategy generation per cluster),
then calibrates it against benign AgentAlign samples (Phase 2: matrix
benign-injection).

```bash
cd src

# These default to local vLLM @ 8040 with model Qwen2.5-72B-Instruct.
export RISK_TREE_LLM_BASE_URL=http://localhost:8040/v1
export RISK_TREE_LLM_API_KEY=EMPTY
export RISK_TREE_LLM_MODEL=Qwen2.5-72B-Instruct

# First-time build (Phase 1 + Phase 2):
python memory_defender.py \
    --attack_results_path ./attack_results.json \
    --agent_align_path ../AgentAlign/agent_align_data_v3.json \
    --output ./final_memory_after_benign_calibration.pkl

# Quick smoke build (50 attack + 50 benign samples, ~5 min):
python memory_defender.py --small_batch \
    --output ./final_memory_smoke.pkl

# Resume from an existing tree, skipping Phase 1 (only re-runs Phase 2):
python memory_defender.py \
    --load_pkl ./final_memory_after_benign_calibration.pkl \
    --output ./final_memory_after_benign_calibration.pkl

# Strip benign embeddings from a pkl (ablation):
python memory_defender.py \
    --clear_benign ./final_memory_after_benign_calibration.pkl \
    --output ./final_memory_cleaned.pkl
```

**Time budget on a single A100 + Qwen2.5-72B (vLLM, TP=4):**

- Phase 1 (attack evolution, ~3 k attacks, 40 threads): **~45 min**
- Phase 2 (benign calibration, batched matrix ops): **~5 min**
- Total: **< 1 h**

After this, `src/final_memory_after_benign_calibration.pkl` and
`src/models/safety_projector.pth` are both ready and the smoke test in §2
will load them successfully.

---

## 4. Running AgentHarm

`proxy_server.py` is a drop-in OpenAI proxy that:

1. (Optionally) routes incoming chat requests through Llama-Guard / GuardAgent
   first.
2. Looks up the top-k most relevant entries from a memory system
   (SafeHarbor / RAG / A-Mem) and injects them as a leading system message.
3. Forwards the augmented request to the upstream LLM.
4. Logs the full request/response trace to `./logs/`.

### 4.1 Start the proxy

```bash
# Terminal 1 — the proxy
export MEMORY_SYSTEM_TYPE=memory_tree   # use SafeHarbor
export LLM_PORT=8040                    # upstream LLM port (e.g. vLLM)
export SERVER_PORT=8055                 # where the proxy listens
export TARGET_AI_API_KEY=EMPTY          # upstream API key (use real key for OpenAI/DS)

# Optional: also enable Llama Guard prompt-level filter on top of memory.
# export ENABLE_LLAMA_GUARD=true
# export LLAMA_GUARD_PORT=$LLM_PORT
# export LLAMA_GUARD_MODEL=qwen-7b      # or anything you serve

python proxy_server.py
```

You should see:

```
✓ 当前任务类型: harmful (可通过环境变量 TASK_TYPE 修改)
✓ Memory System: Memory Tree 已初始化 (总耗时: …s)
✓ Safety Projector 已成功加载并启用
```

### 4.2 Run the benchmark

```bash
# Terminal 2 — the AgentHarm sweep
export EVAL_MODEL=gpt-4o                # any OpenAI-compatible model name your upstream serves
export LLM_PORT=8040
export SERVER_PORT=8055

# OPENAI_BASE_URL is set by the script to point at the proxy.
./run_agentharm.sh
```

That script runs both halves of AgentHarm:

```bash
TASK_TYPE=harmful inspect eval agentharm.py@agentharm \
    --model openai/${EVAL_MODEL} --max-connections 20 --max-tokens 16384

TASK_TYPE=benign  inspect eval agentharm.py@agentharm_benign \
    --model openai/${EVAL_MODEL} --max-connections 20 --max-tokens 16384
```

Results land under `./logs/` as `*.eval` files (open with `inspect view`)
and as `harmful_log_<timestamp>_memory_tree.jsonl` / same for benign.

### 4.3 Useful runtime knobs

| Env var | Default | Effect |
| --- | --- | --- |
| `MEMORY_SYSTEM_TYPE` | unset | One of `memory_tree`, `rag`, `a_mem` |
| `MEMORY_TREE_PATH` | `src/final_memory_after_benign_calibration.pkl` | Override the pkl |
| `ENABLE_LLAMA_GUARD` | `false` | Run Llama-Guard CoT prefilter |
| `ENABLE_GUARDAGENT` | `false` | Run GuardAgent baseline prefilter |
| `LLM_PORT` | `7001` | Upstream LLM port |
| `SERVER_PORT` | `8055` | Proxy listen port |
| `NUM_EXPERIMENTS` | `1` | How many full sweeps `run_agentharm.sh` does |
| `MAX_CONNECTIONS` | `20` | inspect-ai parallelism |
| `MAX_TOKENS` | `16384` | Per-response token budget |

---

## 5. Running Agent-SafetyBench (ASB)

ASB is a **separate** benchmark (not via inspect-ai) that simulates each
agent in a Python sandbox under different threat scenarios. SafeHarbor
plugs in via the same `--memory_system_type` flag.

### 5.1 Generate agent responses

```bash
cd Agent-SafetyBench/evaluation

# Configure your model API. ASB uses a different client class per provider
# (see model_api/), all of which read OPENAI_API_KEY / OPENAI_BASE_URL.
export OPENAI_BASE_URL=http://localhost:8040/v1   # or your real provider
export OPENAI_API_KEY=EMPTY

# SafeHarbor (Memory Tree) — uses src/final_memory_after_benign_calibration.pkl
CUDA_VISIBLE_DEVICES=0 python -u eval.py \
    --model_name gpt4o \
    --greedy 1 \
    --regen_exceed 1 \
    --extra_info "" \
    --memory_system_type memory_tree \
    --max_workers 5

# Other defenses for comparison:
#   --memory_system_type rag        (uses ../../AgentAlign/agent_align_data_v3.json)
#   --memory_system_type a_mem      (requires ../../memory_system_new.pkl)
#   (omit the flag for the no-defense baseline)
```

By default `eval.py` looks for the SafeHarbor pkl at
`<repo>/src/final_memory_after_benign_calibration.pkl`. You can override
with:

```bash
export MEMORY_TREE_PATH=/abs/path/to/your.pkl
```

Generation results land under
`Agent-SafetyBench/evaluation/evaluation_results/<setup>/<model>/gen_res.json`.

### 5.2 Score the responses with ShieldAgent

ASB's official scorer is a fine-tuned LLM judge (`thu-coai/ShieldAgent`).

```bash
cd Agent-SafetyBench/score

# Option A: run the local ShieldAgent model (fast, needs a GPU).
CUDA_VISIBLE_DEVICES=0 python eval_with_shield.py \
    --model_path thu-coai/ShieldAgent \
    --filepath ../evaluation/evaluation_results/tot-gpt4o \
    --filename gen_res.json \
    --target_model_name gpt4o \
    --batch_size 40

# Option B: query a remote ShieldAgent endpoint via async HTTP.
#   Both the URL and API key are env-driven, e.g.:
#   export ASB_JUDGE_API_URL=http://localhost:8031/v1/chat/completions
#   export ASB_JUDGE_API_KEY=EMPTY
python eval.py \
    --filepath ../evaluation/evaluation_results/tot-gpt4o \
    --filename gen_res.json \
    --target_model_name gpt4o
```

The score table is printed to stdout and saved under
`./shield_results/<target_model_name>/`.

---

## 6. Switching defenses

| Defense | Flag(s) |
| --- | --- |
| **SafeHarbor (ours, Memory Tree)** | `MEMORY_SYSTEM_TYPE=memory_tree` |
| RAG baseline                       | `MEMORY_SYSTEM_TYPE=rag` |
| A-Mem baseline                     | `MEMORY_SYSTEM_TYPE=a_mem` |
| Llama Guard                        | `ENABLE_LLAMA_GUARD=true` |
| GuardAgent                         | `ENABLE_GUARDAGENT=true` |
| **No defense (raw LLM)**           | leave all of the above unset |

The same flags work for both `proxy_server.py` (AgentHarm) and
`Agent-SafetyBench/evaluation/eval.py` (ASB, via `--memory_system_type`).
You can stack memory + Llama-Guard if desired.

---

## License & citation

This project is released under the **MIT License**, with the additional clause
inherited from AgentHarm: do not use the dataset / benchmark to do anything
other than improve the safety and security of AI systems. See `LICENSE`.

If you use this code, please cite the SafeHarbor paper (citation forthcoming)
as well as the upstream benchmarks and baselines we build upon:

```bibtex
@inproceedings{agentharm,
  title={AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents},
  author={Andriushchenko, Maksym and Souly, Alexandra and Dziemian, Mateusz and Duenas, Derek and Lin, Maxwell and Wang, Justin and Hendrycks, Dan and Zou, Andy and Kolter, J Zico and Fredrikson, Matt and others},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2024}
}

@article{asb,
  title={Agent-SafetyBench: Evaluating the Safety of LLM Agents},
  author={Zhang, Zhexin and Cui, Shiyao and Lu, Yida and Zhou, Jingzhuo and Yang, Junxiao and Wang, Hongning and Huang, Minlie},
  journal={arXiv preprint arXiv:2412.14470},
  year={2024}
}

@misc{llamaguard3,
  title={Llama Guard 3 8B},
  author={Meta AI},
  year={2024},
  howpublished={\url{https://huggingface.co/meta-llama/Llama-Guard-3-8B}},
  note={Accessed: 2026-01-12}
}

@article{guardagent,
  title={GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning},
  author={Xiang, Zhen and Zheng, Linzhi and Li, Yanjie and Hong, Junyuan and Li, Qinbin and Xie, Han and Zhang, Jiawei and Xiong, Zidi and Xie, Chulin and Yang, Carl and others},
  journal={arXiv preprint arXiv:2406.09187},
  year={2024}
}

@article{amem,
  title={A-Mem: Agentic Memory for LLM Agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2502.12110},
  year={2025}
}
```

**Corresponding code locations:**

- AgentHarm (Andriushchenko et al., 2024) — `agentharm.py`, `agents/`,
  `benchmark/`, `prompts.py`, `scorer.py`, `metric.py`, `utils.py`.
- Agent-SafetyBench (Zhang et al., 2024) — `Agent-SafetyBench/`.
- Llama Guard 3 (Meta AI, 2024) — `src/llama_guard.py`.
- GuardAgent (Xiang et al., 2024) — `baselines/guardagent/`.
- A-Mem (Xu et al., 2025) — `A_mem/`.
- AgentAlign — used as the safety-training corpus.
