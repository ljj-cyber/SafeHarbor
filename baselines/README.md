# Baselines

This folder contains the baselines we compare SafeHarbor against. Each one is
plugged into `proxy_server.py` and selected via environment variables.

| Sub-module | Plugged in via | Notes |
| --- | --- | --- |
| `rag_baseline.py` | `MEMORY_SYSTEM_TYPE=rag` | Simple FAISS retrieval over the AgentAlign safety dataset. Requires `./AgentAlign/agent_align_data_v3.json`. |
| `guardagent/`     | `ENABLE_GUARDAGENT=true` | A clean fork of [GuardAgent](https://arxiv.org/abs/2406.09187). The internal `.git` was removed; configure via `GUARDAGENT_API_KEY` / `GUARDAGENT_API_BASE`. |

For the **A-Mem** baseline, see the top-level `A_mem/` library and set
`MEMORY_SYSTEM_TYPE=a_mem`. The A-Mem memory pickle is *not* shipped with the
repo — see `_deprecated/top_level_tests/A_mem_test.py` for an example builder
script (it expects an upstream LLM and the AgentAlign dataset).

For **Llama Guard**, see `src/llama_guard.py` and set `ENABLE_LLAMA_GUARD=true`
together with `LLAMA_GUARD_PORT` / `LLAMA_GUARD_MODEL`.
