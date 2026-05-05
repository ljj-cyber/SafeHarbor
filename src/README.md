# SafeHarbor core (`src/`)

The actual SafeHarbor method lives here.

| File | Purpose |
| --- | --- |
| `risk_tree.py` | Hierarchical Risk Tree memory + retrieval (the main object that gets pickled) |
| `SafetyProjector.py` | Dual-head projector: encoder branch + safety classifier head |
| `attacker.py` | Build script: generates mutated attacks via an upstream vLLM |
| `memory_defender.py` | Build script: turns attacks into Risk-Tree nodes and trains the projector |
| `llama_guard.py` | Helper used by the Llama Guard baseline (also re-used at evaluation time) |

## Pre-built artifacts

Both files below are loaded automatically by `proxy_server.py` when
`MEMORY_SYSTEM_TYPE=memory_tree`:

| File | Size | What it is |
| --- | --- | --- |
| `final_memory_after_benign_calibration.pkl` | ~145 MB | The latest Risk Tree, post benign-data calibration. |
| `models/safety_projector.pth` | ~1 MB | Trained Safety Projector weights. |

If you want to rebuild them from scratch, see the *Rebuilding the Risk Tree*
section in the top-level `README.md`.

## Configuration env vars

| Variable | Default | Used by |
| --- | --- | --- |
| `RISK_TREE_LLM_BASE_URL` | `http://127.0.0.1:8040/v1` | `risk_tree.py`, `memory_defender.py` |
| `RISK_TREE_LLM_API_KEY`  | `EMPTY` | same |
| `RISK_TREE_LLM_MODEL`    | `Qwen2.5-72B-Instruct` | `memory_defender.py` |
| `ATTACKER_LLM_BASE_URL`  | `http://localhost:8040/v1` | `attacker.py` |
| `ATTACKER_LLM_MODEL`     | `Qwen2.5-72B-Instruct` | `attacker.py` |
| `ATTACKER_DATA_PATH`     | `../AgentAlign/agent_align_data_v3.json` | `attacker.py` |
