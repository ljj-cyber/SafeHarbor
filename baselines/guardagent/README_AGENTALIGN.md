# GuardAgent 在 AgentAlign 数据集上构建记忆

## 概述

这个脚本用于在 AgentAlign 数据集上运行 GuardAgent 来构建防御记忆。构建的记忆可以保存为 JSON 文件，并在后续任务中复用。

## 记忆格式

记忆以 JSON 格式保存，每个记忆项包含：
```json
{
  "agent input": "用户请求内容",
  "agent output": "助手响应内容",
  "subtasks": "任务分解结果",
  "code": "生成的防御代码"
}
```

## 使用方法

### 1. 首次运行（构建新记忆）

```bash
cd /home/beihang/AndyLLM/agentharm/code
bash run_guardagent_agentalign.sh
```

### 2. 测试模式（只处理少量样本）

```bash
bash run_guardagent_agentalign.sh --limit 10
```

### 3. 加载已有记忆继续构建

```bash
bash run_guardagent_agentalign.sh --load_memory
```

### 4. 使用自定义记忆路径

```bash
bash run_guardagent_agentalign.sh --memory_path ./my_memory.json
```

## 记忆复用

### 方式 1：在同一脚本中复用

使用 `--load_memory` 参数即可加载已有记忆：

```bash
python main_agentalign.py --load_memory --memory_path ./guardagent_memory_agentalign.json
```

### 方式 2：在其他脚本中复用

在其他 Python 脚本中加载记忆：

```python
import json

# 加载记忆
with open('./guardagent_memory_agentalign.json', 'r', encoding='utf-8') as f:
    memory = json.load(f)

# 使用记忆初始化 GuardAgent
user_proxy.update_memory(num_shots=3, memory=memory)
```

### 方式 3：合并多个记忆文件

```python
import json

# 加载多个记忆文件
memory1 = json.load(open('memory1.json', 'r', encoding='utf-8'))
memory2 = json.load(open('memory2.json', 'r', encoding='utf-8'))

# 合并记忆（去重）
combined_memory = memory1.copy()
existing_inputs = {item['agent input'] for item in memory1}

for item in memory2:
    if item['agent input'] not in existing_inputs:
        combined_memory.append(item)
        existing_inputs.add(item['agent input'])

# 保存合并后的记忆
with open('combined_memory.json', 'w', encoding='utf-8') as f:
    json.dump(combined_memory, f, indent=2, ensure_ascii=False)
```

## 参数说明

- `--llm`: LLM 模型名称（默认: gpt-4）
- `--seed`: 随机种子（默认: 42）
- `--num_shots`: 示例数量，用于 few-shot 学习（默认: 3）
- `--logs_path`: 日志保存路径（默认: ./logs_agentalign）
- `--dataset_path`: AgentAlign 数据集路径（默认: ../agent_align_data_v3.json）
- `--memory_path`: 记忆保存/加载路径（默认: ./guardagent_memory_agentalign.json）
- `--load_memory`: 是否加载已有记忆
- `--limit`: 限制处理的样本数量（用于测试）

## 记忆的自动保存

- 每处理 10 个样本会自动保存一次中间记忆（防止意外中断丢失数据）
- 处理完成后会保存最终记忆

## 注意事项

1. **记忆文件格式**：记忆以 JSON 格式保存，确保文件编码为 UTF-8
2. **记忆去重**：当前版本不会自动去重，如果需要可以手动处理
3. **记忆大小**：随着记忆增长，检索速度可能会变慢，建议定期清理低质量记忆
4. **兼容性**：记忆格式与原始 GuardAgent 兼容，可以在不同任务间复用

## 示例：在不同任务间复用记忆

```python
# 任务 1：在 AgentAlign 上构建记忆
python main_agentalign.py --memory_path memory_agentalign.json

# 任务 2：在其他数据集上使用已有记忆
python main_other.py --load_memory --memory_path memory_agentalign.json
```

