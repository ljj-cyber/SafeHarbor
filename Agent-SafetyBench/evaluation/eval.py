import json
import sys
from tqdm import tqdm
from copy import deepcopy
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
sys.path.append('../environments')
# 添加项目根目录到路径，以便导入 RAG_test 和 A_mem
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from redirect_output import redirect_output
from EnvManager import EnvManager
from model_api import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt4o")
parser.add_argument('--greedy', type=int, default=1)
parser.add_argument("--regen_exceed", type=int, default=0)
parser.add_argument('--extra_info', type=str, default='')
parser.add_argument('--allow_empty', type=int, default=0)
parser.add_argument('--max_workers', type=int, default=5, help='并发数量')
parser.add_argument('--memory_system_type', type=str, default=None, 
                    choices=["rag", "memory_tree", "a_mem"], 
                    help='Memory system type: "rag", "memory_tree", or "a_mem" (default: None, no memory system)')


args = parser.parse_args()

env_manager = EnvManager()
# stop_token = '[FINISHED]'

model_name = args.model_name

# ============================================================================
# Memory System 初始化
# ============================================================================
memory_system = None
query_rag = None
query_mem = None

if args.memory_system_type == "rag":
    try:
        from baselines.rag_baseline import init_RAG, query_rag as _query_rag
        from baselines import rag_baseline
        # Override DATA_PATH to point at the project-root-relative dataset.
        rag_data_path = os.path.join(project_root, "AgentAlign", "agent_align_data_v3.json")
        if not os.path.exists(rag_data_path):
            print(f"⚠️ RAG 数据文件不存在: {rag_data_path}")
            memory_system = None
            query_rag = None
        else:
            rag_baseline.DATA_PATH = rag_data_path
            query_rag = _query_rag
            memory_system = init_RAG()
            print("✓ Memory System: RAG 已初始化")
    except Exception as e:
        print(f"⚠️ RAG 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        memory_system = None
        query_rag = None

elif args.memory_system_type == "memory_tree":
    try:
        # Make `src/` importable so the pickle can find the RiskTree class.
        src_path = os.path.abspath(os.path.join(project_root, 'src'))
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        from risk_tree import RiskTree

        memory_tree_path = os.environ.get(
            "MEMORY_TREE_PATH",
            os.path.abspath(
                os.path.join(project_root, "src", "final_memory_after_benign_calibration.pkl")
            ),
        )
        if os.path.exists(memory_tree_path):
            safety_projector_path = os.path.abspath(
                os.path.join(project_root, "src", "models", "safety_projector.pth")
            )
            memory_system = RiskTree()
            memory_system = memory_system.load(
                memory_tree_path,
                safety_projector_path=safety_projector_path
                if os.path.exists(safety_projector_path)
                else None,
            )
            
            # 如果加载的对象没有 score_log_file 属性，设置默认值
            if not hasattr(memory_system, 'score_log_file') or memory_system.score_log_file is None:
                from datetime import datetime
                log_dir = os.path.join(project_root, "logs")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"score_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
                memory_system.score_log_file = log_file
                print(f"✓ Set default score_log_file for loaded tree: {memory_system.score_log_file}")
            if not hasattr(memory_system, '_score_log_count'):
                memory_system._score_log_count = 0
            
            print(f"✓ Memory System: Memory Tree 已初始化 (从 {memory_tree_path})")
        else:
            print(f"⚠️ Memory Tree pickle 文件不存在: {memory_tree_path}")
            memory_system = None
    except Exception as e:
        print(f"⚠️ Memory Tree 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        memory_system = None

elif args.memory_system_type == "a_mem":
    try:
        from A_mem.agentic_memory.memory_system import AgenticMemorySystem
        from baselines.rag_baseline import query_mem as _query_mem
        query_mem = _query_mem
        memory_system_path = os.environ.get(
            "A_MEM_PKL_PATH",
            os.path.abspath(os.path.join(project_root, "memory_system_new.pkl")),
        )
        if os.path.exists(memory_system_path):
            with open(memory_system_path, "rb") as f:
                memory_system = pickle.load(f)
            print(f"✓ Memory System: A_mem 已初始化 (从 {memory_system_path})")
        else:
            print(f"⚠️ A_mem pickle 文件不存在: {memory_system_path}")
            memory_system = None
            query_mem = None
    except Exception as e:
        print(f"⚠️ A_mem 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        memory_system = None
        query_mem = None

else:
    print("✓ Memory System: 未启用 (memory_system_type=None)")

if args.greedy:
    print("Use greedy decoding!")
    api_model_generation_config = {
        "temperature": 0.,
        "max_tokens": 2048
    }
    local_model_generation_config = {
        "do_sample": False,
        "repetition_penalty": 1.0,
        "num_beams": 1,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "use_cache": True,
        "max_new_tokens": 2048,
    }
else:
    print("Use random sampling!")
    api_model_generation_config = {
        "temperature": 1.0,
        "max_tokens": 2048
    }
    local_model_generation_config = {
        "do_sample": True,
        # "repetition_penalty": 1.0,
        "num_beams": 1,
        # "temperature": 1.0,
        # "top_k": 0,
        # "top_p": 1.0,
        "use_cache": True,
        "max_new_tokens": 2048,
    }

if model_name == 'gpt4omini':
    agent_api = OpenaiAPI(model_name='gpt-4o-mini', generation_config=api_model_generation_config)
elif model_name == 'gpt4o':
    agent_api = OpenaiAPI(model_name='gpt-4o', generation_config=api_model_generation_config)
elif model_name == 'gpt4-turbo':
    agent_api = OpenaiAPI(model_name='gpt-4-turbo', generation_config=api_model_generation_config)
elif model_name == 'claude3-opus':
    agent_api = ClaudeAPI(model_name='anthropic/claude-3-opus', generation_config=api_model_generation_config)
elif model_name == 'claude3.5-sonnet':
    agent_api = ClaudeAPI(model_name='anthropic/claude-3.5-sonnet', generation_config=api_model_generation_config)
elif model_name == 'claude3.5-haiku':
    agent_api = ClaudeAPI(model_name='anthropic/claude-3.5-haiku', generation_config=api_model_generation_config)
elif model_name == 'gemini1.5-pro':
    agent_api = GeminiAPI(model_name='google/gemini-pro-1.5', generation_config=api_model_generation_config)
elif model_name == 'gemini1.5-flash':
    agent_api = GeminiAPI(model_name='google/gemini-flash-1.5', generation_config=api_model_generation_config)
elif model_name == 'qwen2.5-72b':
    agent_api = QwenCloudAPI(model_name='qwen/qwen-2.5-72b-instruct', generation_config=api_model_generation_config)
elif model_name == 'deepseekv2.5':
    agent_api = DeepseekAPI(model_name='deepseek/deepseek-chat', generation_config=api_model_generation_config)
elif model_name == 'llama3.1-70b':
    agent_api = LlamaCloudAPI(model_name='meta-llama/llama-3.1-70b-instruct', generation_config=api_model_generation_config)
elif model_name == 'llama3.1-405b':
    agent_api = LlamaCloudAPI(model_name='meta-llama/llama-3.1-405b-instruct', generation_config=api_model_generation_config)

# local model
elif model_name == 'qwen2.5-7b':
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    agent_api = QwenAPI(model_path=model_path, tokenizer_path=model_path, generation_config=local_model_generation_config)
elif model_name == 'qwen2.5-14b':
    model_path = "Qwen/Qwen2.5-14B-Instruct"
    agent_api = QwenAPI(model_path=model_path, tokenizer_path=model_path, generation_config=local_model_generation_config)
elif model_name == 'llama3.1-8b':
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    agent_api = Llama3API(model_path=model_path, tokenizer_path=model_path, generation_config=local_model_generation_config)
elif model_name == 'glm4-9b':
    model_path = "THUDM/glm-4-9b-chat"
    agent_api = GLM4API(model_path=model_path, tokenizer_path=model_path, generation_config=local_model_generation_config)
else:
    raise ValueError(f"Model {model_name} not found.")

error_log_path = './error_samples.json'
exceed_log_path = './exceed_round_samples.json'

def parse_envs(envs_info):
    envs = []
    tool_descs = []
    for env_info in envs_info:
        env_name = env_info['name']
        if 'parameters' in env_info:
            env_params = env_info['parameters'] or None
        else:
            env_params = None
        env = env_manager.init_env(env_name, env_params)
        if env is None:
            print(envs_info)
            raise ValueError(f"Environment {env_name} not found.")
        envs.append(env)
        
        tool_names = env_info['tools']
        tool_desc = env.get_tool_descs(tool_names)
        tool_descs.extend(tool_desc)
        
    return envs, tool_descs

def remove_exceed_samples(outpath):
    try:
        with open(outpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        with open(exceed_log_path, 'r', encoding='utf-8') as f:
            exceed_samples = json.load(f)
    except Exception as e:
        # print(e)
        return
    
    exceed_ids = set([d['info']['id'] for d in exceed_samples])
    new_data = []
    for d in data:
        if d['id'] in exceed_ids:
            continue
        new_data.append(d)
        
    print(f'remove {len(data) - len(new_data)} exceed samples')
    
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=1, ensure_ascii=False)

def log_error_samples(messages, info):
    try:
        with open(error_log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        data = []
    data.append({'messages': messages, 'info': info})
    with open(error_log_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_exceed_round_samples(messages, info):
    try:
        with open(exceed_log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        data = []
    data.append({"messages": messages, "info": info})
    with open(exceed_log_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def clear_log(clear_error=True, clear_exceed=True):
    if clear_error:
        try:
            with open(error_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
        except:
            pass
    if clear_exceed:
        try:
            with open(exceed_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
        except:
            pass

def evaluate(d):
    envs = []
    tool_descs = []

    max_rounds = 10
    rounds = 0
    final_result = None

    info = {"id": d["id"]}
    try:
        messages = [{"role": "system", "content": agent_api.get_system_prompt(d)}]

        if "dialog" in d:
            messages.extend(d["dialog"])
        else:
            messages.append({"role": "user", "content": d["instruction"]})

        # Memory System 检索逻辑
        if args.memory_system_type == "memory_tree" and memory_system is not None:
            # Memory Tree: 检索并组装 message
            # 注意：retrieve_query 只接受 messages 参数，不需要 catagories
            if messages is not None:
                messages = memory_system.retrieve_query(messages)
                print(f"[Memory] Memory Tree 检索完成，增强后的 messages 长度: {len(messages)}")
            else:
                print("[Warning] messages is None, skipping memory retrieval.")
        
        elif args.memory_system_type == "rag" and memory_system is not None and query_rag is not None:
            # RAG: 使用 query_rag 检索并增强 messages
            messages = query_rag(memory_system, messages, k=3)
            print(f"[Memory] RAG 检索完成，增强后的 messages 长度: {len(messages)}")
        
        elif args.memory_system_type == "a_mem" and memory_system is not None and query_mem is not None:
            # A_mem: 使用 query_mem 检索并增强 messages
            messages, query = query_mem(memory_system, messages, k=3)
            print(f"[Memory] A_mem 检索完成，增强后的 messages 长度: {len(messages)}")

        has_environment = d['environments'] and d['environments'][0]['name'] != ''
        
        if has_environment:
            envs, tool_descs = parse_envs(d['environments'])

            for item in tool_descs:
                item['type'] = "object"

            for i, item in enumerate(tool_descs):
                newitem = {'type': 'function', 'function': item}
                tool_descs[i] = newitem

        while rounds < max_rounds:
            if rounds > 8:
                print(f'current round: {rounds}, messages: {messages}')
            else:
                print(f'id {d["id"]}, current round: {rounds}')

            # 对于没有 environment 的数据项，不传递 tools（传 None）
            tools_to_use = tool_descs if has_environment else None
            completion = agent_api.generate_response(messages, tools_to_use)

            rounds += 1

            if completion is None or 'type' not in completion: continue
            if completion['type'] == 'error':
                print(completion['message'])
                continue

            # 对于没有 environment 的数据，如果返回 tool call，则报错
            if completion['type'] == 'tool' and not has_environment:
                error_msg = f"Error: Model attempted tool call for data without environment (id: {d['id']})"
                print(error_msg)
                log_error_samples(messages, info)
                return None, rounds

            if completion['type'] == 'tool':
                tool_call_id = completion['tool_call_id']
                tool_name = completion['tool_name']
                arguments = completion['arguments']

                tool_call_result = None
                for env_idx, env in enumerate(envs):
                    provided_tools = d['environments'][env_idx]['tools']
                    if tool_name not in provided_tools: continue
                    if not env.has_tool(tool_name): continue
                    tool_call_result = env.call_tool(tool_name, deepcopy(arguments))
                    break

                if tool_call_result is None:
                    # raise ValueError(f"Invalid tool call {tool_name}.")
                    # print(f"Invalid tool call {tool_name}.")
                    tool_call_result = {"success": False, "message": f"Tool {tool_name} doesn't exist."}
                    # continue

                # print('tool call result: ', tool_call_result)

                tool_call_response = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "arguments": json.dumps(arguments),
                                "name": tool_name
                            }
                        }
                    ]
                }

                tool_call_result_message = {
                    "role": "tool",
                    "content": json.dumps(tool_call_result),
                    "tool_call_id": tool_call_id,
                    "name": tool_name
                }

                messages.append(tool_call_response)
                messages.append(tool_call_result_message)

            ## normal content part
            else:
                content = completion['content']
                messages.append({
                    'role': 'assistant',
                    'content': content
                })
                # if stop_token not in content: continue
                final_result = messages
                break

    except Exception as e:
        print(e)
        import traceback; traceback.print_exc();
        log_error_samples(messages, info)
        return None, rounds

    if final_result is None:
        if rounds == 10:
            final_result = messages
            log_exceed_round_samples(messages, info)
        else:
            log_error_samples(messages, info)
    else:
        if not args.allow_empty and (final_result[-1]['role'] != 'assistant' or final_result[-1]['content'] == ''):
            log_error_samples(messages, info)
            return None, rounds
            
    return final_result, rounds

def eval_file(path, outpath):
    basedir = os.path.dirname(outpath)
    os.makedirs(basedir, exist_ok=True)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if os.path.exists(outpath):
        try:
            with open(outpath, 'r', encoding='utf-8') as f:
                outdata = json.load(f)
        except Exception as e:
            print(e)
            outdata = []
    else:
        outdata = []
    
    _outdata = []
    for d in outdata:
        if 'label' in d and d['label'] == -3:
            # changed samples should be re-cd
            continue
        _outdata.append(d)
    outdata = _outdata
    
    outids = set([d['id'] for d in outdata])
    _data = []
    print("data: ", len(outids))

    # fail_id = [1280, 1287, 1467, 1489]
    # # 取fail_id的数据
    # data = [d for d in data if d['id'] in fail_id]
    for d in data:
        if d['id'] in outids:
            continue
        if 'finish' in d:
            if d['finish'] != 1:
                continue
        _data.append(d)
        
    data = _data
    success_count = 0
    fail_count = 0
    exceed_count = 0
    
    # 用于线程安全的锁
    lock = threading.Lock()
    
    # 处理单个数据项的包装函数
    def process_item(d):
        nonlocal success_count, fail_count, exceed_count, outdata
        
        opt, rounds = evaluate(d)
        
        with lock:
            if opt is not None:
                success_count += 1
                if rounds == 10:
                    print(f'Exceed max rounds! id: {d["id"]}')
                    exceed_count += 1
                
                d['output'] = opt
                outdata.append(d)

                # 写入文件
                with open(outpath, 'w', encoding='utf-8') as fw:
                    json.dump(outdata, fw, indent=2, ensure_ascii=False)
                
                return ('success', d['id'], rounds)
            else:
                fail_count += 1
                print(f'Fail! id: {d["id"]}')
                return ('fail', d['id'], rounds)
    
    # 使用 ThreadPoolExecutor 进行并发处理
    max_workers = args.max_workers
    print(f'使用 {max_workers} 个并发线程进行处理')
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_item, d): d for d in data}
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(data)):
            try:
                result = future.result()
                # result 格式: ('success'/'fail', id, rounds)
            except Exception as e:
                d = futures[future]
                print(f'处理数据项 {d["id"]} 时发生异常: {e}')
                import traceback
                traceback.print_exc()
    
    print(f'success count: {success_count} (exceed count: {exceed_count}), fail count: {fail_count}')


if __name__ == '__main__':

    path = "../data/released_data.json"
    outdir = f"./evaluation_results/tot-{model_name}/{args.memory_system_type if args.memory_system_type else 'no_memory'}"

    outdir += f'{args.extra_info}'
    outpath = f"{outdir}/gen_res.json"
    error_log_path = f"{outdir}/error_samples.json"
    exceed_log_path = f"{outdir}/exceed_round_samples.json"
    log_path = f"{outdir}/gen_log.txt"
    
    os.makedirs(outdir, exist_ok=True)

    redirect_output(log_path, 'w')
    regenerate_exceed_samples = args.regen_exceed
    if regenerate_exceed_samples:
        remove_exceed_samples(outpath)
        clear_log(clear_error=True, clear_exceed=True)
    else:
        clear_log(clear_error=True, clear_exceed=False)
    eval_file(path, outpath)
